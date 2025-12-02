"""High coverage tests for the codegen module and AST transformer."""

from __future__ import annotations

import ast
from typing import Any, Callable

import pytest

from src.codegen.generator import _BASE_IMPORTS, _CallMappingTransformer, CodeGenerator
from src.frontend.parser import ConstexprMetadata, ParsedKernel
from triton.runtime import jit as triton_jit
import triton.language.core as tlc


@pytest.fixture
def find_call_key() -> Callable[[ParsedKernel, Callable[[ast.Call], bool]], str]:
    """Return a helper that extracts ``function:lineno:col`` keys for call nodes."""

    def _finder(parsed_kernel: ParsedKernel, predicate: Callable[[ast.Call], bool]) -> str:
        func_node = next(node for node in parsed_kernel.tree.body if isinstance(node, ast.FunctionDef))
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call) and predicate(node):
                return f"{func_node.name}:{node.lineno}:{node.col_offset}"
        raise AssertionError("No matching call node found")

    return _finder


class TestCodeGenerator:
    def test_generate_single_kernel_basic(
        self,
        code_generator: CodeGenerator,
        parsed_kernel_factory: Callable[..., ParsedKernel],
        ttgir_output_factory: Callable[..., Any],
    ) -> None:
        """A kernel with no callees still gains the base imports and decorator."""

        parsed = parsed_kernel_factory(
            """
            def root_kernel(x):
                return x + 1
            """
        )
        module = code_generator.generate(parsed, ttgir_output_factory())
        assert [kernel.name for kernel in module.kernels] == ["root_kernel"]
        assert module.constexprs == []
        for base in _BASE_IMPORTS:
            assert base in module.imports
        assert "@gluon.constexpr_function" in module.kernels[0].source

    def test_generate_multi_kernel_bfs_and_nested_call_graph(
        self,
        fake_kernel_parser,
        code_generator_factory: Callable[[], CodeGenerator],
        parsed_kernel_factory: Callable[..., ParsedKernel],
        call_dependency_factory: Callable[[object], Any],
        ttgir_output_factory: Callable[..., Any],
    ) -> None:
        """Nested kernels discovered during transformation join the BFS queue."""

        child_c = parsed_kernel_factory(
            """
            def child_c(x):
                return x - 1
            """,
            func_name="child_c",
            is_jit=True,
        )
        child_b = parsed_kernel_factory(
            """
            def child_b(x):
                return child_c(x) * 2
            """,
            func_name="child_b",
            globals_extra={"child_c": child_c.kernel},
            is_jit=True,
        )
        child_b.call_graph = {
            next(iter(child_b.call_graph)): [call_dependency_factory(child_c.kernel)]
        }
        child_a = parsed_kernel_factory(
            """
            def child_a(x):
                return child_b(x) + 3
            """,
            func_name="child_a",
            globals_extra={"child_b": child_b.kernel},
            is_jit=True,
        )
        child_a.call_graph = {
            next(iter(child_a.call_graph)): [call_dependency_factory(child_b.kernel)]
        }
        nested_kernel = parsed_kernel_factory(
            """
            def nested_child(x):
                return x * 4
            """,
            func_name="nested_child",
            is_jit=True,
        )
        root = parsed_kernel_factory(
            """
            def root_kernel(x):
                return nested_child(x) + x
            """,
            globals_extra={"nested_child": nested_kernel.kernel},
        )
        root.call_graph = {
            next(iter(root.call_graph)): [call_dependency_factory(child_a.kernel)]
        }

        for parsed_kernel in (child_a, child_b, child_c, nested_kernel):
            fake_kernel_parser.register(parsed_kernel)

        module = code_generator_factory().generate(root, ttgir_output_factory())
        assert [kernel.name for kernel in module.kernels] == [
            "root_kernel",
            "child_a",
            "nested_child",
            "child_b",
            "child_c",
        ]
        assert module.diagnostics == []

    def test_constexpr_reconstruction_and_fallback(
        self,
        code_generator: CodeGenerator,
        parsed_kernel_factory: Callable[..., ParsedKernel],
        ttgir_output_factory: Callable[..., Any],
    ) -> None:
        """Constexpr metadata is rewritten to ttgl.* and falls back to repr values."""

        constexpr_entries = (
            ConstexprMetadata(
                name="SOURCE_CONST",
                module="tests.codegen",
                assignment="SOURCE_CONST = tl.arange(0, 4)",
            ),
            ConstexprMetadata(name="FALLBACK_CONST", module=None, assignment=None),
        )
        globals_map = {
            "SOURCE_CONST": tlc.constexpr(4),
            "FALLBACK_CONST": tlc.constexpr(8),
            "MISSING_CONST": tlc.constexpr(16),
        }
        parsed = parsed_kernel_factory(
            """
            def root_kernel(x):
                return SOURCE_CONST + FALLBACK_CONST + MISSING_CONST
            """,
            globals_extra=globals_map,
            constexpr_metadata=constexpr_entries,
        )
        module = code_generator.generate(parsed, ttgir_output_factory())
        assert any("ttgl.arange" in line for line in module.constexprs)
        assert any(line.startswith("FALLBACK_CONST =") for line in module.constexprs)
        assert "Missing constexpr metadata for 'MISSING_CONST'." in module.diagnostics

    def test_layout_metadata_serialization_for_blocked_and_slice(
        self,
        code_generator: CodeGenerator,
        parsed_kernel_factory: Callable[..., ParsedKernel],
        ttgir_output_factory: Callable[..., Any],
        layouts_module,
        find_call_key: Callable[..., str],
    ) -> None:
        """Layout dataclasses become ttgl.* constructor calls in the emitted AST."""

        parsed = parsed_kernel_factory(
            """
            def root_kernel(tensor):
                data = tl.arange(0, 8, 2)
                return tensor[None, 0] + data
            """,
        )
        blocked_layout = layouts_module.BlockedLayout(shape=(4, 2), order=(1, 0))
        slice_layout = layouts_module.SliceLayout(
            dim=0,
            parent=layouts_module.SharedLayout(axes=(0, 1)),
        )
        arange_key = find_call_key(parsed, lambda node: getattr(node.func, "attr", "") == "arange")
        func_node = next(node for node in parsed.tree.body if isinstance(node, ast.FunctionDef))
        slice_node = next(node for node in ast.walk(func_node) if isinstance(node, ast.Subscript))
        slice_key = f"{func_node.name}:{slice_node.lineno}:{slice_node.col_offset}"
        ttgir_output = ttgir_output_factory(
            layouts={
                arange_key: {"layout": blocked_layout},
                slice_key: slice_layout,
            }
        )

        module = code_generator.generate(parsed, ttgir_output)
        source = module.kernels[0].source
        assert "tl_arange" in source and "layout=ttgl.BlockedLayout" in source
        assert "ttgl.SliceLayout" in source
        assert module.diagnostics == []

    def test_slice_layout_default_when_metadata_missing(
        self,
        code_generator: CodeGenerator,
        parsed_kernel_factory: Callable[..., ParsedKernel],
        ttgir_output_factory: Callable[..., Any],
    ) -> None:
        """Broadcast subscripts fall back to default layouts when metadata is absent."""

        parsed = parsed_kernel_factory(
            """
            def root_kernel(tensor):
                return tensor[None, 1]
            """,
        )
        module = code_generator.generate(parsed, ttgir_output_factory())
        assert any("default_blocked_layout" in kernel.source for kernel in module.kernels)
        assert any("Missing SliceLayout metadata" in diag for diag in module.diagnostics)
        assert any(
            line.startswith("from triton.tools.triton_to_gluon_translater.translator_helpers import")
            and "default_blocked_layout" in line
            for line in module.imports
        )

    def test_import_management_includes_helper_and_support_imports(
        self,
        code_generator: CodeGenerator,
        parsed_kernel_factory: Callable[..., ParsedKernel],
        ttgir_output_factory: Callable[..., Any],
        find_call_key: Callable[..., str],
        layouts_module,
    ) -> None:
        """Mapped ops add helper imports while layout wrappers add support imports."""

        parsed = parsed_kernel_factory(
            """
            def root_kernel(tensor):
                x = tl.arange(0, 8)
                y = tl.dot(tensor, tensor)
                z = tl.program_id(0)
                w = tl.reshape(tensor, tensor.shape)
                q = tensor.split(2)
                return x + z + w[0] + q
            """,
        )
        blocked_layout = layouts_module.BlockedLayout(shape=(8,), order=(0,))
        arange_key = find_call_key(parsed, lambda node: getattr(node.func, "attr", "") == "arange")
        dot_key = find_call_key(parsed, lambda node: getattr(node.func, "attr", "") == "dot")
        pid_key = find_call_key(parsed, lambda node: getattr(node.func, "attr", "") == "program_id")
        layouts = {
            arange_key: {"layout": blocked_layout},
            dot_key: {"layout_a": blocked_layout, "layout_b": blocked_layout},
            pid_key: {"layout": blocked_layout},
        }
        module = code_generator.generate(parsed, ttgir_output_factory(layouts=layouts))
        helper_imports = {
            "from src.mapping.functions.arange import tl_arange",
            "from src.mapping.functions.dot import tl_dot",
            "from src.mapping.functions.program_id import tl_program_id",
        }
        support_imports = {
            "from triton.tools.triton_to_gluon_translater.translator_helpers import reset_to_default_layout",
            "from triton.tools.triton_to_gluon_translater.translator_helpers import set_split_src_layout",
        }
        assert helper_imports.issubset(set(module.imports))
        assert support_imports.issubset(set(module.imports))

    def test_jit_kernel_receives_gluon_jit_decorator_and_annotation_rewrites(
        self,
        code_generator: CodeGenerator,
        parsed_kernel_factory: Callable[..., ParsedKernel],
        ttgir_output_factory: Callable[..., Any],
        find_call_key: Callable[..., str],
        layouts_module,
    ) -> None:
        """JIT kernels gain @gluon.jit and tl.constexpr annotations become ttgl.constexpr."""

        parsed = parsed_kernel_factory(
            """
            def root_kernel(x: tl.constexpr, tensor=None):
                return tl.program_id(0)
            """,
            is_jit=True,
        )
        pid_key = find_call_key(parsed, lambda node: getattr(node.func, "attr", "") == "program_id")
        blocked_layout = layouts_module.BlockedLayout(shape=(1,), order=(0,))
        module = code_generator.generate(
            parsed,
            ttgir_output_factory(layouts={pid_key: {"layout": blocked_layout}}),
        )
        kernel_source = module.kernels[0].source
        assert kernel_source.splitlines()[0] == "@gluon.jit"
        assert "x: ttgl.constexpr" in kernel_source

    def test_unmapped_ops_preserve_tl_calls_and_report_diagnostics(
        self,
        code_generator: CodeGenerator,
        parsed_kernel_factory: Callable[..., ParsedKernel],
        ttgir_output_factory: Callable[..., Any],
    ) -> None:
        """Unknown tl.* ops stay untouched and emit an unmapped diagnostic."""

        parsed = parsed_kernel_factory(
            """
            def root_kernel(x):
                return tl.fake_op(x)
            """,
        )
        module = code_generator.generate(parsed, ttgir_output_factory())
        assert "tl.fake_op" in module.kernels[0].source
        assert any("Unmapped Triton op" in diag for diag in module.diagnostics)
        assert "import triton.language as tl" in module.imports

    def test_empty_kernel_body_is_emitted(
        self,
        code_generator: CodeGenerator,
        parsed_kernel_factory: Callable[..., ParsedKernel],
        ttgir_output_factory: Callable[..., Any],
    ) -> None:
        """Kernels with an empty body (pass) emit without errors."""

        parsed = parsed_kernel_factory(
            """
            def root_kernel(x):
                pass
            """
        )
        module = code_generator.generate(parsed, ttgir_output_factory())
        assert "pass" in module.kernels[0].source

    def test_kernel_without_metadata_reports_missing_layout(
        self,
        code_generator: CodeGenerator,
        parsed_kernel_factory: Callable[..., ParsedKernel],
        ttgir_output_factory: Callable[..., Any],
        find_call_key: Callable[..., str],
    ) -> None:
        """Missing TTGIR layout metadata surfaces a diagnostic."""

        parsed = parsed_kernel_factory(
            """
            def root_kernel(tensor):
                return tl.dot(tensor, tensor)
            """
        )
        module = code_generator.generate(parsed, ttgir_output_factory())
        assert any("Missing TTGIR metadata" in diag for diag in module.diagnostics)

    def test_circular_call_graph_is_handled_without_duplicates(
        self,
        fake_kernel_parser,
        code_generator_factory: Callable[[], CodeGenerator],
        parsed_kernel_factory: Callable[..., ParsedKernel],
        call_dependency_factory: Callable[[object], Any],
        ttgir_output_factory: Callable[..., Any],
    ) -> None:
        """Call graphs with cycles terminate thanks to the visited set."""

        callee = parsed_kernel_factory(
            """
            def callee(x):
                return x
            """,
            func_name="callee",
        )
        root = parsed_kernel_factory(
            """
            def root_kernel(x):
                return callee(x)
            """,
            globals_extra={"callee": callee.kernel},
        )
        root_key = next(iter(root.call_graph))
        callee_key = next(iter(callee.call_graph))
        root.call_graph = {root_key: [call_dependency_factory(callee.kernel)]}
        callee.call_graph = {callee_key: [call_dependency_factory(root.kernel)]}
        fake_kernel_parser.register(callee)
        module = code_generator_factory().generate(root, ttgir_output_factory())
        assert [kernel.name for kernel in module.kernels] == ["root_kernel", "callee"]

    def test_invalid_layout_object_raises_type_error(
        self,
        code_generator: CodeGenerator,
        parsed_kernel_factory: Callable[..., ParsedKernel],
        ttgir_output_factory: Callable[..., Any],
        find_call_key: Callable[..., str],
        layouts_module,
    ) -> None:
        """Non-dataclass layout metadata triggers a TypeError for easier debugging."""

        parsed = parsed_kernel_factory(
            """
            def root_kernel():
                return tl.arange(0, 4)
            """
        )
        call_key = find_call_key(parsed, lambda node: getattr(node.func, "attr", "") == "arange")
        invalid_layout = layouts_module.BlockedLayout(shape={1, 2}, order=(0,))
        ttgir = ttgir_output_factory(layouts={call_key: {"layout": invalid_layout}})
        with pytest.raises(TypeError):
            code_generator.generate(parsed, ttgir)

    def test_full_pipeline_output_parses(
        self,
        code_generator: CodeGenerator,
        parsed_kernel_factory: Callable[..., ParsedKernel],
        ttgir_output_factory: Callable[..., Any],
        find_call_key: Callable[..., str],
        layouts_module,
    ) -> None:
        """Gluon modules can be unparsed and re-parsed as valid Python."""

        parsed = parsed_kernel_factory(
            """
            def root_kernel(tensor, value: tl.constexpr = 2):
                data = tl.arange(0, 8)
                pid = tl.program_id(0)
                reshaped = tensor.reshape(tensor.shape)
                return data + pid + reshaped
            """,
            is_jit=True,
        )
        arange_key = find_call_key(parsed, lambda node: getattr(node.func, "attr", "") == "arange")
        pid_key = find_call_key(parsed, lambda node: getattr(node.func, "attr", "") == "program_id")
        blocked_layout = layouts_module.BlockedLayout(shape=(8,), order=(0,))
        ttgir = ttgir_output_factory(
            layouts={
                arange_key: {"layout": blocked_layout},
                pid_key: {"layout": blocked_layout},
            }
        )
        module = code_generator.generate(parsed, ttgir)
        module_source = ast.unparse(module.module_ast)
        compile(module_source, "<generated>", "exec")
        assert module.imports[0] == _BASE_IMPORTS[0]


class TestCallMappingTransformer:
    def _build_transformer(
        self,
        source: str,
        parsed_kernel_factory: Callable[..., ParsedKernel],
        ttgir_output_factory: Callable[..., Any],
        mapping_registry_factory: Callable[[], Any],
        *,
        is_jit: bool = False,
        layouts: dict[str, Any] | None = None,
        spec_matches: dict[str, Any] | None = None,
        globals_extra: dict[str, Any] | None = None,
    ) -> tuple[_CallMappingTransformer, ParsedKernel]:
        parsed = parsed_kernel_factory(source, is_jit=is_jit, globals_extra=globals_extra)
        registry = mapping_registry_factory()
        transformer = _CallMappingTransformer(
            parsed_kernel=parsed,
            ttgir_output=ttgir_output_factory(layouts=layouts or {}, spec_matches=spec_matches or {}),
            registry=registry,
            is_jit=is_jit,
        )
        return transformer, parsed

    def test_visit_call_rewrites_registered_ops(
        self,
        parsed_kernel_factory: Callable[..., ParsedKernel],
        ttgir_output_factory: Callable[..., Any],
        mapping_registry_factory: Callable[[], Any],
        find_call_key: Callable[..., str],
        layouts_module,
    ) -> None:
        """Mapped Triton ops become helper calls with metadata keywords."""

        key_kernel = parsed_kernel_factory(
            """
            def root_kernel(a):
                x = tl.arange(0, 4)
                y = tl.dot(a, a)
                return tl.program_id(0) + x + y
            """
        )
        blocked_layout = layouts_module.BlockedLayout(shape=(4,), order=(0,))
        arange_key = find_call_key(key_kernel, lambda node: getattr(node.func, "attr", "") == "arange")
        dot_key = find_call_key(key_kernel, lambda node: getattr(node.func, "attr", "") == "dot")
        pid_key = find_call_key(key_kernel, lambda node: getattr(node.func, "attr", "") == "program_id")
        layouts = {
            arange_key: {"layout": blocked_layout},
            dot_key: {"layout_a": blocked_layout, "layout_b": blocked_layout},
            pid_key: {"layout": blocked_layout},
        }
        transformer, parsed_kernel = self._build_transformer(
            """
            def root_kernel(a):
                x = tl.arange(0, 4)
                y = tl.dot(a, a)
                return tl.program_id(0) + x + y
            """,
            parsed_kernel_factory,
            ttgir_output_factory,
            mapping_registry_factory,
            layouts=layouts,
        )
        transformed = transformer.visit(ast.fix_missing_locations(parsed_kernel.tree))
        text = ast.unparse(transformed)
        assert "tl_arange" in text and "layout=ttgl.BlockedLayout" in text
        assert "tl_dot" in text and "layout_a" in text and "layout_b" in text
        assert "tl_program_id" in text and "layout=" in text
        assert transformer.helper_imports  # helper imports collected

    def test_visit_call_unmapped_ops_leave_tl_calls(
        self,
        parsed_kernel_factory: Callable[..., ParsedKernel],
        ttgir_output_factory: Callable[..., Any],
        mapping_registry_factory: Callable[[], Any],
    ) -> None:
        """Unknown calls stay as tl.* expressions and report diagnostics."""

        transformer, parsed = self._build_transformer(
            """
            def root_kernel(x):
                return tl.unknown(x)
            """,
            parsed_kernel_factory,
            ttgir_output_factory,
            mapping_registry_factory,
        )
        transformed = transformer.visit(ast.fix_missing_locations(parsed.tree))
        assert "tl.unknown" in ast.unparse(transformed)
        assert transformer.diagnostics

    def test_visit_attribute_rewrites_dtype_and_tensor_descriptor(
        self,
        parsed_kernel_factory: Callable[..., ParsedKernel],
        ttgir_output_factory: Callable[..., Any],
        mapping_registry_factory: Callable[[], Any],
    ) -> None:
        """tl.* attribute references map to ttgl equivalents."""

        transformer, parsed = self._build_transformer(
            """
            def root_kernel():
                return tl.float32, tl.tensor_descriptor
            """,
            parsed_kernel_factory,
            ttgir_output_factory,
            mapping_registry_factory,
        )
        transformed = transformer.visit(ast.fix_missing_locations(parsed.tree))
        text = ast.unparse(transformed)
        assert "ttgl.float32" in text
        assert "ttgl.nvidia.hopper.tma.tensor_descriptor" in text

    def test_visit_name_tracks_constexpr_and_nested_kernels(
        self,
        parsed_kernel_factory: Callable[..., ParsedKernel],
        ttgir_output_factory: Callable[..., Any],
        mapping_registry_factory: Callable[[], Any],
    ) -> None:
        """Name resolution records constexpr bindings and nested jit kernels."""

        nested = triton_jit.jit(lambda x: x)
        globals_extra = {"CONST": tlc.constexpr(1), "nested_kernel": nested}
        transformer, parsed = self._build_transformer(
            """
            def root_kernel(x):
                return CONST + nested_kernel(x)
            """,
            parsed_kernel_factory,
            ttgir_output_factory,
            mapping_registry_factory,
            globals_extra=globals_extra,
        )
        transformer.visit(ast.fix_missing_locations(parsed.tree))
        assert "CONST" in transformer.referenced_constexpr
        assert nested in transformer.nested_kernels

    def test_visit_function_adds_decorator_and_rewrites_annotations(
        self,
        parsed_kernel_factory: Callable[..., ParsedKernel],
        ttgir_output_factory: Callable[..., Any],
        mapping_registry_factory: Callable[[], Any],
    ) -> None:
        """Function defs acquire gluon decorators and constexpr annotations."""

        transformer, parsed = self._build_transformer(
            """
            def root_kernel(x: tl.constexpr):
                return x
            """,
            parsed_kernel_factory,
            ttgir_output_factory,
            mapping_registry_factory,
            is_jit=True,
        )
        transformed = transformer.visit(ast.fix_missing_locations(parsed.tree))
        fn_node = next(node for node in transformed.body if isinstance(node, ast.FunctionDef))
        decorator_ids = [ast.unparse(dec) for dec in fn_node.decorator_list]
        assert "gluon.jit" in decorator_ids
        assert ast.unparse(fn_node.args.args[0].annotation) == "ttgl.constexpr"

    def test_visit_subscript_wraps_broadcasts_with_convert_layout(
        self,
        parsed_kernel_factory: Callable[..., ParsedKernel],
        ttgir_output_factory: Callable[..., Any],
        mapping_registry_factory: Callable[[], Any],
        layouts_module,
    ) -> None:
        """None-based indexing wraps tensors in ttgl.convert_layout calls."""

        parsed = parsed_kernel_factory(
            """
            def root_kernel(tensor):
                return tensor[None, 0]
            """
        )
        func_node = next(node for node in parsed.tree.body if isinstance(node, ast.FunctionDef))
        subscript = next(node for node in ast.walk(func_node) if isinstance(node, ast.Subscript))
        key = f"{func_node.name}:{subscript.lineno}:{subscript.col_offset}"
        layouts = {
            key: layouts_module.SliceLayout(
                dim=0,
                parent=layouts_module.SharedLayout(axes=(0, 1)),
            )
        }
        transformer, parsed = self._build_transformer(
            """
            def root_kernel(tensor):
                return tensor[None, 0]
            """,
            parsed_kernel_factory,
            ttgir_output_factory,
            mapping_registry_factory,
            layouts=layouts,
        )
        transformed = transformer.visit(ast.fix_missing_locations(parsed.tree))
        assert "ttgl.convert_layout" in ast.unparse(transformed)

    def test_layout_reset_wrappers_wrap_tensor_methods(
        self,
        parsed_kernel_factory: Callable[..., ParsedKernel],
        ttgir_output_factory: Callable[..., Any],
        mapping_registry_factory: Callable[[], Any],
    ) -> None:
        """reshape/split calls get wrapped so layout metadata is preserved."""

        transformer, parsed = self._build_transformer(
            """
            def root_kernel(tensor):
                return tensor.reshape(tensor.shape).split(2)
            """,
            parsed_kernel_factory,
            ttgir_output_factory,
            mapping_registry_factory,
        )
        transformed = transformer.visit(ast.fix_missing_locations(parsed.tree))
        text = ast.unparse(transformed)
        assert "reset_to_default_layout" in text
        assert "set_split_src_layout" in text
        assert any(
            line.startswith("from triton.tools.triton_to_gluon_translater.translator_helpers")
            for line in transformer.support_imports
        )
