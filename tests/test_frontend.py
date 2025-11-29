"""High-coverage tests for the Triton→Gluon frontend parser and scope helpers.

The suite validates KernelParser, call-graph discovery, constexpr metadata
tracking, scope analysis, expression evaluation, and integration workflows.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import inspect
import math
import os
import sys
import textwrap
import types
import uuid
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import pytest


def _install_triton_stubs() -> None:
    """Provide lightweight Triton modules so the frontend can import cleanly."""

    if "triton" in sys.modules:
        return

    def _safe_source(fn: Any) -> str:
        if fn is None:
            return "pass"
        try:
            return inspect.getsource(fn)
        except (OSError, TypeError):
            return "pass"

    triton_mod = types.ModuleType("triton")
    language_mod = types.ModuleType("triton.language")
    core_mod = types.ModuleType("triton.language.core")
    runtime_mod = types.ModuleType("triton.runtime")
    jit_mod = types.ModuleType("triton.runtime.jit")

    class Constexpr(int):
        def __new__(cls, value: int) -> "Constexpr":
            obj = int.__new__(cls, value)
            obj.value = value
            return obj

    Constexpr.__module__ = "triton.language.core"

    class DType(str):
        def __new__(cls, value: str = "") -> "DType":
            return str.__new__(cls, value)

    DType.__module__ = "triton.language.core"

    def _unwrap_if_constexpr(value: Any) -> Any:
        return getattr(value, "value", value)

    core_mod.constexpr = Constexpr
    core_mod.dtype = DType
    core_mod._unwrap_if_constexpr = staticmethod(_unwrap_if_constexpr)

    language_mod.core = core_mod
    language_mod.constexpr = Constexpr
    language_mod.dtype = DType

    class JITCallable:
        def __init__(
            self,
            fn: Any = None,
            *,
            src: Optional[str] = None,
            name: Optional[str] = None,
            module: Optional[str] = None,
            globals_map: Optional[dict[str, Any]] = None,
        ) -> None:
            self.fn = fn
            base_name = name or getattr(fn, "__name__", "jit_callable")
            self.__name__ = base_name
            self.__qualname__ = getattr(fn, "__qualname__", base_name)
            self.__module__ = module or getattr(fn, "__module__", "tests.fake")
            if globals_map is None:
                if getattr(fn, "__globals__", None) is not None:
                    self.__globals__ = fn.__globals__
            else:
                globals_map.setdefault("__name__", self.__module__)
                self.__globals__ = globals_map
            self.src = src if src is not None else _safe_source(fn)
            self._src = self.src

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            if callable(self.fn):
                return self.fn(*args, **kwargs)
            raise RuntimeError("Stub JITCallable is not executable")

        def __getitem__(self, _grid: Any) -> "JITCallable":
            return self

    JITCallable.__module__ = "triton.runtime.jit"

    class JITFunction(JITCallable):
        pass

    JITFunction.__module__ = "triton.runtime.jit"

    def jit(func: Optional[Callable[..., Any]] = None, *, name: Optional[str] = None):
        def decorator(target: Callable[..., Any]) -> JITFunction:
            return JITFunction(target, name=name)

        if func is None:
            return decorator
        return decorator(func)

    jit_mod.JITCallable = JITCallable
    jit_mod.JITFunction = JITFunction
    jit_mod.jit = jit

    runtime_mod.jit = jit_mod
    triton_mod.runtime = runtime_mod
    triton_mod.language = language_mod

    sys.modules["triton"] = triton_mod
    sys.modules["triton.language"] = language_mod
    sys.modules["triton.language.core"] = core_mod
    sys.modules["triton.runtime"] = runtime_mod
    sys.modules["triton.runtime.jit"] = jit_mod


_install_triton_stubs()

from src.frontend import (  # noqa: E402  (tests need stubs before import)
    CallDependency,
    ConstexprMetadata,
    KernelParser,
    ParsedKernel,
    ScopeAnalyzer,
    SymbolBinding,
    SymbolTable,
    SymbolType,
)
from triton.runtime import jit as triton_jit
import triton.language.core as tlc


def _write_module_file(root: Path, module_name: str, source: str) -> Path:
    """Create ``module_name`` under ``root`` with ``source`` and return the path."""

    parts = module_name.split(".")
    target_dir = root
    for part in parts[:-1]:
        target_dir /= part
        target_dir.mkdir(exist_ok=True)
        init_file = target_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("")
    module_path = target_dir / f"{parts[-1]}.py"
    module_path.write_text(textwrap.dedent(source))
    return module_path


def _call_graph_key(kernel: object) -> str:
    """Mirror the parser's qualified-name key to avoid accidental collisions in tests."""

    base = getattr(kernel, "fn", kernel)
    simple_name = getattr(base, "__name__", getattr(base, "__qualname__", repr(base)))
    qual_component = getattr(base, "__qualname__", simple_name)
    module_name = getattr(base, "__module__", None)
    qualified = ".".join(part for part in (module_name, qual_component) if part)
    return qualified or simple_name


@pytest.fixture
def kernel_parser() -> KernelParser:
    """Provide a fresh KernelParser per test."""

    return KernelParser()


@pytest.fixture
def scope_analyzer() -> ScopeAnalyzer:
    """Provide an isolated ScopeAnalyzer per test."""

    return ScopeAnalyzer()


@pytest.fixture
def fake_kernel_factory() -> Callable[..., triton_jit.JITFunction]:
    """Factory that creates stub JIT kernels with controllable globals."""

    def _factory(
        name: str,
        body: str,
        *,
        module_name: str = "tests.fake",
        globals_map: Optional[dict[str, Any]] = None,
    ) -> triton_jit.JITFunction:
        mapping = globals_map if globals_map is not None else {}
        if globals_map is None:
            mapping = {"__name__": module_name}
        else:
            mapping.setdefault("__name__", module_name)
        fn_stub = types.SimpleNamespace(
            __name__=name,
            __qualname__=name,
            __module__=module_name,
            __globals__=mapping,
        )
        source = textwrap.dedent(body)
        return triton_jit.JITFunction(fn_stub, src=source, module=module_name, globals_map=mapping)

    return _factory


@pytest.fixture
def module_builder(tmp_path: Path) -> Iterable[Callable[[str, str], tuple[types.ModuleType, Path]]]:
    """Helper that writes temporary modules and imports them for metadata testing."""

    created_modules: list[str] = []
    sys_path_added = False
    tmp_path_str = str(tmp_path)
    if tmp_path_str not in sys.path:
        sys.path.insert(0, tmp_path_str)
        sys_path_added = True

    def _builder(module_name: str, source: str) -> tuple[types.ModuleType, Path]:
        module_path = _write_module_file(tmp_path, module_name, source)
        importlib.invalidate_caches()
        if module_name in sys.modules:
            del sys.modules[module_name]
        module = importlib.import_module(module_name)
        created_modules.append(module_name)
        return module, module_path

    yield _builder

    for module_name in created_modules:
        sys.modules.pop(module_name, None)
    if sys_path_added:
        with contextlib.suppress(ValueError):
            sys.path.remove(tmp_path_str)


def _evaluate_assignment(expr_src: str, globals_map: Optional[dict[str, Any]] = None) -> Any:
    """Analyze ``value = expr_src`` and return the evaluated binding."""

    tree = ast.parse(f"value = {expr_src}")
    analyzer = ScopeAnalyzer()
    base_globals: dict[str, Any] = {"int": int, "float": float, "bool": bool, "str": str}
    if globals_map:
        base_globals.update(globals_map)
    table = analyzer.analyze(tree, globals_map=base_globals)
    return table.lookup("value")


def _regular_reference_kernel(x):
    return x + 1


def _jit_reference_kernel(x):
    return x * 2


def _dataclass_reference_kernel(x):
    return x


# --- KernelParser tests -------------------------------------------------------


def test_kernel_parser_regular_function_captures_source_and_globals(kernel_parser: KernelParser) -> None:
    """Basic regression: ensure source, filename, globals, and AST are preserved."""

    parsed = kernel_parser.parse_kernel(_regular_reference_kernel)
    assert parsed.kernel is _regular_reference_kernel
    assert parsed.source == inspect.getsource(_regular_reference_kernel)
    assert parsed.filename == inspect.getsourcefile(_regular_reference_kernel)
    assert parsed.globals_map is _regular_reference_kernel.__globals__
    assert isinstance(parsed.tree, ast.AST)
    assert parsed.call_graph[_call_graph_key(_regular_reference_kernel)] == []
    assert parsed.name == _regular_reference_kernel.__name__


def test_kernel_parser_jit_function_uses_fn_globals_fallback(kernel_parser: KernelParser) -> None:
    """JITFunction instances should pull globals from the wrapped python function."""

    jit_kernel = triton_jit.JITFunction(_jit_reference_kernel)
    parsed = kernel_parser.parse_kernel(jit_kernel)
    assert parsed.kernel is jit_kernel
    assert parsed.source == jit_kernel.src
    assert parsed.filename is None  # stub JITFunction has no file metadata
    assert parsed.globals_map is _jit_reference_kernel.__globals__
    assert parsed.call_graph[_call_graph_key(jit_kernel)] == []


def test_kernel_parser_missing_source_returns_stub_ast(kernel_parser: KernelParser) -> None:
    """Fallback path should tolerate kernels that have no introspectable source."""

    class AnonymousKernel:
        __name__ = "anonymous_kernel"

        def __call__(self):
            return 0

    kernel = AnonymousKernel()
    parsed = kernel_parser.parse_kernel(kernel)
    assert parsed.source == ""
    assert parsed.filename is None
    assert parsed.globals_map is None
    assert parsed.call_graph[_call_graph_key(kernel)] == []
    assert isinstance(parsed.tree.body[0], ast.Pass)


def test_parsed_kernel_dataclass_fields_and_name_property(kernel_parser: KernelParser) -> None:
    """ParsedKernel should remain a dataclass with the agreed-upon fields."""

    parsed = kernel_parser.parse_kernel(_dataclass_reference_kernel)
    assert is_dataclass(ParsedKernel)
    assert [field.name for field in fields(ParsedKernel)] == [
        "kernel",
        "tree",
        "source",
        "filename",
        "globals_map",
        "call_graph",
        "constexpr_metadata",
    ]
    assert parsed.name == _dataclass_reference_kernel.__name__


# --- Call graph discovery tests ----------------------------------------------


def test_call_graph_kernel_without_dependencies(kernel_parser: KernelParser, fake_kernel_factory: Callable[..., Any]) -> None:
    """A kernel with no Triton calls should have an empty adjacency list."""

    kernel = fake_kernel_factory(
        "root",
        """
        def root(x):
            return x + 1
        """,
    )
    graph = kernel_parser.parse_kernel(kernel).call_graph
    root_key = _call_graph_key(kernel)
    assert graph[root_key] == []


def test_call_graph_records_single_level_dependency(kernel_parser: KernelParser, fake_kernel_factory: Callable[..., Any]) -> None:
    """A single child call should emit exactly one CallDependency."""

    child = fake_kernel_factory(
        "child",
        """
        def child(x):
            return x - 1
        """,
    )
    root_scope = {"child": child}
    root = fake_kernel_factory(
        "root",
        """
        def root(x):
            return child(x)
        """,
        globals_map=root_scope,
    )
    deps = kernel_parser.parse_kernel(root).call_graph[_call_graph_key(root)]
    assert len(deps) == 1
    dependency = deps[0]
    assert dependency.name == "child"
    assert dependency.module == child.__module__
    assert dependency.obj is child


def test_call_graph_builds_transitive_dependencies(kernel_parser: KernelParser, fake_kernel_factory: Callable[..., Any]) -> None:
    """Breadth-first traversal should materialize multi-hop callees (regression)."""

    leaf = fake_kernel_factory(
        "leaf",
        """
        def leaf(x):
            return x
        """,
    )
    mid_scope = {"leaf": leaf}
    mid = fake_kernel_factory(
        "mid",
        """
        def mid(x):
            return leaf(x)
        """,
        globals_map=mid_scope,
    )
    root_scope = {"mid": mid}
    root = fake_kernel_factory(
        "root",
        """
        def root(x):
            temp = mid(x)
            return temp
        """,
        globals_map=root_scope,
    )
    graph = kernel_parser.parse_kernel(root).call_graph
    root_key = _call_graph_key(root)
    mid_key = _call_graph_key(mid)
    leaf_key = _call_graph_key(leaf)
    assert root_key in graph and graph[root_key][0].obj is mid
    assert mid_key in graph and graph[mid_key][0].obj is leaf
    assert graph[leaf_key] == []


def test_call_graph_handles_kernel_launch_syntax(kernel_parser: KernelParser, fake_kernel_factory: Callable[..., Any]) -> None:
    """Launch expressions kernel[grid](...) must still record the base callable."""

    callee = fake_kernel_factory(
        "launch_target",
        """
        def launch_target(x):
            return x
        """,
    )
    scope = {"launch_target": callee}
    root = fake_kernel_factory(
        "root",
        """
        def root(x):
            return launch_target[0](x)
        """,
        globals_map=scope,
    )
    deps = kernel_parser.parse_kernel(root).call_graph[_call_graph_key(root)]
    assert len(deps) == 1 and deps[0].obj is callee


def test_call_graph_records_multiple_dependencies_without_duplicates(
    kernel_parser: KernelParser, fake_kernel_factory: Callable[..., Any]
) -> None:
    """Calling distinct kernels multiple times should deduplicate per callee."""

    left = fake_kernel_factory(
        "branch",
        """
        def branch(x):
            return x
        """,
    )
    right = fake_kernel_factory(
        "branch",
        """
        def branch(x):
            return x * 2
        """,
        module_name="tests.alt",
    )
    scope = {"left": left, "right": right}
    root = fake_kernel_factory(
        "root",
        """
        def root(x):
            left(x)
            right(x)
            left(x)
            return x
        """,
        globals_map=scope,
    )
    deps = kernel_parser.parse_kernel(root).call_graph[_call_graph_key(root)]
    assert {dep.obj for dep in deps} == {left, right}


def test_call_graph_prevents_infinite_loops_for_circular_dependencies(
    kernel_parser: KernelParser, fake_kernel_factory: Callable[..., Any]
) -> None:
    """Guard against previously observed infinite loops when A→B→A."""

    kernel_a = fake_kernel_factory(
        "kernel_a",
        """
        def kernel_a(x):
            return kernel_b(x)
        """,
    )
    kernel_b = fake_kernel_factory(
        "kernel_b",
        """
        def kernel_b(x):
            return kernel_a(x)
        """,
    )
    kernel_a.fn.__globals__["kernel_b"] = kernel_b
    kernel_b.fn.__globals__["kernel_a"] = kernel_a
    graph = kernel_parser.parse_kernel(kernel_a).call_graph
    kernel_a_key = _call_graph_key(kernel_a)
    kernel_b_key = _call_graph_key(kernel_b)
    assert graph[kernel_a_key][0].obj is kernel_b
    assert graph[kernel_b_key][0].obj is kernel_a


def test_call_graph_preserves_qualified_names_for_colliding_functions(
    kernel_parser: KernelParser, fake_kernel_factory: Callable[..., Any]
) -> None:
    """Two callees with the same simple name must coexist via qualified names."""

    alpha = fake_kernel_factory(
        "shared",
        """
        def shared(x):
            return x
        """,
        module_name="pkg.alpha",
    )
    beta = fake_kernel_factory(
        "shared",
        """
        def shared(x):
            return x * 2
        """,
        module_name="pkg.beta",
    )
    root = fake_kernel_factory(
        "root",
        """
        def root(x):
            alpha_alias(x)
            beta_alias(x)
        """,
        globals_map={"alpha_alias": alpha, "beta_alias": beta},
    )
    deps = kernel_parser.parse_kernel(root).call_graph[_call_graph_key(root)]
    assert len(deps) == 2
    qualified = {dep.qualified_name for dep in deps}
    assert qualified == {f"{alpha.__module__}.shared", f"{beta.__module__}.shared"}


def test_call_graph_qualified_names_prevent_collision_across_modules(
    kernel_parser: KernelParser,
    fake_kernel_factory: Callable[..., Any],
    module_builder: Callable[[str, str], tuple[types.ModuleType, Path]],
) -> None:
    """Two kernels with same name from different modules should both appear in call graph."""

    alpha_module, _ = module_builder(
        "pkg.alpha",
        """
        from triton.runtime.jit import jit

        @jit
        def shared(x):
            return x + 1
        """,
    )
    beta_module, _ = module_builder(
        "pkg.beta",
        """
        from triton.runtime.jit import jit

        @jit
        def shared(x):
            return x * 2
        """,
    )
    root = fake_kernel_factory(
        "root",
        """
        def root(x):
            alpha_shared(x)
            beta_shared(x)
            return x
        """,
        globals_map={"alpha_shared": alpha_module.shared, "beta_shared": beta_module.shared},
    )
    parsed = kernel_parser.parse_kernel(root)
    graph = parsed.call_graph
    root_key = _call_graph_key(root)
    alpha_key = _call_graph_key(alpha_module.shared)
    beta_key = _call_graph_key(beta_module.shared)

    assert root_key in graph
    assert alpha_key in graph
    assert beta_key in graph

    qualified_children = {dep.qualified_name for dep in graph[root_key]}
    assert qualified_children == {alpha_key, beta_key}
    assert graph[alpha_key] == []
    assert graph[beta_key] == []


def test_call_dependency_dataclass_captured_fields(kernel_parser: KernelParser, fake_kernel_factory: Callable[..., Any]) -> None:
    """CallDependency instances must expose the expected dataclass fields."""

    callee = fake_kernel_factory(
        "callee",
        """
        def callee(x):
            return x
        """,
    )
    root = fake_kernel_factory(
        "root",
        """
        def root(x):
            return callee(x)
        """,
        globals_map={"callee": callee},
    )
    dependency = kernel_parser.parse_kernel(root).call_graph[_call_graph_key(root)][0]
    assert is_dataclass(CallDependency)
    assert [field.name for field in fields(CallDependency)] == ["name", "qualified_name", "module", "obj"]
    assert dependency.obj is callee
    assert dependency.module == callee.__module__


# --- Constexpr metadata tests -------------------------------------------------


def test_constexpr_metadata_captures_runtime_assignment(module_builder, kernel_parser: KernelParser) -> None:
    """Runtime tl.constexpr values should be attributed to the kernel's module."""

    module_name = f"pkg_runtime_{uuid.uuid4().hex}"
    module, module_path = module_builder(
        module_name,
        """
        import triton.language as tl

        BLOCK = tl.constexpr(128)

        def runtime_kernel(x):
            return x + BLOCK
        """,
    )
    parsed = kernel_parser.parse_kernel(module.runtime_kernel)
    metadata = {entry.name: entry for entry in parsed.constexpr_metadata}
    entry = metadata["BLOCK"]
    assert os.path.abspath(entry.module) == os.path.abspath(str(module_path))
    assert entry.assignment == "BLOCK = tl.constexpr(128)"


def test_constexpr_metadata_captures_annotated_assignment(module_builder, kernel_parser: KernelParser) -> None:
    """Annotated constexpr bindings must be collected before runtime evaluation."""

    module_name = f"pkg_annotated_{uuid.uuid4().hex}"
    module, module_path = module_builder(
        module_name,
        """
        import triton.language as tl

        BLOCK: tl.constexpr = 64

        def annotated_kernel(x):
            BLOCK: tl.constexpr = BLOCK
            return BLOCK + x
        """,
    )
    parsed = kernel_parser.parse_kernel(module.annotated_kernel)
    metadata = {entry.name: entry for entry in parsed.constexpr_metadata}
    entry = metadata["BLOCK"]
    assert entry.name == "BLOCK"
    assert entry.assignment == "BLOCK: tl.constexpr = 64"
    assert os.path.abspath(entry.module) == os.path.abspath(str(module_path))


def test_constexpr_metadata_records_parameter_annotations_for_all_arg_types(
    module_builder, kernel_parser: KernelParser
) -> None:
    """Parameters annotated as constexpr should be recorded for arg, kwonly, vararg, and kwarg."""

    module_name = f"pkg_params_{uuid.uuid4().hex}"
    module_builder(
        module_name,
        """
        import triton.language as tl

        def parameter_kernel(
            x,
            block: tl.constexpr,
            *varargs: tl.constexpr,
            kwonly: tl.constexpr,
            **kwparams: tl.constexpr,
        ):
            return block if kwonly else kwparams.get("fallback", 0)
        """,
    )
    module = importlib.import_module(module_name)
    parsed = kernel_parser.parse_kernel(module.parameter_kernel)
    names = {entry.name for entry in parsed.constexpr_metadata}
    assert {"block", "varargs", "kwonly", "kwparams"}.issubset(names)


def test_constexpr_metadata_traces_absolute_imports(module_builder, kernel_parser: KernelParser) -> None:
    """`from .config import BLOCK` should attribute metadata to the config file."""

    package = f"pkg_abs_{uuid.uuid4().hex}"
    _, config_path = module_builder(
        f"{package}.config",
        """
        import triton.language as tl

        BLOCK = tl.constexpr(256)
        """,
    )
    module, _ = module_builder(
        f"{package}.kernels",
        """
        import triton.language as tl
        from .config import BLOCK

        def imported_kernel(x):
            return x + BLOCK
        """,
    )
    parsed = kernel_parser.parse_kernel(module.imported_kernel)
    entry = parsed.constexpr_metadata[0]
    assert os.path.abspath(entry.module) == os.path.abspath(str(config_path))
    assert entry.assignment == "BLOCK = tl.constexpr(256)"


def test_constexpr_metadata_traces_relative_imports_across_packages(module_builder, kernel_parser: KernelParser) -> None:
    """`from ..helpers import SIZE` should walk upward and locate helper modules."""

    package = f"pkg_rel_{uuid.uuid4().hex}"
    _, helpers_path = module_builder(
        f"{package}.helpers",
        """
        import triton.language as tl

        SIZE = tl.constexpr(512)
        """,
    )
    module, _ = module_builder(
        f"{package}.sub.kernels",
        """
        from ..helpers import SIZE

        def rel_kernel(x):
            return x + SIZE
        """,
    )
    parsed = kernel_parser.parse_kernel(module.rel_kernel)
    entry = parsed.constexpr_metadata[0]
    assert entry.name == "SIZE"
    assert os.path.abspath(entry.module) == os.path.abspath(str(helpers_path))
    assert entry.assignment == "SIZE = tl.constexpr(512)"


def test_constexpr_metadata_dataclass_contract() -> None:
    """Ensure ConstexprMetadata remains a simple slotted dataclass."""

    assert is_dataclass(ConstexprMetadata)
    assert [field.name for field in fields(ConstexprMetadata)] == ["name", "module", "assignment"]


# --- ScopeAnalyzer tests ------------------------------------------------------


def test_scope_analyzer_records_basic_and_annotated_assignments(scope_analyzer: ScopeAnalyzer) -> None:
    """Assignments and annotated assignments should populate the symbol table."""

    tree = ast.parse(
        """
value = 42
annotated: int = 7
        """
    )
    table = scope_analyzer.analyze(tree)
    assert table.lookup("value") == 42
    assert table.lookup("annotated") == 7


def test_scope_analyzer_registers_function_definitions_from_globals(scope_analyzer: ScopeAnalyzer) -> None:
    """Function defs should resolve to the globals_map binding (jit decorated)."""

    def helper():
        return "from globals"

    tree = ast.parse(
        """
def helper():
    return "shadow"
        """
    )
    table = scope_analyzer.analyze(tree, globals_map={"helper": helper})
    assert table.lookup("helper") is helper


def test_scope_analyzer_symbol_classification_for_all_known_types(scope_analyzer: ScopeAnalyzer) -> None:
    """Symbol types BUILTIN/JIT_FUNCTION/CONSTEXPR/DTYPE/VARIABLE are all distinguishable."""

    def raw_kernel(x):
        return x

    globals_map = {
        "built_in": len,
        "jit_kernel": triton_jit.JITFunction(raw_kernel),
        "constexpr_value": tlc.constexpr(1),
        "dtype_value": tlc.dtype("fp32"),
        "variable": object(),
    }
    table = scope_analyzer.analyze(ast.parse("pass"), globals_map=globals_map)
    assert table.get_binding("built_in").symbol_type == SymbolType.BUILTIN
    assert table.get_binding("jit_kernel").symbol_type == SymbolType.JIT_FUNCTION
    assert table.get_binding("constexpr_value").symbol_type == SymbolType.CONSTEXPR
    assert table.get_binding("dtype_value").symbol_type == SymbolType.DTYPE
    assert table.get_binding("variable").symbol_type == SymbolType.VARIABLE


def test_symbol_binding_and_table_behaviors() -> None:
    """SymbolTable define/lookup/merge APIs plus SymbolBinding dataclass contract."""

    assert is_dataclass(SymbolBinding)
    table = SymbolTable()
    table.define("alpha", 1)
    assert table.lookup("alpha") == 1
    other = SymbolTable()
    other.define("beta", 2)
    table.merge(other)
    assert table.lookup("beta") == 2
    assert table.get_binding("gamma") is None


def test_scope_analyzer_ignores_dunder_builtins_in_globals(scope_analyzer: ScopeAnalyzer) -> None:
    """__builtins__ should not leak into the reconstructed symbol table."""

    table = scope_analyzer.analyze(ast.parse("pass"), globals_map={"__builtins__": {"len": len}, "value": 1})
    assert table.lookup("value") == 1
    assert table.lookup("__builtins__") is None


# --- Expression evaluation tests ---------------------------------------------


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("123", 123),
        ("3.14", 3.14),
        ("'hello'", "hello"),
        ("True", True),
        ("None", None),
    ],
)
def test_expression_literals(expr: str, expected: Any) -> None:
    """Literal expressions evaluate directly."""

    assert _evaluate_assignment(expr) == expected


def test_expression_collections_supported() -> None:
    """List, tuple, and dict literals should round-trip."""

    assert _evaluate_assignment("(1, 2, 3)") == (1, 2, 3)
    assert _evaluate_assignment("[1, 2, 3]") == [1, 2, 3]
    assert _evaluate_assignment("{'a': 1, 'b': 2}") == {"a": 1, "b": 2}


def test_expression_ternary_ifexp() -> None:
    """If expressions respect the evaluated condition."""

    assert _evaluate_assignment("10 if True else 20") == 10
    assert _evaluate_assignment("10 if False else 20") == 20


def test_expression_boolean_operations() -> None:
    """and/or/not semantics should match Python truth tables."""

    assert _evaluate_assignment("True and False") is False
    assert _evaluate_assignment("False or 99") == 99
    assert _evaluate_assignment("not False") is True


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("1 == 1", True),
        ("1 != 2", True),
        ("3 < 2", False),
        ("3 <= 3", True),
        ("4 > 5", False),
        ("5 >= 5", True),
        ("3 in [1, 2, 3]", True),
        ("4 not in [1, 2, 3]", True),
    ],
)
def test_expression_comparisons(expr: str, expected: Any) -> None:
    """Comparisons (==, !=, <, >, <=, >=, in, not in) should be evaluated."""

    assert _evaluate_assignment(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("1 + 2", 3),
        ("5 - 3", 2),
        ("2 * 3", 6),
        ("8 / 4", 2.0),
        ("7 // 2", 3),
        ("7 % 2", 1),
        ("2 ** 3", 8),
        ("1 << 3", 8),
        ("8 >> 1", 4),
        ("1 | 2", 3),
        ("3 ^ 1", 2),
        ("3 & 1", 1),
    ],
)
def test_expression_binary_operations(expr: str, expected: Any) -> None:
    """Binary arithmetic and bitwise ops are supported."""

    assert _evaluate_assignment(expr) == expected


def test_expression_unary_operations() -> None:
    """Unary +, -, not, and bitwise invert should evaluate."""

    assert _evaluate_assignment("+5") == 5
    assert _evaluate_assignment("-5") == -5
    assert _evaluate_assignment("~1") == -2
    assert _evaluate_assignment("not True") is False


def test_expression_builtin_constructors() -> None:
    """int/float/bool/str are whitelisted call targets."""

    assert _evaluate_assignment("int(3.9)") == 3
    assert _evaluate_assignment("float(3)") == 3.0
    assert _evaluate_assignment("bool(0)") is False
    assert _evaluate_assignment("str(5)") == "5"


def test_expression_math_function_calls() -> None:
    """Math module functions should be executable when the module is in scope."""

    globals_map = {"math": math}
    assert _evaluate_assignment("math.sqrt(16)", globals_map=globals_map) == 4.0


def test_expression_nested_structures() -> None:
    """Nested expressions combine ternaries, math calls, and constructors."""

    globals_map = {"math": math}
    expr = "int(math.sqrt(16) + (2 if True else 3))"
    assert _evaluate_assignment(expr, globals_map=globals_map) == 6


def test_expression_handles_none_and_failures() -> None:
    """Unknown names or failing calls should yield None rather than raising."""

    assert _evaluate_assignment("unknown + 1") is None
    assert _evaluate_assignment("(1, unknown)") is None
    assert _evaluate_assignment("{None: 1}") is None
    assert _evaluate_assignment("int('nan!')") is None
    assert _evaluate_assignment("True and missing") is None


# --- Integration tests -------------------------------------------------------


def test_integration_kernel_parser_and_scope_analyzer_on_triton_pattern(
    module_builder, kernel_parser: KernelParser
) -> None:
    """End-to-end: real Triton-like module with imports, constexprs, and launches."""

    package = f"pkg_integration_{uuid.uuid4().hex}"
    _, config_path = module_builder(
        f"{package}.config",
        """
        import triton.language as tl

        TILE = tl.constexpr(32)
        """,
    )
    module, kernels_path = module_builder(
        f"{package}.kernels",
        """
        import triton.language as tl
        from triton.runtime.jit import jit
        from .config import TILE

        BLOCK = tl.constexpr(64)

        @jit
        def leaf(x):
            return x + TILE

        @jit
        def mid(x):
            return leaf(x) + BLOCK

        @jit
        def root(x, factor: tl.constexpr, *flags: tl.constexpr):
            mid[0](x)
            return mid(x) * factor + BLOCK + TILE
        """,
    )
    parsed = kernel_parser.parse_kernel(module.root)
    graph = parsed.call_graph
    root_key = _call_graph_key(module.root)
    mid_key = _call_graph_key(module.mid)
    assert graph[root_key][0].obj.fn is module.mid.fn
    assert graph[mid_key][0].obj.fn is module.leaf.fn

    names = {entry.name for entry in parsed.constexpr_metadata}
    assert {"BLOCK", "TILE", "factor", "flags"}.issubset(names)
    tile_entry = next(entry for entry in parsed.constexpr_metadata if entry.name == "TILE")
    assert os.path.abspath(tile_entry.module) == os.path.abspath(str(config_path))

    analyzer = ScopeAnalyzer()
    table = analyzer.analyze(parsed.tree, globals_map=parsed.globals_map)
    assert table.get_binding("leaf").symbol_type == SymbolType.JIT_FUNCTION
    assert table.get_binding("BLOCK").symbol_type == SymbolType.CONSTEXPR
    assert table.get_binding("TILE").symbol_type == SymbolType.CONSTEXPR


def test_integration_scope_analyzer_with_complex_expression_mix(scope_analyzer: ScopeAnalyzer) -> None:
    """Large expression block mixes literals, math, ternaries, collections, and calls."""

    source = '''
result = int(math.sqrt(49)) + (5 if True else 0)
mask = (1 << 3) & 0b1110
config = {'alpha': BLOCK, 'beta': tlc.constexpr(16)}
enabled = True and not False
'''
    globals_map = {"math": math, "BLOCK": 32, "tlc": tlc, "int": int}
    tree = ast.parse(textwrap.dedent(source))
    table = scope_analyzer.analyze(tree, globals_map=globals_map)
    assert table.lookup("result") == 12
    assert table.lookup("mask") == 8
    assert table.lookup("config")["alpha"] == 32
    assert isinstance(table.lookup("config")["beta"], tlc.constexpr)
    assert table.lookup("enabled") is True
