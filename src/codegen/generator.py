"""AST-based code generator that mirrors the reference translator's flow."""

from __future__ import annotations

import ast
import copy
import sys
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Callable, Iterable, List, Sequence as TypingSequence

import triton.language.core as tlc
from triton.runtime import jit as triton_jit

from ..frontend.parser import ConstexprMetadata, KernelParser, ParsedKernel
from ..mapping.function_registry import MappingFunctionRegistry, registry as default_registry
from ..ttgir import (
    DistributedLayout,
    LayoutMetadata,
    SharedLayout,
    SliceLayout,
    TTGIROutput,
)


@dataclass(slots=True)
class GluonKernel:
    """Represents one lowered kernel ready for emission."""

    name: str
    source: str


@dataclass(slots=True)
class GluonModule:
    """Bundle of imports, helper blocks, and kernels."""

    kernels: List[GluonKernel] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    constexprs: List[str] = field(default_factory=list)
    diagnostics: List[str] = field(default_factory=list)
    module_ast: ast.Module | None = None


_BASE_IMPORTS: tuple[str, ...] = (
    "from triton.experimental import gluon",
    "from triton.experimental.gluon import language as ttgl",
    "import triton.language as tl",
)


def _call_graph_key(kernel_obj: object) -> str:
    base = getattr(kernel_obj, "fn", kernel_obj)
    simple_name = getattr(base, "__name__", getattr(base, "__qualname__", repr(base)))
    qual_component = getattr(base, "__qualname__", simple_name)
    module_name = getattr(base, "__module__", None)
    qualified = ".".join(part for part in (module_name, qual_component) if part)
    return qualified or simple_name


def _dedupe_preserve_order(lines: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for line in lines:
        if not line or line in seen:
            continue
        ordered.append(line)
        seen.add(line)
    return ordered


def _import_lines_to_nodes(lines: Iterable[str]) -> list[ast.stmt]:
    nodes: list[ast.stmt] = []
    for line in lines:
        parsed = ast.parse(line)
        nodes.extend(parsed.body)
    return nodes


class _ConstexprAssignmentRewriter(ast.NodeTransformer):
    """Rewrite tl.* references inside constexpr assignments to ttgl.*."""

    _TL_NAMES = {"tl", "tlc"}

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:  # noqa: D401
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id in self._TL_NAMES:
            return ast.copy_location(
                ast.Attribute(value=ast.Name(id="ttgl", ctx=ast.Load()), attr=node.attr, ctx=node.ctx),
                node,
            )
        return node


def _reconstruct_constexpr_statements(
    metadata: TypingSequence[ConstexprMetadata],
    referenced_names: set[str],
    value_lookup: Mapping[str, Any] | None = None,
) -> tuple[list[ast.stmt], set[str]]:
    if not referenced_names:
        return [], set()
    rewriter = _ConstexprAssignmentRewriter()
    emitted: list[ast.stmt] = []
    seen: set[str] = set()
    missing: set[str] = set()
    values = value_lookup or {}

    def emit_from_source(source: str) -> ast.Module | None:
        try:
            return ast.parse(source)
        except SyntaxError:
            return None

    for entry in metadata:
        name = entry.name
        if name in seen or name not in referenced_names:
            continue
        assignment_src = entry.assignment
        parsed = emit_from_source(assignment_src) if assignment_src else None
        if parsed is None and name in values:
            fallback_src = f"{name} = {repr(values[name])}"
            parsed = emit_from_source(fallback_src)
        if parsed is not None:
            rewritten = rewriter.visit(parsed)
            ast.fix_missing_locations(rewritten)
            emitted.extend(rewritten.body)
            seen.add(name)
        else:
            missing.add(name)
    unresolved = referenced_names - seen
    missing.update(unresolved)
    return emitted, missing


class CodeGenerator:
    """Walk Triton ASTs, invoke mapping helpers, and emit Gluon source strings."""

    def __init__(self, registry: MappingFunctionRegistry | None = None, kernel_parser: KernelParser | None = None) -> None:
        self.registry = registry or default_registry
        self.registry.load_builtin_functions()
        self.kernel_parser = kernel_parser or KernelParser()

    def generate(self, parsed_kernel: ParsedKernel, ttgir_output: TTGIROutput) -> GluonModule:
        """Produce a :class:`GluonModule` for ``parsed_kernel`` and its callees."""

        graph: dict[str, list] = {key: list(deps) for key, deps in parsed_kernel.call_graph.items()}
        root_key = _call_graph_key(parsed_kernel.kernel)
        graph.setdefault(root_key, [])
        callable_lookup: dict[str, object] = {root_key: parsed_kernel.kernel}
        self._augment_callable_lookup(callable_lookup, graph.values())

        queue: deque[str] = deque([root_key])
        parsed_cache: dict[str, ParsedKernel] = {root_key: parsed_kernel}
        visited: set[str] = set()
        ordered_functions: list[ast.AST] = []
        helper_imports: set[str] = set()
        support_imports: set[str] = set()
        referenced_constexpr: set[str] = set()
        metadata_entries: list[ConstexprMetadata] = list(parsed_kernel.constexpr_metadata)
        constexpr_values: dict[str, Any] = {}
        self._merge_constexpr_values(constexpr_values, parsed_kernel.globals_map)
        diagnostics: list[str] = list(ttgir_output.diagnostics)

        while queue:
            key = queue.popleft()
            if key in visited:
                continue
            kernel_ast = parsed_cache.get(key)
            if kernel_ast is None:
                callable_obj = callable_lookup.get(key)
                if callable_obj is None:
                    diagnostics.append(f"Call graph entry '{key}' is missing a callable reference; skipping translation.")
                    continue
                kernel_ast = self._parse_kernel(callable_obj, diagnostics)
                if kernel_ast is None:
                    continue
                parsed_cache[key] = kernel_ast
                metadata_entries.extend(kernel_ast.constexpr_metadata)
                self._merge_constexpr_values(constexpr_values, kernel_ast.globals_map)
                self._merge_call_graph(graph, kernel_ast.call_graph)
                self._augment_callable_lookup(callable_lookup, kernel_ast.call_graph.values())

            visited.add(key)
            callable_obj = callable_lookup.get(key, kernel_ast.kernel)
            transformer = _CallMappingTransformer(
                parsed_kernel=kernel_ast,
                ttgir_output=ttgir_output,
                registry=self.registry,
                is_jit=self._is_jit_callable(callable_obj),
            )
            transformed = transformer.visit(copy.deepcopy(kernel_ast.tree))
            ast.fix_missing_locations(transformed)
            function_nodes = [
                node
                for node in getattr(transformed, "body", [])
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            ordered_functions.extend(function_nodes)
            helper_imports.update(transformer.helper_imports)
            support_imports.update(transformer.support_imports)
            referenced_constexpr.update(transformer.referenced_constexpr)
            diagnostics.extend(transformer.diagnostics)

            for dependency in graph.get(key, []):
                dep_key = getattr(dependency, "qualified_name", None) or getattr(dependency, "name", None)
                if not dep_key:
                    continue
                callable_lookup.setdefault(dep_key, getattr(dependency, "obj", None))
                graph.setdefault(dep_key, [])
                if dep_key not in visited:
                    queue.append(dep_key)

            for nested in transformer.nested_kernels:
                nested_key = _call_graph_key(nested)
                callable_lookup.setdefault(nested_key, nested)
                if nested_key not in graph:
                    nested_kernel = self._parse_kernel(nested, diagnostics)
                    if nested_kernel is None:
                        continue
                    parsed_cache[nested_key] = nested_kernel
                    metadata_entries.extend(nested_kernel.constexpr_metadata)
                    self._merge_constexpr_values(constexpr_values, nested_kernel.globals_map)
                    self._merge_call_graph(graph, nested_kernel.call_graph)
                    self._augment_callable_lookup(callable_lookup, nested_kernel.call_graph.values())
                if nested_key not in visited:
                    queue.append(nested_key)

        constexpr_nodes, missing_constexpr = _reconstruct_constexpr_statements(
            metadata_entries, referenced_constexpr, constexpr_values
        )
        for name in sorted(missing_constexpr):
            diagnostics.append(f"Missing constexpr metadata for '{name}'.")

        import_lines = _dedupe_preserve_order(
            list(_BASE_IMPORTS)
            + sorted(support_imports)
            + sorted(helper_imports)
        )
        module_body: list[ast.stmt] = []
        module_body.extend(_import_lines_to_nodes(import_lines))
        module_body.extend(constexpr_nodes)
        module_body.extend(ordered_functions)
        module_ast = ast.Module(body=module_body, type_ignores=[])
        ast.fix_missing_locations(module_ast)

        kernels = [
            GluonKernel(name=getattr(node, "name", f"kernel_{idx}"), source=ast.unparse(node))
            for idx, node in enumerate(ordered_functions)
        ]
        constexpr_sources = [ast.unparse(stmt) for stmt in constexpr_nodes]
        diagnostics = _dedupe_preserve_order(diagnostics)

        return GluonModule(
            kernels=kernels,
            imports=import_lines,
            constexprs=constexpr_sources,
            diagnostics=diagnostics,
            module_ast=module_ast,
        )

    def _parse_kernel(self, kernel_obj: object, diagnostics: list[str]) -> ParsedKernel | None:
        try:
            return self.kernel_parser.parse_kernel(kernel_obj)
        except Exception as exc:  # noqa: BLE001 - defensive: parsing relies on user code
            diagnostics.append(f"Failed to parse kernel '{_call_graph_key(kernel_obj)}': {exc}")
            return None

    @staticmethod
    def _is_jit_callable(kernel_obj: object | None) -> bool:
        return isinstance(kernel_obj, triton_jit.JITCallable)

    @staticmethod
    def _augment_callable_lookup(lookup: dict[str, object], dependency_groups: Iterable[Sequence]) -> None:
        for group in dependency_groups:
            for dependency in group:
                dep_key = getattr(dependency, "qualified_name", None) or getattr(dependency, "name", None)
                dep_obj = getattr(dependency, "obj", None)
                if dep_key and dep_obj is not None:
                    lookup.setdefault(dep_key, dep_obj)

    @staticmethod
    def _merge_call_graph(target: dict[str, list], new_graph: Mapping[str, Sequence]) -> None:
        for key, deps in new_graph.items():
            bucket = target.setdefault(key, [])
            for dependency in deps:
                if dependency not in bucket:
                    bucket.append(dependency)

    @staticmethod
    def _merge_constexpr_values(target: dict[str, Any], source: Mapping[str, Any] | None) -> None:
        if not source:
            return
        for name, value in source.items():
            target.setdefault(name, value)


class _CallMappingTransformer(ast.NodeTransformer):
    """Internal helper that rewrites calls using the mapping registry."""

    _METADATA_PARAMS: Mapping[str, Sequence[str]] = {
        "arange": ("layout",),
        "dot": ("layout_a", "layout_b", "input_precision"),
        "dot_scaled": ("layout_a", "layout_b", "input_precision"),
        "program_id": ("layout",),
    }
    _LAYOUT_RESET_OPS: set[str] = {"reshape", "trans", "permute", "join", "reduce", "split"}

    def __init__(
        self,
        parsed_kernel: ParsedKernel,
        ttgir_output: TTGIROutput,
        registry: MappingFunctionRegistry,
        *,
        is_jit: bool,
    ) -> None:
        super().__init__()
        self.parsed_kernel = parsed_kernel
        self.ttgir_output = ttgir_output
        self.registry = registry
        self.is_jit = is_jit
        self.helper_imports: set[str] = set()
        self.support_imports: set[str] = set()
        self.referenced_constexpr: set[str] = set()
        self.nested_kernels: set[object] = set()
        self.diagnostics: list[str] = []
        self.scope: dict[str, Any] = dict(parsed_kernel.globals_map or {})
        if parsed_kernel.filename and "__file__" not in self.scope:
            self.scope["__file__"] = parsed_kernel.filename
        self._current_function: ast.FunctionDef | None = None

    # ------------------------------------------------------------------ visits
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:  # pragma: no cover - parity shim
        return self._visit_function(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: D401
        node = self.generic_visit(node)
        resolved = self._resolve_value(node)
        if isinstance(resolved, triton_jit.JITCallable):
            self.nested_kernels.add(resolved)
            normalized = self._callable_name(resolved)
            return ast.copy_location(ast.Name(id=normalized, ctx=node.ctx), node)
        if isinstance(resolved, tlc.constexpr):
            identifier = getattr(node, "id", None)
            if identifier:
                self.referenced_constexpr.add(identifier)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:  # noqa: D401
        node = self.generic_visit(node)
        resolved = self._resolve_value(node)
        if isinstance(resolved, tlc.dtype):
            return ast.copy_location(self._ttgl_attr(node.attr), node)
        if resolved is tlc.dtype and node.attr == "dtype":
            return ast.copy_location(self._ttgl_attr("dtype"), node)
        if resolved is tlc.tensor and node.attr == "tensor":
            return ast.copy_location(self._ttgl_attr("tensor"), node)
        if resolved is tlc.constexpr and node.attr == "constexpr":
            return ast.copy_location(self._ttgl_attr("constexpr"), node)
        if node.attr == "tensor_descriptor":
            return ast.copy_location(self._ttgl_attr("nvidia.hopper.tma.tensor_descriptor"), node)
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:  # noqa: D401
        node = self.generic_visit(node)
        expanded_dim = self._expanded_dim_for_none_index(node.slice)
        if expanded_dim is None:
            return node
        layout_expr = self._slice_layout_from_metadata(node)
        if layout_expr is None:
            layout_expr = self._default_slice_layout(node, expanded_dim)
            self.diagnostics.append(
                f"Missing SliceLayout metadata for broadcast at {self._node_lookup_key(node)}; using default layout."
            )
        ast.copy_location(layout_expr, node)
        converted = ast.Call(
            func=self._ttgl_attr("convert_layout"),
            args=[node.value, layout_expr],
            keywords=[],
        )
        ast.copy_location(converted, node.value)
        return ast.copy_location(ast.Subscript(value=converted, slice=node.slice, ctx=node.ctx), node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        resolved_callable = self._unwrap_callable(self._resolve_value(node.func))
        if isinstance(resolved_callable, triton_jit.JITCallable):
            self.nested_kernels.add(resolved_callable)
            replacement = ast.copy_location(
                ast.Name(id=self._callable_name(resolved_callable), ctx=ast.Load()),
                node.func,
            )
            return ast.copy_location(ast.Call(func=replacement, args=node.args, keywords=node.keywords), node)
        op_name = self._resolve_triton_op(node.func)
        if op_name is None:
            return self._wrap_tensor_method_layout(node)
        helper_fn = self.registry.lookup(op_name)
        if helper_fn is None:
            self.diagnostics.append(
                f"Unmapped Triton op '{op_name}' at {self._node_lookup_key(node)}; leaving call unchanged."
            )
            return self._wrap_layout_sensitive_builtin(node, op_name)
        replacement = self._rewrite_call(node, op_name, helper_fn)
        if replacement is None:
            return self._wrap_layout_sensitive_builtin(node, op_name)
        import_stmt = self.registry.get_import_path(op_name)
        if import_stmt:
            self.helper_imports.add(import_stmt)
        return self._wrap_layout_sensitive_builtin(replacement, op_name)

    # ------------------------------------------------------------------ helpers
    def _visit_function(self, node: ast.AST) -> ast.AST:
        prev = self._current_function
        self._current_function = node  # type: ignore[assignment]
        try:
            self._rewrite_constexpr_parameters(getattr(node, "args", None))
            node = self.generic_visit(node)
        finally:
            self._current_function = prev
        decorator_name = "jit" if self.is_jit else "constexpr_function"
        decorators = getattr(node, "decorator_list", [])
        if not any(self._is_gluon_decorator(dec, decorator_name) for dec in decorators):
            decorators.insert(0, self._gluon_attr(decorator_name))
        return node

    def _rewrite_constexpr_parameters(self, arguments: ast.arguments | None) -> None:
        if arguments is None:
            return
        for arg in list(getattr(arguments, "posonlyargs", [])) + list(arguments.args):
            arg.annotation = self._rewrite_annotation(arg.annotation)
        for arg in arguments.kwonlyargs:
            arg.annotation = self._rewrite_annotation(arg.annotation)
        if arguments.vararg is not None:
            arguments.vararg.annotation = self._rewrite_annotation(arguments.vararg.annotation)
        if arguments.kwarg is not None:
            arguments.kwarg.annotation = self._rewrite_annotation(arguments.kwarg.annotation)

    def _rewrite_annotation(self, annotation: ast.expr | None) -> ast.expr | None:
        if annotation is None:
            return None
        if self._is_constexpr_annotation(annotation):
            return self._ttgl_attr("constexpr")
        return annotation

    def _is_constexpr_annotation(self, annotation: ast.expr) -> bool:
        resolved = self._resolve_value(annotation)
        return resolved is tlc.constexpr

    def _resolve_triton_op(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Attribute):
            base = node.value
            if isinstance(base, ast.Name) and base.id in {"tl", "tlc"}:
                return node.attr
        elif isinstance(node, ast.Name):
            return node.id
        return None

    def _rewrite_call(self, node: ast.Call, op_name: str, helper_fn: Callable[..., Any]) -> ast.AST | None:
        helper_name = getattr(helper_fn, "__name__", None)
        if not helper_name:
            return None
        metadata_keywords = self._build_metadata_keywords(node, op_name)
        new_keywords = list(node.keywords)
        new_keywords.extend(metadata_keywords)
        rewritten = ast.Call(
            func=ast.Name(id=helper_name, ctx=ast.Load()),
            args=node.args,
            keywords=new_keywords,
        )
        ast.copy_location(rewritten, node)
        return rewritten

    def _build_metadata_keywords(self, node: ast.Call, op_name: str) -> list[ast.keyword]:
        expected = self._METADATA_PARAMS.get(op_name, ())
        if not expected:
            return []
        existing = {kw.arg for kw in node.keywords if kw.arg}
        keywords: list[ast.keyword] = []
        for meta_name in expected:
            if meta_name in existing:
                continue
            value = self._metadata_value_for(node, meta_name)
            if value is None:
                self.diagnostics.append(
                    f"Missing TTGIR metadata '{meta_name}' for {self._node_lookup_key(node)}."
                )
                continue
            value_node = self._value_to_ast(value)
            ast.copy_location(value_node, node)
            keywords.append(ast.keyword(arg=meta_name, value=value_node))
        return keywords

    def _wrap_layout_sensitive_builtin(self, node: ast.Call, op_name: str) -> ast.AST:
        if op_name not in self._LAYOUT_RESET_OPS:
            return node
        updated = node
        if op_name == "split":
            updated = self._wrap_split_source_argument(updated)
        return self._wrap_reset_to_default(updated)

    def _wrap_tensor_method_layout(self, node: ast.Call) -> ast.AST:
        if not isinstance(node.func, ast.Attribute):
            return node
        attr_name = node.func.attr
        if attr_name not in self._LAYOUT_RESET_OPS:
            return node
        base_obj = node.func.value
        if isinstance(base_obj, ast.Name) and base_obj.id in {"tl", "tlc", "ttgl"}:
            return node
        updated_call = node
        if attr_name == "split":
            updated_call = self._wrap_attribute_split_call(node)
        return self._wrap_reset_to_default(updated_call)

    def _wrap_split_source_argument(self, node: ast.Call) -> ast.Call:
        if not node.args:
            return node
        self._ensure_layout_helper_import("set_split_src_layout")
        source_arg = node.args[0]
        wrapped_src = ast.Call(
            func=ast.Name(id="set_split_src_layout", ctx=ast.Load()),
            args=[source_arg],
            keywords=[],
        )
        ast.copy_location(wrapped_src, source_arg)
        node.args[0] = wrapped_src
        return node

    def _wrap_attribute_split_call(self, node: ast.Call) -> ast.Call:
        receiver = node.func.value
        self._ensure_layout_helper_import("set_split_src_layout")
        wrapped_receiver = ast.Call(
            func=ast.Name(id="set_split_src_layout", ctx=ast.Load()),
            args=[receiver],
            keywords=[],
        )
        ast.copy_location(wrapped_receiver, receiver)
        new_func = ast.Attribute(
            value=ast.copy_location(wrapped_receiver, receiver),
            attr=node.func.attr,
            ctx=ast.Load(),
        )
        ast.copy_location(new_func, node.func)
        new_call = ast.Call(func=new_func, args=list(node.args), keywords=list(node.keywords))
        ast.copy_location(new_call, node)
        return new_call

    def _wrap_reset_to_default(self, call_node: ast.AST) -> ast.AST:
        self._ensure_layout_helper_import("reset_to_default_layout")
        wrapper = ast.Call(
            func=ast.Name(id="reset_to_default_layout", ctx=ast.Load()),
            args=[call_node],
            keywords=[],
        )
        return ast.copy_location(wrapper, call_node)

    def _ensure_layout_helper_import(self, helper_name: str) -> None:
        self.support_imports.add(
            f"from triton.tools.triton_to_gluon_translater.translator_helpers import {helper_name}"
        )

    def _metadata_value_for(self, node: ast.Call, meta_name: str) -> Any:
        if meta_name.startswith("layout"):
            return self._extract_layout_metadata(node, meta_name)
        if meta_name == "input_precision":
            return self._extract_spec_metadata(node, meta_name)
        return None

    def _extract_layout_metadata(self, node: ast.Call, meta_name: str) -> Any:
        node_key = self._node_lookup_key(node)
        layout_entry = self.ttgir_output.layouts.get(node_key)
        if isinstance(layout_entry, Mapping):
            return layout_entry.get(meta_name)
        return layout_entry

    def _extract_spec_metadata(self, node: ast.Call, meta_name: str) -> Any:
        node_key = self._node_lookup_key(node)
        candidates = (f"{node_key}:{meta_name}", meta_name)
        for key in candidates:
            if key in self.ttgir_output.spec_matches:
                return self.ttgir_output.spec_matches[key]
        return None

    def _node_lookup_key(self, node: ast.AST) -> str:
        func_name = self._current_function.name if self._current_function else "<module>"
        lineno = getattr(node, "lineno", -1)
        col = getattr(node, "col_offset", -1)
        return f"{func_name}:{lineno}:{col}"

    def _value_to_ast(self, value: Any) -> ast.AST:
        if isinstance(value, LayoutMetadata):
            return self._layout_to_ast(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return ast.Constant(value=value)
        return ast.Constant(value=None)

    def _gluon_attr(self, attr: str) -> ast.AST:
        return ast.Attribute(
            value=ast.Name(id="gluon", ctx=ast.Load()),
            attr=attr,
            ctx=ast.Load(),
        )

    def _ttgl_attr(self, attr: str) -> ast.Attribute:
        parts = attr.split(".")
        expr: ast.expr = ast.Name(id="ttgl", ctx=ast.Load())
        for part in parts:
            expr = ast.Attribute(value=expr, attr=part, ctx=ast.Load())
        return expr  # type: ignore[return-value]

    def _is_gluon_decorator(self, decorator: ast.AST, target: str | None = None) -> bool:
        if not isinstance(decorator, ast.Attribute):
            return False
        if not (isinstance(decorator.value, ast.Name) and decorator.value.id == "gluon"):
            return False
        if target is not None and decorator.attr != target:
            return False
        return True

    # ---------------------------- resolution helpers -------------------------
    def _resolve_value(self, expr: ast.AST | None) -> Any:
        if expr is None:
            return None
        if isinstance(expr, ast.Name):
            if expr.id in self.scope:
                return self.scope[expr.id]
            return sys.modules.get(expr.id)
        if isinstance(expr, ast.Attribute):
            base = self._resolve_value(expr.value)
            if base is None:
                return None
            return getattr(base, expr.attr, None)
        if isinstance(expr, ast.Subscript):
            return self._resolve_value(expr.value)
        return None

    @staticmethod
    def _unwrap_callable(value: Any) -> Any:
        if value is None:
            return None
        unwrap = getattr(tlc, "_unwrap_if_constexpr", None)
        if callable(unwrap):
            try:
                return unwrap(value)
            except Exception:  # noqa: BLE001 - upstream helper best-effort
                return value
        return value

    @staticmethod
    def _callable_name(callable_obj: triton_jit.JITCallable) -> str:
        base = getattr(callable_obj, "fn", callable_obj)
        return getattr(base, "__name__", getattr(base, "__qualname__", "kernel"))

    # ---------------------------- layout helpers -----------------------------
    def _expanded_dim_for_none_index(self, slice_node: ast.AST | None) -> int | None:
        if isinstance(slice_node, ast.Tuple) and len(slice_node.elts) == 2:
            first, second = slice_node.elts
            if isinstance(first, ast.Constant) and first.value is None:
                return 0
            if isinstance(second, ast.Constant) and second.value is None:
                return 1
        return None

    def _slice_layout_from_metadata(self, node: ast.Subscript) -> ast.expr | None:
        node_key = self._node_lookup_key(node)
        layout = self.ttgir_output.layouts.get(node_key)
        if isinstance(layout, SliceLayout):
            return self._layout_to_ast(layout)
        return None

    def _layout_to_ast(self, layout: LayoutMetadata | None) -> ast.expr:
        if layout is None:
            return ast.Constant(value=None)
        if not is_dataclass(layout):
            raise TypeError(f"Unsupported layout object: {layout!r}")
        ctor_args = [self._layout_value_to_ast(getattr(layout, field.name)) for field in fields(layout)]
        return ast.Call(
            func=self._ttgl_attr(layout.__class__.__name__),
            args=ctor_args,
            keywords=[],
        )

    def _layout_value_to_ast(self, value: Any) -> ast.expr:
        layout_types = (DistributedLayout, SharedLayout)
        if isinstance(value, layout_types):
            return self._layout_to_ast(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return ast.List(elts=[self._layout_value_to_ast(item) for item in value], ctx=ast.Load())
        if isinstance(value, (bool, int, float, str)) or value is None:
            return ast.Constant(value=value)
        raise TypeError(f"Unsupported layout attribute value: {value!r} ({type(value).__name__})")

    def _default_slice_layout(self, node: ast.Subscript, expanded_dim: int) -> ast.expr:
        self.support_imports.add(
            "from triton.tools.triton_to_gluon_translater.translator_helpers import default_blocked_layout"
        )
        tensor_type = ast.Attribute(value=node.value, attr="type", ctx=ast.Load())
        shape_attr = ast.Attribute(value=tensor_type, attr="shape", ctx=ast.Load())
        length_expr = ast.Subscript(value=shape_attr, slice=ast.Constant(value=0), ctx=ast.Load())
        parent_shape = (
            ast.List(elts=[length_expr, ast.Constant(value=1)], ctx=ast.Load())
            if expanded_dim == 0
            else ast.List(elts=[ast.Constant(value=1), length_expr], ctx=ast.Load())
        )
        blocked_layout = ast.Call(
            func=ast.Name(id="default_blocked_layout", ctx=ast.Load()),
            args=[parent_shape, ast.Call(func=self._ttgl_attr("num_warps"), args=[], keywords=[])],
            keywords=[],
        )
        return ast.Call(
            func=self._ttgl_attr("SliceLayout"),
            args=[ast.Constant(value=expanded_dim), blocked_layout],
            keywords=[],
        )


__all__ = ["CodeGenerator", "GluonKernel", "GluonModule"]
