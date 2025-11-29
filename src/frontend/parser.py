"""Kernel source parsing helpers."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import os
import sys
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Set

import triton.language.core as tlc
from triton.runtime import jit as triton_jit


@dataclass(slots=True)
class CallDependency:
    """Edge in the kernel call graph captured during parsing."""

    name: str
    qualified_name: str
    module: Optional[str]
    obj: Optional[object]


@dataclass(slots=True)
class ConstexprMetadata:
    """Metadata describing a constexpr binding discovered in kernel source."""

    name: str
    module: Optional[str]
    assignment: Optional[str]


@dataclass(slots=True)
class ParsedKernel:
    """In-memory representation of a Triton kernel's Python source."""

    kernel: object
    tree: ast.AST
    source: str
    filename: Optional[str]
    globals_map: Optional[Mapping[str, Any]]
    call_graph: Dict[str, Sequence[CallDependency]]
    constexpr_metadata: Sequence[ConstexprMetadata]

    @property
    def name(self) -> str:
        """Return a friendly kernel name for logging."""

        return getattr(self.kernel, "__name__", repr(self.kernel))


class _CallGraphRecorder:
    """Helper that deduplicates callees for the kernel call graph."""

    def __init__(self) -> None:
        self._dependencies: Dict[str, CallDependency] = {}

    def record(self, callable_obj: Any) -> None:
        if not isinstance(callable_obj, triton_jit.JITCallable):
            return
        base_function = getattr(callable_obj, "fn", callable_obj)
        simple_name = getattr(base_function, "__name__", getattr(base_function, "__qualname__", repr(base_function)))
        module_name = getattr(base_function, "__module__", None)
        qual_name = getattr(base_function, "__qualname__", simple_name)
        qualified_name = ".".join(part for part in (module_name, qual_name) if part)
        key = qualified_name or simple_name
        if key in self._dependencies:
            return
        self._dependencies[key] = CallDependency(
            name=simple_name,
            qualified_name=qualified_name or simple_name,
            module=module_name,
            obj=callable_obj,
        )

    def adjacency(self, kernel_name: str) -> Dict[str, Sequence[CallDependency]]:
        return {kernel_name: list(self._dependencies.values())}


class _ConstexprTracker:
    """Collect constexpr references and their defining modules."""

    def __init__(self, scope: MutableMapping[str, Any], module_path: Optional[str]) -> None:
        self.scope = scope
        self.module_path = module_path
        self.references: Dict[str, Set[str]] = {}

    def record_resolved(self, identifier: str, resolved_obj: Any) -> None:
        module_origin = self._module_from_object(resolved_obj) or self._scope_module_path()
        if not module_origin:
            return
        self.references.setdefault(module_origin, set()).add(identifier)

    def record_annotation(self, identifier: str) -> None:
        module_origin = self._scope_module_path()
        if not module_origin:
            return
        self.references.setdefault(module_origin, set()).add(identifier)

    def build_metadata(self) -> Sequence[ConstexprMetadata]:
        return list(_collect_constexpr_metadata(self.references))

    def _scope_module_path(self) -> Optional[str]:
        module_file = self.scope.get("__file__", self.module_path)
        if isinstance(module_file, str):
            return os.path.abspath(module_file)
        return None

    def _module_from_object(self, resolved_obj: Any) -> Optional[str]:
        if isinstance(resolved_obj, tlc.constexpr):
            # tl.constexpr instances inherit the tl module, so attribute them to the
            # kernel's file (mirrors translator queue handling to keep metadata stable).
            return self._scope_module_path()
        module_name = getattr(resolved_obj, "__module__", None)
        if module_name:
            origin = _resolve_import_origin(module_name)
            if origin:
                return os.path.abspath(origin)
            module = sys.modules.get(module_name)
            module_file = getattr(module, "__file__", None) if module is not None else None
            if isinstance(module_file, str):
                return os.path.abspath(module_file)
        try:
            source_path = inspect.getsourcefile(resolved_obj) or inspect.getfile(resolved_obj)
        except (OSError, TypeError):  # noqa: BLE001 - reflective best-effort
            source_path = None
        return os.path.abspath(source_path) if isinstance(source_path, str) else None


class KernelParser:
    """Parse Triton kernels into ASTs ready for IR construction.

    The parser currently focuses on capturing enough context (source text,
    filename, globals) so later phases can reconstruct constexpr bindings and
    align with TTGIR metadata.
    """

    def parse_kernel(self, kernel: object) -> ParsedKernel:
        """Return :class:`ParsedKernel` for ``kernel``.

        TODO: Preserve decorator trivia, docstrings, and inline comments so the
        emitted Gluon source can mirror the original formatting.
        """

        tree, source, filename, globals_map, metadata_collector = self._parse_tree_and_metadata(kernel)
        root_name = self._kernel_name(kernel)
        call_graph = self._build_call_graph(
            root_name=root_name, initial_collector=metadata_collector, root_kernel=kernel
        )
        constexpr_metadata = metadata_collector.extract_constexpr_metadata()
        return ParsedKernel(
            kernel=kernel,
            tree=tree,
            source=source,
            filename=filename,
            globals_map=globals_map,
            call_graph=call_graph,
            constexpr_metadata=constexpr_metadata,
        )

    def _get_source(self, kernel: object) -> str:
        try:
            if hasattr(kernel, "src"):  # Triton JITFunction stores the raw source
                return getattr(kernel, "src")
            if hasattr(kernel, "_src"):
                return getattr(kernel, "_src")
            return inspect.getsource(kernel)
        except (OSError, TypeError):
            # TODO: Surface a warning so users know fallback scaffolding is used.
            return ""

    def _parse_tree_and_metadata(
        self, kernel: object
    ) -> tuple[ast.AST, str, Optional[str], Optional[Mapping[str, Any]], "_KernelMetadataCollector"]:
        source = self._get_source(kernel)
        filename = self._get_filename(kernel)
        tree = ast.parse(source or "pass")
        ast.fix_missing_locations(tree)
        globals_map = getattr(kernel, "__globals__", None)
        if globals_map is None and isinstance(kernel, triton_jit.JITFunction):
            # JITFunction instances proxy the original python function via ``fn``.
            fn_obj = getattr(kernel, "fn", None)
            globals_map = getattr(fn_obj, "__globals__", None)
        metadata_collector = _KernelMetadataCollector(globals_map=globals_map, module_path=filename)
        metadata_collector.visit(tree)
        return tree, source, filename, globals_map, metadata_collector

    def _get_filename(self, kernel: object) -> Optional[str]:
        try:
            return inspect.getsourcefile(kernel)
        except (OSError, TypeError):
            return None

    def _kernel_name(self, kernel: object) -> str:
        return getattr(kernel, "__name__", repr(kernel))

    def _build_call_graph(
        self, root_name: str, initial_collector: "_KernelMetadataCollector", root_kernel: object
    ) -> Dict[str, Sequence[CallDependency]]:
        # Mirror the translator's breadth-first queue so we capture transitive dependencies.
        adjacency: Dict[str, Sequence[CallDependency]] = {}
        direct_graph = initial_collector.build_call_graph(self_name=root_name)
        direct_dependencies = list(direct_graph.get(root_name, []))
        adjacency[root_name] = direct_dependencies

        queue: deque[triton_jit.JITCallable] = deque()
        seen_callables: Set[int] = set()
        if isinstance(root_kernel, triton_jit.JITCallable):
            seen_callables.add(id(root_kernel))

        def enqueue_from_dependencies(dependencies: Sequence[CallDependency]) -> None:
            for dependency in dependencies:
                callee = getattr(dependency, "obj", None)
                if not isinstance(callee, triton_jit.JITCallable):
                    continue
                callee_id = id(callee)
                if callee_id in seen_callables:
                    continue
                seen_callables.add(callee_id)
                queue.append(callee)

        enqueue_from_dependencies(direct_dependencies)

        while queue:
            callee = queue.popleft()
            callee_name, callee_dependencies = self._parse_dependency_call_graph(callee)
            adjacency.setdefault(callee_name, callee_dependencies)
            enqueue_from_dependencies(callee_dependencies)

        return adjacency

    def _parse_dependency_call_graph(self, callable_obj: triton_jit.JITCallable) -> tuple[str, Sequence[CallDependency]]:
        _, _, _, _, collector = self._parse_tree_and_metadata(callable_obj)
        callee_name = self._kernel_name(callable_obj)
        callee_graph = collector.build_call_graph(self_name=callee_name)
        return callee_name, list(callee_graph.get(callee_name, []))


class _KernelMetadataCollector(ast.NodeVisitor):
    """Collect call graph and constexpr metadata from a kernel AST."""

    def __init__(self, globals_map: Optional[Mapping[str, Any]], module_path: Optional[str]) -> None:
        self.scope: MutableMapping[str, Any] = dict(globals_map or {})
        if "__file__" not in self.scope and module_path:
            self.scope["__file__"] = module_path
        self.module_path = module_path
        self._call_graph = _CallGraphRecorder()
        self._constexpr_tracker = _ConstexprTracker(self.scope, module_path)
        self._annotated_constexpr: Set[str] = set()

    # --- ast.NodeVisitor overrides -------------------------------------------------
    def visit_Call(self, node: ast.Call) -> Any:  # noqa: ANN401 - NodeVisitor API
        resolved_callable = self._unwrap_callable(self._resolve_value(node.func))
        if resolved_callable is None and isinstance(node.func, ast.Subscript):
            # ``kernel[grid](...)`` launches expose the base callable through ``value``.
            resolved_callable = self._unwrap_callable(self._resolve_value(node.func.value))
        self._call_graph.record(resolved_callable)
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:  # noqa: ANN401 - NodeVisitor API
        resolved = self._resolve_value(node)
        identifier = getattr(node, "id", None)
        if isinstance(identifier, str):
            if isinstance(resolved, tlc.constexpr):
                self._constexpr_tracker.record_resolved(identifier, resolved)
            elif identifier in self._annotated_constexpr:
                self._constexpr_tracker.record_annotation(identifier)
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:  # noqa: ANN401 - NodeVisitor API
        if self._is_constexpr_annotation(node.annotation):
            # Annotated constexprs behave like tl.constexpr globals even before runtime binding.
            for identifier in _collect_names([node.target]):
                self._annotated_constexpr.add(identifier)
                self._constexpr_tracker.record_annotation(identifier)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:  # noqa: ANN401 - NodeVisitor API
        self._record_constexpr_parameters(node.args)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:  # noqa: ANN401 - NodeVisitor API
        self._record_constexpr_parameters(node.args)
        return self.generic_visit(node)

    # --- public helpers -----------------------------------------------------------
    def build_call_graph(self, self_name: str) -> Dict[str, Sequence[CallDependency]]:
        """Return adjacency list keyed by the kernel function name."""

        return self._call_graph.adjacency(self_name)

    def extract_constexpr_metadata(self) -> Sequence[ConstexprMetadata]:
        """Return captured constexpr metadata by parsing defining modules."""

        return self._constexpr_tracker.build_metadata()

    # --- resolution helpers -------------------------------------------------------
    def _resolve_value(self, expr: ast.AST) -> Any:
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
            # Treat launch expressions ``kernel[grid]`` as references to ``kernel``.
            return self._resolve_value(expr.value)
        return None

    def _is_constexpr_annotation(self, annotation: Optional[ast.AST]) -> bool:
        if annotation is None:
            return False
        resolved = self._resolve_value(annotation)
        return resolved is tlc.constexpr

    def _record_constexpr_parameters(self, arguments: ast.arguments) -> None:
        # Capture tl.constexpr annotations that live in the function signature just like
        # the translator does when rewriting Triton kernels.
        for arg in list(getattr(arguments, "posonlyargs", [])) + list(arguments.args):
            self._track_constexpr_parameter(arg)
        for arg in arguments.kwonlyargs:
            self._track_constexpr_parameter(arg)
        self._track_constexpr_parameter(arguments.vararg)
        self._track_constexpr_parameter(arguments.kwarg)

    def _track_constexpr_parameter(self, arg_node: Optional[ast.arg]) -> None:
        if arg_node is None:
            return
        if self._is_constexpr_annotation(arg_node.annotation):
            identifier = getattr(arg_node, "arg", None)
            if identifier:
                self._annotated_constexpr.add(identifier)
                self._constexpr_tracker.record_annotation(identifier)

    def _unwrap_callable(self, value: Any) -> Any:
        if value is None:
            return None
        if hasattr(tlc, "_unwrap_if_constexpr"):
            try:
                return tlc._unwrap_if_constexpr(value)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 - defensive
                return value
        return value


def _collect_constexpr_metadata(constexpr_map: Mapping[str, Set[str]]) -> Iterable[ConstexprMetadata]:
    """Parse modules referenced in ``constexpr_map`` and yield metadata entries."""

    if not constexpr_map:
        return []

    cache: Dict[str, tuple[Dict[str, ast.AST], Dict[str, str]]] = {}
    metadata: list[ConstexprMetadata] = []
    for module_path, identifiers in constexpr_map.items():
        assigns, imports = _parse_assignments(module_path, cache)
        for identifier in sorted(identifiers):
            assignment_node = assigns.get(identifier)
            defining_module = module_path
            if assignment_node is None:
                imported_module = imports.get(identifier)
                origin = None
                if isinstance(imported_module, str):
                    if os.path.isabs(imported_module) and os.path.exists(imported_module):
                        origin = imported_module
                    else:
                        origin = _resolve_import_origin(imported_module)
                if origin:
                    defining_module = origin
                    imported_assigns, _ = _parse_assignments(origin, cache)
                    assignment_node = imported_assigns.get(identifier)
            assignment_src = _unparse_node_safe(assignment_node)
            metadata.append(ConstexprMetadata(name=identifier, module=defining_module, assignment=assignment_src))
    return metadata


def _parse_assignments(
    module_path: str, cache: MutableMapping[str, tuple[Dict[str, ast.AST], Dict[str, str]]] | None = None
) -> tuple[Dict[str, ast.AST], Dict[str, str]]:
    normalized_path = os.path.abspath(module_path)
    if cache is not None and normalized_path in cache:
        return cache[normalized_path]

    assigns: Dict[str, ast.AST] = {}
    imports: Dict[str, str] = {}
    try:
        with open(normalized_path, "r", encoding="utf-8") as handle:
            module_ast = ast.parse(handle.read())
    except OSError:
        module_ast = None

    if module_ast is not None:
        for stmt in getattr(module_ast, "body", []):
            if isinstance(stmt, ast.Assign):
                for identifier in _collect_names(stmt.targets):
                    assigns[identifier] = stmt
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                for identifier in _collect_names([stmt.target]):
                    assigns[identifier] = stmt
            elif isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    alias_name = alias.asname or alias.name.split(".")[-1]
                    if stmt.level == 0 and isinstance(stmt.module, str):
                        imports[alias_name] = stmt.module
                    else:
                        origin = _resolve_relative_import_path(normalized_path, stmt.level, stmt.module)
                        if origin:
                            imports[alias_name] = origin

    if cache is not None:
        cache[normalized_path] = (assigns, imports)
    return assigns, imports


def _resolve_relative_import_path(module_path: str, level: int, module: Optional[str]) -> Optional[str]:
    """Resolve ``from .module import`` statements to filesystem paths."""
    if level <= 0 or not module_path:
        return None
    base_dir = os.path.dirname(module_path)
    target_dir = base_dir
    for _ in range(level - 1):
        target_dir = os.path.dirname(target_dir)
    if not target_dir:
        return None
    if module:
        for part in module.split("."):
            if not part:
                continue
            target_dir = os.path.join(target_dir, part)
        return _pick_module_file(target_dir)
    package_init = os.path.join(target_dir, "__init__.py")
    return package_init if os.path.isfile(package_init) else None


def _pick_module_file(base_path: str) -> Optional[str]:
    """Prefer ``module.py`` over ``module/__init__.py`` when both exist."""
    module_file = f"{base_path}.py"
    if os.path.isfile(module_file):
        return module_file
    package_init = os.path.join(base_path, "__init__.py")
    if os.path.isfile(package_init):
        return package_init
    return None


def _collect_names(targets: Sequence[ast.AST]) -> Iterable[str]:
    for target in targets:
        if isinstance(target, ast.Name):
            yield target.id
        elif isinstance(target, (ast.Tuple, ast.List)):
            yield from _collect_names(target.elts)


def _resolve_import_origin(module_name: Optional[str]) -> Optional[str]:
    if not module_name:
        return None
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError):
        return None
    if spec and getattr(spec, "origin", None):
        return spec.origin
    return None


def _unparse_node_safe(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 - ast.unparse best-effort
        return None
