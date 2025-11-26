"""Symbol-table helpers for constexpr reconstruction."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional


@dataclass(slots=True)
class SymbolTable:
    """Lightweight mapping of identifiers to their bound objects."""

    bindings: Dict[str, Any] = field(default_factory=dict)

    def define(self, name: str, value: Any) -> None:
        """Record a binding within the current scope."""

        self.bindings[name] = value

    def lookup(self, name: str) -> Any:
        """Return the object bound to ``name`` if known."""

        return self.bindings.get(name)

    def merge(self, other: "SymbolTable") -> None:
        """Merge bindings from ``other`` into ``self`` (shallow copy)."""

        self.bindings.update(other.bindings)


class ScopeAnalyzer(ast.NodeVisitor):
    """Walks a kernel AST and reconstructs constexpr/global bindings."""

    def __init__(self) -> None:
        self.table = SymbolTable()

    def analyze(self, tree: ast.AST, globals_map: Optional[Dict[str, Any]] = None) -> SymbolTable:
        """Populate :class:`SymbolTable` for ``tree``.

        TODO: Propagate nonlocal scopes, class-level constants, and imported
        modules. For now we simply copy the globals map and visit assignments in
        the AST body.
        """

        if globals_map:
            for name, value in globals_map.items():
                self.table.define(name, value)
        self.visit(tree)
        return self.table

    # --- ast.NodeVisitor overrides -------------------------------------------------
    def visit_Assign(self, node: ast.Assign) -> Any:  # noqa: ANN401 - NodeVisitor API
        value = None  # TODO: evaluate constexpr expression safely
        for target in node.targets:
            for identifier in self._collect_names(target):
                self.table.define(identifier, value)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:  # noqa: ANN401
        self.table.define(node.name, None)
        return self.generic_visit(node)

    # --- helpers -------------------------------------------------------------------
    def _collect_names(self, node: ast.AST) -> Iterable[str]:
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, (ast.Tuple, ast.List)):
            for element in node.elts:
                yield from self._collect_names(element)
