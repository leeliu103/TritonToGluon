"""Construct annotated IR from a parsed Triton kernel."""

from __future__ import annotations

import ast
from typing import Iterable

from ..frontend.parser import ParsedKernel
from ..ttgir.layouts import LayoutMap, LayoutMetadata
from .annotated import AnnotatedKernel, AnnotatedNode


class IRBuilder:
    """Pair Python AST statements with TTGIR-derived metadata."""

    def build(self, parsed: ParsedKernel, *, layouts: LayoutMap | None = None) -> AnnotatedKernel:
        """Return an :class:`AnnotatedKernel` for ``parsed``.

        The builder keeps the structure intentionally lightweight: each AST
        statement becomes an :class:`AnnotatedNode` seeded with any known layout
        metadata extracted from TTGIR. Future passes can replace the placeholder
        heuristics in :meth:`_scan_ast` with true AST↔TTGIR alignment while
        reusing this façade.
        """

        kernel = AnnotatedKernel(name=parsed.name, source=parsed.source)
        for node in self._scan_ast(parsed.tree, layouts):
            kernel.add_node(node)
        return kernel

    # ------------------------------------------------------------------ helpers
    def _scan_ast(self, tree: ast.AST, layouts: LayoutMap | None) -> Iterable[AnnotatedNode]:
        """Emit placeholder nodes for top-level statements."""

        body = getattr(tree, "body", [])
        for index, statement in enumerate(body):
            node_id = f"n{index}"
            layout = self._lookup_layout(layouts, node_id)
            yield AnnotatedNode(id=node_id, ast_node=statement, op=type(statement).__name__, layout=layout)

    def _lookup_layout(self, layouts: LayoutMap | None, node_id: str) -> LayoutMetadata | None:
        if layouts is None:
            return None
        return layouts.get(node_id)
