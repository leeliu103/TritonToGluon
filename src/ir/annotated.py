"""Thin AST annotations enriched with TTGIR metadata."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

from ..ttgir.layouts import LayoutMetadata


@dataclass(slots=True)
class AnnotatedNode:
    """IR leaf that glues Triton AST statements to TTGIR metadata."""

    id: str
    ast_node: ast.AST
    op: str
    operands: Sequence[str] = field(default_factory=tuple)
    dtype: str | None = None
    layout: LayoutMetadata | None = None
    ttgir_value: str | None = None


@dataclass(slots=True)
class AnnotatedKernel:
    """Container for the nodes produced from one Triton kernel AST."""

    name: str
    source: str
    nodes: List[AnnotatedNode] = field(default_factory=list)

    def add_node(self, node: AnnotatedNode) -> None:
        self.nodes.append(node)

    def walk(self) -> Iterable[AnnotatedNode]:
        """Iterate over annotated nodes in insertion order."""

        yield from self.nodes
