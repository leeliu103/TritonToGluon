"""Helpers that identify source locations for Triton operations."""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class NodeLocation:
    """Unique identifier for a Triton operation using source location."""

    file: str
    op_name: str
    lineno: int
    col_offset: int

    @classmethod
    def from_ast(cls, node: ast.AST, op_name: str, filename: str | None) -> "NodeLocation":
        """Build a :class:`NodeLocation` from an AST node plus operation metadata."""

        file_id = filename or "<unknown>"
        lineno = getattr(node, "lineno", -1)
        col_offset = getattr(node, "col_offset", -1)
        return cls(file=file_id, op_name=op_name, lineno=lineno, col_offset=col_offset)


__all__ = ["NodeLocation"]
