"""Kernel source parsing helpers."""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(slots=True)
class ParsedKernel:
    """In-memory representation of a Triton kernel's Python source."""

    kernel: object
    tree: ast.AST
    source: str
    filename: Optional[str]
    globals_map: Optional[Mapping[str, Any]]

    @property
    def name(self) -> str:
        """Return a friendly kernel name for logging."""

        return getattr(self.kernel, "__name__", repr(self.kernel))


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

        source = self._get_source(kernel)
        filename = self._get_filename(kernel)
        tree = ast.parse(source or "pass")
        ast.fix_missing_locations(tree)
        globals_map = getattr(kernel, "__globals__", None)
        return ParsedKernel(kernel=kernel, tree=tree, source=source, filename=filename, globals_map=globals_map)

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

    def _get_filename(self, kernel: object) -> Optional[str]:
        try:
            return inspect.getsourcefile(kernel)
        except (OSError, TypeError):
            return None
