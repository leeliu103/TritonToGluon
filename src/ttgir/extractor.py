"""Stub TTGIR extractor that returns placeholder metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Sequence

from .layouts import LayoutMap

if TYPE_CHECKING:  # pragma: no cover - type hint helpers
    from ..frontend.parser import ParsedKernel  # circular import guard


@dataclass(slots=True)
class TTGIROutput:
    """Container for TTGIR data passed to mapping/codegen layers."""

    spec_matches: Mapping[str, object] = field(default_factory=dict)
    layouts: LayoutMap = field(default_factory=dict)
    diagnostics: Sequence[str] = field(default_factory=tuple)


class TTGIRExtractor:
    """Placeholder extractor that will eventually consume MLIR/TTGIR dumps."""

    def extract(
        self,
        parsed_kernel: "ParsedKernel",
        mlir_dump: str | None = None,
    ) -> TTGIROutput:
        """Return a :class:`TTGIROutput` with empty metadata."""

        _ = parsed_kernel  # Avoid unused argument warning while stubbed.
        _ = mlir_dump
        return TTGIROutput()


__all__ = ["TTGIROutput", "TTGIRExtractor"]
