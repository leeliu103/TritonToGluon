"""Stub TTGIR extractor that returns placeholder metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .layouts import LayoutMetadata
from .location import NodeLocation

if TYPE_CHECKING:  # pragma: no cover - type hint helpers
    from ..frontend.parser import ParsedKernel  # circular import guard


@dataclass
class NodeMetadata:
    """All metadata for a single Triton operation (layouts only for now)."""

    layouts: dict[str, LayoutMetadata] = field(default_factory=dict)

    def get_layout(self, name: str | None) -> LayoutMetadata | None:
        """Return a specific layout (or the only layout present)."""

        if name is None:
            if len(self.layouts) == 1:
                return next(iter(self.layouts.values()))
            return None
        return self.layouts.get(name)


@dataclass
class TTGIROutput:
    """TTGIR extraction results with location-based lookup."""

    metadata: dict[NodeLocation, NodeMetadata] = field(default_factory=dict)

    def get_layout(self, location: NodeLocation, name: str | None) -> LayoutMetadata | None:
        """Return layout metadata for location if present."""

        entry = self.metadata.get(location)
        return entry.get_layout(name) if entry else None


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


__all__ = ["NodeMetadata", "TTGIROutput", "TTGIRExtractor"]
