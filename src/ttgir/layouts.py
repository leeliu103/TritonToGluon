"""TTGIR layout data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class LayoutMetadata:
    """Base class for layout descriptors extracted from TTGIR dumps."""

    kind: str

    def describe(self) -> str:
        """Return a human-readable summary useful for debugging."""

        return self.kind


@dataclass(frozen=True, slots=True)
class BlockedLayout(LayoutMetadata):
    """Represents ``ttgl.BlockedLayout`` attributes."""

    size_per_thread: Sequence[int]
    threads_per_warp: Sequence[int]
    warps_per_cta: Sequence[int]
    order: Sequence[int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", "blocked")


@dataclass(frozen=True, slots=True)
class SliceLayout(LayoutMetadata):
    """Represents ``ttgl.SliceLayout`` metadata."""

    dimension: int
    parent: BlockedLayout

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", "slice")


LayoutMap = Mapping[str, LayoutMetadata]
