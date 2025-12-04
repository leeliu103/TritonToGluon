"""Schema definitions for operation metadata requirements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(slots=True)
class OpMetadataSpec:
    """Schema defining what layout metadata an operation needs."""

    layouts: Sequence[str] = field(default_factory=tuple)

    def layout_names(self) -> tuple[str, ...]:
        """Return the required layout names as a tuple."""

        return tuple(self.layouts)


__all__ = ["OpMetadataSpec"]
