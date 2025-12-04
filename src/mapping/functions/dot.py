"""Runtime helpers for ``tl.dot`` and related lowerings."""

from __future__ import annotations

from typing import Any

from triton.experimental import gluon

from triton.tools.triton_to_gluon_translater.translator_helpers import (
    tl_dot as upstream_tl_dot,
    tl_dot_decomposed_scale_to_16 as upstream_tl_dot_scaled,
)

from ..function_registry import registry
from ..metadata_schema import OpMetadataSpec


@registry.register("dot", schema=OpMetadataSpec(layouts=("layout_a", "layout_b")))
@gluon.jit
def tl_dot(
    a: Any,
    b: Any,
    acc: Any | None = None,
    *,
    layout_a: Any | None = None,
    layout_b: Any | None = None,
    input_precision: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Invoke the upstream ``tl_dot`` helper with optional TTGIR metadata."""

    _ = (layout_a, layout_b)
    return upstream_tl_dot(a, b, acc, input_precision=input_precision, **kwargs)


@registry.register("dot_scaled", schema=OpMetadataSpec(layouts=("layout_a", "layout_b")))
@gluon.jit
def tl_dot_scaled(
    a: Any,
    b: Any,
    scale: Any,
    *,
    layout_a: Any | None = None,
    layout_b: Any | None = None,
    input_precision: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Lower ``tl.dot_scaled`` through the upstream decomposed helper."""

    _ = (layout_a, layout_b, input_precision)
    return upstream_tl_dot_scaled(a, b, scale=scale, **kwargs)


__all__ = ["tl_dot", "tl_dot_scaled"]
