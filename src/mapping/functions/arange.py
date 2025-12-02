"""Runtime helper for lowering ``tl.arange``."""

from __future__ import annotations

from typing import Any

from triton.experimental import gluon
import triton.experimental.gluon as ttgl

from ..function_registry import registry


@registry.register("arange")
@gluon.jit
def tl_arange(
    start: Any,
    stop: Any | None = None,
    step: Any | None = None,
    *,
    layout: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Map ``tl.arange`` directly to the Gluon implementation.

    ``layout`` is threaded separately so TTGIR metadata can drive layout-aware
    lowering without relying on global state.
    """

    args: list[Any] = []
    if start is not None:
        args.append(start)
    if stop is not None:
        args.append(stop)
    if step is not None:
        args.append(step)

    call_kwargs = dict(kwargs)
    if layout is not None and "layout" not in call_kwargs:
        call_kwargs["layout"] = layout

    return ttgl.arange(*args, **call_kwargs)


__all__ = ["tl_arange"]
