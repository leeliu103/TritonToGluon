"""Runtime helper for lowering ``tl.program_id``."""

from __future__ import annotations

from typing import Any

from triton.experimental import gluon
import triton.experimental.gluon as ttgl

from ..function_registry import registry


@registry.register("program_id")
@gluon.jit
def tl_program_id(
    axis: Any | None = None,
    *,
    layout: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Lower ``tl.program_id`` to the Gluon API with optional layout metadata."""

    call_kwargs = dict(kwargs)
    if layout is not None and "layout" not in call_kwargs:
        call_kwargs["layout"] = layout

    if axis is None:
        return ttgl.program_id(**call_kwargs)
    return ttgl.program_id(axis, **call_kwargs)


__all__ = ["tl_program_id"]
