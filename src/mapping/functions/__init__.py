"""Builtin mapping functions that populate the default registry."""

from __future__ import annotations

# Import submodules for their registration side effects.
from . import arange as _arange  # noqa: F401
from . import dot as _dot  # noqa: F401
from . import program_id as _program_id  # noqa: F401

__all__ = ["_arange", "_dot", "_program_id"]
