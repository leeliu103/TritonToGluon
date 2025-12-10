"""Thin compatibility layer for Gluon layout objects."""

from __future__ import annotations

from typing import TypeAlias, Union

from triton.experimental.gluon.language._layouts import (
    AutoLayout,
    BlockedLayout,
    CoalescedLayout,
    DistributedLayout,
    DistributedLinearLayout,
    DotOperandLayout,
    NVMMADistributedLayout,
    NVMMASharedLayout,
    PaddedSharedLayout,
    SharedLayout,
    SharedLinearLayout,
    SliceLayout,
    SwizzledSharedLayout,
)
from triton.experimental.gluon.language.amd._layouts import (
    AMDMFMALayout,
    AMDWMMALayout,
)

LayoutMetadata: TypeAlias = Union[
    AutoLayout,
    BlockedLayout,
    CoalescedLayout,
    DistributedLinearLayout,
    DotOperandLayout,
    NVMMADistributedLayout,
    SliceLayout,
    NVMMASharedLayout,
    AMDMFMALayout,
    AMDWMMALayout,
    PaddedSharedLayout,
    SharedLinearLayout,
    SwizzledSharedLayout,
]

__all__ = [
    "LayoutMetadata",
    "AutoLayout",
    "BlockedLayout",
    "CoalescedLayout",
    "DistributedLayout",
    "DistributedLinearLayout",
    "DotOperandLayout",
    "NVMMADistributedLayout",
    "NVMMASharedLayout",
    "AMDMFMALayout",
    "AMDWMMALayout",
    "PaddedSharedLayout",
    "SharedLayout",
    "SharedLinearLayout",
    "SliceLayout",
    "SwizzledSharedLayout",
]
