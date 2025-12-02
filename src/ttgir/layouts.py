"""Thin compatibility layer for Gluon layout objects."""

from __future__ import annotations

from typing import Mapping, TypeAlias, Union

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

LayoutMetadata: TypeAlias = Union[
    AutoLayout,
    BlockedLayout,
    CoalescedLayout,
    DistributedLinearLayout,
    DotOperandLayout,
    NVMMADistributedLayout,
    SliceLayout,
    NVMMASharedLayout,
    PaddedSharedLayout,
    SharedLinearLayout,
    SwizzledSharedLayout,
]

LayoutMap = Mapping[str, LayoutMetadata]

__all__ = [
    "LayoutMetadata",
    "LayoutMap",
    "AutoLayout",
    "BlockedLayout",
    "CoalescedLayout",
    "DistributedLayout",
    "DistributedLinearLayout",
    "DotOperandLayout",
    "NVMMADistributedLayout",
    "NVMMASharedLayout",
    "PaddedSharedLayout",
    "SharedLayout",
    "SharedLinearLayout",
    "SliceLayout",
    "SwizzledSharedLayout",
]
