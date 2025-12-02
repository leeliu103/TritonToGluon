"""Semantic mapping between Triton AST nodes and Gluon constructs."""

from __future__ import annotations

from .function_registry import MappingFunctionRegistry, registry

__all__ = ["MappingFunctionRegistry", "registry"]
