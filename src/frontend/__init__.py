"""Frontend utilities for the slim Triton→Gluon pipeline.

The frontend is responsible for parsing Triton kernels, collecting their
Python AST, and constructing the symbol tables required for TTGIR alignment.
Only a thin façade is provided here so downstream modules can import a stable
API while the implementations remain stubs.
"""

from __future__ import annotations

from .parser import CallDependency, ConstexprMetadata, KernelParser, ParsedKernel
from .scope import ScopeAnalyzer, SymbolBinding, SymbolTable, SymbolType

__all__ = [
    "CallDependency",
    "ConstexprMetadata",
    "KernelParser",
    "ParsedKernel",
    "SymbolBinding",
    "ScopeAnalyzer",
    "SymbolTable",
    "SymbolType",
]
