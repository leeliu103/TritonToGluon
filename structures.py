"""Shared dataclasses for Gluon tooling."""

from dataclasses import dataclass


@dataclass
class BuiltinOp:
    """Represents a discovered Gluon builtin operation."""

    name: str
    file_path: str
