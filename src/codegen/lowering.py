"""Lower annotated IR into an in-memory Gluon module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ..ir.annotated import AnnotatedKernel, AnnotatedNode
from ..mapping.registry import MappingRegistry


@dataclass(slots=True)
class GluonInstruction:
    """Represents a single high-level Gluon statement."""

    op: str
    args: List[str] = field(default_factory=list)
    comment: str | None = None


@dataclass(slots=True)
class GluonKernel:
    """Container for the lowered body of one kernel."""

    name: str
    instructions: List[GluonInstruction] = field(default_factory=list)
    decorators: List[str] = field(default_factory=lambda: ["@gluon.jit"])


@dataclass(slots=True)
class GluonModule:
    """Top-level Gluon module with imports and kernels."""

    kernels: List[GluonKernel] = field(default_factory=list)
    imports: List[str] = field(
        default_factory=lambda: [
            "from triton.experimental import gluon",
            "from triton.experimental.gluon import language as ttgl",
        ]
    )


@dataclass(slots=True)
class LoweringContext:
    """Mutable lowering state for the active kernel."""

    kernel: AnnotatedKernel
    instructions: List[GluonInstruction] = field(default_factory=list)


class LoweringPipeline:
    """Translate annotated nodes into Gluon instructions."""

    def __init__(self, registry: MappingRegistry) -> None:
        self.registry = registry

    def lower(self, annotated: AnnotatedKernel) -> GluonModule:
        """Return a :class:`GluonModule` for ``annotated``."""

        ctx = LoweringContext(kernel=annotated)
        for node in annotated.walk():
            ctx.instructions.append(self._lower_node(node))
        kernel = GluonKernel(name=annotated.name, instructions=ctx.instructions)
        return GluonModule(kernels=[kernel])

    # ------------------------------------------------------------------ helpers
    def _lower_node(self, node: AnnotatedNode) -> GluonInstruction:
        """Create a placeholder Gluon instruction for ``node``."""

        spec = self.registry.lookup(node.op)
        if spec:
            comment = f"mapped via {spec.name}"
        else:
            comment = f"TODO: map {node.op}"
        return GluonInstruction(op="pass", comment=comment)


__all__ = [
    "GluonInstruction",
    "GluonKernel",
    "GluonModule",
    "LoweringPipeline",
]
