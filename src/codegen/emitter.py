"""Emit Gluon AST structures as Python source files."""

from __future__ import annotations

from pathlib import Path

from .lowering import GluonModule


class CodeEmitter:
    """Serialize :class:`GluonModule` objects to disk."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def emit(self, module: GluonModule, *, kernel_name: str) -> Path:
        """Write ``module`` to ``output_dir`` and return the file path.

        TODO: Replace the string builder with an AST-based emitter so formatting
        matches CPython's ``ast.unparse`` output.
        """

        path = self.output_dir / f"{kernel_name}.gluon.py"
        lines: list[str] = ["# Auto-generated Gluon kernel (skeleton)"]
        lines.extend(module.imports)
        lines.append("")
        for kernel in module.kernels:
            lines.extend(kernel.decorators)
            lines.append(f"def {kernel.name}(*args, **kwargs):")
            if not kernel.instructions:
                lines.append("    pass  # TODO: populate with lowered IR")
            else:
                for instr in kernel.instructions:
                    comment = f"  # {instr.comment}" if instr.comment else ""
                    lines.append(f"    {instr.op}({', '.join(instr.args)}){comment}")
            lines.append("")
        path.write_text("\n".join(lines) + "\n")
        return path
