"""Emit Gluon AST structures as Python source files."""

from __future__ import annotations

from pathlib import Path

from .generator import GluonModule, _BASE_IMPORTS


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

        import_block = module.imports or list(_BASE_IMPORTS)
        lines.extend(import_block)
        if import_block:
            lines.append("")

        if module.constexprs:
            lines.extend(module.constexprs)
            lines.append("")

        for kernel in module.kernels:
            lines.append(kernel.source)
            lines.append("")

        path.write_text("\n".join(lines).rstrip() + "\n")
        return path
