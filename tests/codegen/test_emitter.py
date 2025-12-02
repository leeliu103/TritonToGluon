"""Tests for the simple Gluon code emitter."""

from __future__ import annotations

import ast
from pathlib import Path

from src.codegen.emitter import CodeEmitter
from src.codegen.generator import GluonKernel, GluonModule, _BASE_IMPORTS


def _read(path: Path) -> list[str]:
    return path.read_text().splitlines()


def test_emit_writes_full_module(tmp_path: Path) -> None:
    """Emitter writes imports, constexprs, and kernels in order."""

    module = GluonModule(
        kernels=[GluonKernel(name="foo", source="@gluon.jit\ndef foo():\n    return 1")],
        imports=["import math"],
        constexprs=["VALUE = 3"],
    )
    emitter = CodeEmitter(tmp_path)
    path = emitter.emit(module, kernel_name="foo")
    lines = _read(path)
    assert lines[0].startswith("# Auto-generated Gluon kernel")
    assert "import math" in lines
    assert "VALUE = 3" in lines
    assert "def foo" in path.read_text()


def test_emit_falls_back_to_base_imports(tmp_path: Path) -> None:
    """When ``module.imports`` is empty the emitter uses the base defaults."""

    module = GluonModule(kernels=[GluonKernel(name="foo", source="def foo():\n    pass")])
    emitter = CodeEmitter(tmp_path)
    path = emitter.emit(module, kernel_name="foo")
    lines = _read(path)
    for base in _BASE_IMPORTS:
        assert base in lines


def test_emit_outputs_parseable_python(tmp_path: Path) -> None:
    """Round-trip compile ensures the serialized module is valid Python."""

    module = GluonModule(
        kernels=[
            GluonKernel(
                name="foo",
                source="""@gluon.constexpr_function\ndef foo(x):\n    return x""",
            ),
            GluonKernel(name="bar", source="def bar():\n    return foo(1)"),
        ],
    )
    path = CodeEmitter(tmp_path).emit(module, kernel_name="combo")
    ast.parse(path.read_text())
