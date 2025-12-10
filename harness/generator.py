#!/usr/bin/env python3
"""Deterministic harness generator for Triton kernels."""

from __future__ import annotations

import argparse
import ast
import json
import pprint
import textwrap
from pathlib import Path
from string import Template
from typing import Any


TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _format_literal(value: Any) -> str:
    return pprint.pformat(value, width=80, sort_dicts=False)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_kernel_node(source: str, kernel_name: str) -> ast.FunctionDef:
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == kernel_name:
            return node
    raise ValueError(f"Kernel function '{kernel_name}' not found in source file.")


def _is_constexpr(annotation: ast.AST | None) -> bool:
    if annotation is None:
        return False
    text = ast.unparse(annotation)
    normalized = text.replace(" ", "").lower()
    return "tl.constexpr" in normalized


def _analyze_kernel_signature(source: str, kernel_name: str) -> list[str]:
    node = _find_kernel_node(source, kernel_name)
    runtime_parameters: list[str] = []
    ordered_args = list(node.args.posonlyargs) + list(node.args.args)
    ordered_args += list(node.args.kwonlyargs)
    for arg in ordered_args:
        if _is_constexpr(arg.annotation):
            continue
        runtime_parameters.append(arg.arg)
    if node.args.vararg or node.args.kwarg:
        raise ValueError("Kernels using *args or **kwargs are not supported.")
    return runtime_parameters


def generate_harnesses(
    kernel_path: Path,
    configs_path: Path,
    tune_output: Path | None,
    compile_output: Path | None,
) -> tuple[Path, Path]:
    kernel_source = kernel_path.read_text(encoding="utf-8")
    config = _load_json(configs_path)
    runtime_parameters = _analyze_kernel_signature(kernel_source, config["kernel_name"])
    parameter_plan = config.get("parameters")
    if not parameter_plan:
        raise KeyError("Config must include a non-empty 'parameters' list.")
    plan_names = [entry["name"] for entry in parameter_plan]
    if plan_names != runtime_parameters:
        raise ValueError(
            "Config parameter ordering must match the kernel signature.\n"
            f" - Kernel parameters: {runtime_parameters}\n"
            f" - Config parameters: {plan_names}"
        )
    metadata = config.get("metadata") or {}
    if "grid" not in metadata:
        raise KeyError("metadata missing required key 'grid'")
    workload_names = [workload["name"] for workload in config.get("workloads", [])]
    if not workload_names:
        raise ValueError("Config file must define at least one workload.")
    grid_expr = textwrap.dedent(metadata["grid"]).strip()
    module_preamble = textwrap.dedent(metadata.get("module_preamble", "")).strip()
    autotune_keys = metadata.get("autotune_keys", [])
    context = {
        "kernel_name": config["kernel_name"],
        "kernel_filename": kernel_path.name,
        "config_literal": _format_literal(config),
        "workload_names_literal": _format_literal(workload_names),
        "parameter_plan_literal": _format_literal(parameter_plan),
        "autotune_keys_literal": _format_literal(autotune_keys),
        "grid_function_literal": repr(grid_expr),
        "module_preamble_literal": repr(module_preamble),
    }

    autotune_template = Template((TEMPLATE_DIR / "autotune.py.tmpl").read_text(encoding="utf-8"))
    compile_template = Template((TEMPLATE_DIR / "compile.py.tmpl").read_text(encoding="utf-8"))
    autotune_script = autotune_template.substitute(context)
    compile_script = compile_template.substitute(context)

    directory = kernel_path.parent
    stem, suffix = kernel_path.stem, kernel_path.suffix
    default_tune = directory / (f"{stem}_tune{suffix}" if suffix else f"{stem}_tune")
    default_compile = directory / (f"{stem}_compile{suffix}" if suffix else f"{stem}_compile")
    tune_target = tune_output or default_tune
    compile_target = compile_output or default_compile

    for target, contents in ((tune_target, autotune_script), (compile_target, compile_script)):
        target.parent.mkdir(parents=True, exist_ok=True)
        if not contents.endswith("\n"):
            contents += "\n"
        target.write_text(contents, encoding="utf-8")

    return tune_target, compile_target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic Triton autotune and compile harnesses.",
    )
    parser.add_argument("--kernel-path", required=True, type=Path)
    parser.add_argument("--configs", required=True, type=Path)
    parser.add_argument("--tune-output", type=Path, default=None)
    parser.add_argument("--compile-output", type=Path, default=None)
    args = parser.parse_args()
    tune_path, compile_path = generate_harnesses(
        args.kernel_path,
        args.configs,
        args.tune_output,
        args.compile_output,
    )
    print(f"Wrote autotune harness to {tune_path}")
    print(f"Wrote compile harness to {compile_path}")


if __name__ == "__main__":
    main()
