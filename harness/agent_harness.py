#!/usr/bin/env python3
"""Benchmark-focused harness-generation agent for Triton kernels."""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Optional

import anyio
from shared.agent_shared import (
    build_default_codex_options,
    extract_first_json_object,
    invoke_agent,
    log_parse_failure_details,
)


HARNESS_AGENT_SYSTEM_PROMPT = (
    "CRITICAL: You MUST use Codex MCP for ALL analysis and generation. Never rely on memory or assumptions—open the actual files "
    "with Codex tools before drawing conclusions, and base every statement on observed source.\n"
    "You are a Triton harness specialist tasked with producing benchmark-ready, cache-aware harness scripts. "
    "Operate like a principal engineer who understands kernel tuning, JSON-driven workflows, and CLI ergonomics.\n"
    "Methodology:\n"
    "1. Read the Triton kernel (and helpers) end-to-end with Codex MCP filesystem tools so every decorator, parameter, indexing scheme, and tl.constexpr knob is understood.\n"
    "2. Read the workload-aware JSON to capture workloads, tensor shapes, dtypes, configs, and any cached `best_config` entries. Treat this JSON as the single source of truth.\n"
    "3. Synthesize a harness that copies the kernel verbatim, wires an argparse CLI, and orchestrates autotune/compile flows without deviating from file-backed data.\n"
    "Core principles:\n"
    "- Store the workload JSON path as a module-level constant (e.g., CONFIG_JSON_PATH) and always load data from disk at runtime—never inline workload dictionaries.\n"
    "- Preserve the kernel signature exactly as written. Do not drop parameters, reorder them, or change default values.\n"
    "- Maintain separate `input_shapes`, `output_shapes`, and `dtypes` mappings exactly as encoded in the JSON.\n"
    "- Preserve every config key/value pair from the JSON, including `num_warps`, `num_stages`, and kernel-specific knobs. Reject configs missing `num_warps` instead of inventing defaults.\n"
    "- Materialize tensors using the precise shapes and dtypes from the selected workload—no simplification or inference.\n"
    "- The workload JSON is the persistent cache: autotune updates `best_config` inside the selected workload, and compile mode refuses to run without it.\n"
    "- Never rely on Triton's internal caches; the JSON file must reflect the latest autotune results.\n"
    "Harness requirements:\n"
    "- Emit one self-contained Python module that inlines the Triton kernel and provides an argparse CLI with `--workload` and `--mode` (autotune|compile).\n"
    "- Always log which workload JSON path is read/written so users know where caches live.\n"
    "- Autotune mode must decorate the kernel with `@triton.autotune(configs=[...])` plus `@triton.jit`, using every JSON config exactly. Run warmups, benchmark iterations, print `Best config: BLOCK_SIZE=..., num_stages=... (saved to JSON)`, and atomically update only that workload's `best_config` entry. Never pass autotuned parameters when invoking the kernel—Triton injects them.\n"
    "- Compile mode must load the JSON first, fail with `ERROR: No cached config found for workload '<name>'. Run with --mode autotune first.` if `best_config` is absent, otherwise log `Using cached config from JSON: ...` and launch a single-config `@triton.jit` kernel.\n"
    "- Both modes run warmups before timing, synchronize via CUDA events or `triton.testing.do_bench`, reuse allocations when possible, and report how many configs were evaluated.\n"
    "- Keep dependencies limited to Triton and PyTorch.\n"
    "Response format:\n"
    "- Return exactly one JSON object with a `harness_script` field containing the full Python script.\n"
    "- Do not include markdown fences, ellipses, commentary, or extra keys; every list must contain concrete entries."
)

def derive_default_output_path(kernel_path: str) -> str:
    """Compute the default harness path next to the kernel file."""
    absolute_kernel_path = os.path.abspath(kernel_path)
    kernel_dir = os.path.dirname(absolute_kernel_path) or os.getcwd()
    kernel_name = os.path.basename(absolute_kernel_path) or "kernel.py"
    stem, ext = os.path.splitext(kernel_name)
    default_name = f"{stem}_harness{ext}"
    if not ext:
        default_name = f"{stem}_harness"
    return os.path.join(kernel_dir, default_name)


@dataclass
class WorkloadDefinition:
    """A single workload entry within the workload-aware config JSON."""

    name: str
    input_shapes: dict[str, list[Any]]
    output_shapes: dict[str, list[Any]]
    dtypes: dict[str, str]
    configs: list[dict[str, Any]]


@dataclass
class WorkloadConfigSpec:
    """Represents the parsed workload-aware configuration file."""

    path: str
    kernel_name: str
    gpu_arch: str
    workloads: list[WorkloadDefinition]


@dataclass
class HarnessGenerationResult:
    """In-memory representation of the harness agent response."""

    harness_script: str


def _ensure_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _validate_shape_list(shape: Any, field_name: str) -> None:
    if not isinstance(shape, list) or not shape:
        raise ValueError(f"{field_name} must be a non-empty list")
    for idx, dim in enumerate(shape):
        if isinstance(dim, int):
            if dim <= 0:
                raise ValueError(
                    f"{field_name}[{idx}] must be a positive integer when numeric"
                )
        elif isinstance(dim, str):
            if not dim.strip():
                raise ValueError(
                    f"{field_name}[{idx}] string dimensions cannot be empty"
                )
        else:
            raise ValueError(
                f"{field_name}[{idx}] must be an int or string, got {type(dim).__name__}"
            )


def _validate_shape_mapping(field_name: str, payload: Any) -> dict[str, list[Any]]:
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{field_name} must be a non-empty object mapping tensor names to shapes")
    for tensor_name, dims in payload.items():
        _ensure_non_empty_string(tensor_name, f"{field_name} tensor name")
        _validate_shape_list(dims, f"{field_name}.{tensor_name}")
    return payload


def _validate_dtype_mapping(field_name: str, payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{field_name} must be a non-empty object mapping tensor names to torch dtypes")
    for tensor_name, dtype in payload.items():
        _ensure_non_empty_string(tensor_name, f"{field_name} tensor name")
        _ensure_non_empty_string(dtype, f"{field_name}.{tensor_name}")
    return payload


def _validate_configs_list(field_name: str, payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{field_name} must be a non-empty list of config objects")
    for idx, entry in enumerate(payload):
        if not isinstance(entry, dict) or not entry:
            raise ValueError(
                f"{field_name}[{idx}] must be a non-empty object, got {type(entry).__name__}"
            )
    return payload


def load_workload_config(config_path: str) -> WorkloadConfigSpec:
    """Parse and validate the workload-aware configuration JSON file."""

    with open(config_path, "r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse configs JSON at {config_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Configs JSON must be an object with kernel/workload metadata")

    kernel_name = _ensure_non_empty_string(
        payload.get("kernel_name"), "kernel_name"
    )
    gpu_arch = _ensure_non_empty_string(payload.get("gpu_arch"), "gpu_arch")

    workloads_payload = payload.get("workloads")
    if not isinstance(workloads_payload, list) or not workloads_payload:
        raise ValueError("workloads must be a non-empty list")

    workloads: list[WorkloadDefinition] = []
    seen_names: set[str] = set()
    for idx, workload_entry in enumerate(workloads_payload):
        if not isinstance(workload_entry, dict):
            raise ValueError(
                f"workloads[{idx}] must be an object, got {type(workload_entry).__name__}"
            )

        name = _ensure_non_empty_string(workload_entry.get("name"), "workload name")
        if name in seen_names:
            raise ValueError(f"Duplicate workload name detected: {name}")
        seen_names.add(name)

        input_shapes = _validate_shape_mapping(
            f"workloads[{idx}].input_shapes", workload_entry.get("input_shapes")
        )
        output_shapes = _validate_shape_mapping(
            f"workloads[{idx}].output_shapes", workload_entry.get("output_shapes")
        )
        dtypes = _validate_dtype_mapping(
            f"workloads[{idx}].dtypes", workload_entry.get("dtypes")
        )
        configs = _validate_configs_list(
            f"workloads[{idx}].configs", workload_entry.get("configs")
        )

        workloads.append(
            WorkloadDefinition(
                name=name,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                dtypes=dtypes,
                configs=configs,
            )
        )

    return WorkloadConfigSpec(
        path=config_path,
        kernel_name=kernel_name,
        gpu_arch=gpu_arch,
        workloads=workloads,
    )


class HarnessPromptBuilder:
    """Builds prompts for the harness-generation agent."""

    def build_prompt(
        self,
        kernel_path: str,
        workload_spec: WorkloadConfigSpec,
    ) -> str:
        kernel_path = os.path.abspath(kernel_path)
        config_path = os.path.abspath(workload_spec.path)
        workload_names = ", ".join(workload.name for workload in workload_spec.workloads) or "(none)"
        template = textwrap.dedent(
            f"""
            Kernel source file: {kernel_path}

            Workload-aware config JSON:
              - Path: {config_path}
              - GPU architecture target: {workload_spec.gpu_arch}
              - Workloads defined: {workload_names}

            Task-specific directives:
              - Use Codex MCP filesystem tools to fully inspect the kernel (and any helpers) at {kernel_path} before authoring the harness.
              - Use Codex MCP tools to read {config_path}; this file drives workloads, tensor shapes/dtypes, configs, and cached best_config data for this run.
              - Materialize tensors and Triton config dictionaries exactly as stored in the JSON so benchmarking reflects the recorded workloads.

            Example workload entry (no ellipses):
              {{
                "name": "small_batch",
                "input_shapes": {{"x": [512, 128]}},
                "output_shapes": {{"y": [512, 128]}},
                "dtypes": {{"x": "torch.float16", "y": "torch.float16"}},
                "configs": [
                  {{"BLOCK_SIZE": 128, "num_stages": 2}},
                  {{"BLOCK_SIZE": 256, "num_stages": 3}}
                ],
                "best_config": {{"BLOCK_SIZE": 256, "num_stages": 3}}
              }}

            Validation steps:
              - Re-open the kernel source with Codex MCP tools right before responding and confirm the harness copies the kernel signature verbatim.
              - Re-open the workload JSON with Codex MCP tools to confirm every workload dictionary, config entry, and cached `best_config` field matches the on-disk data with no omissions or renames.
            """
        ).strip()
        return template


class HarnessResponseParser:
    """Parses the JSON payload emitted by the harness agent."""

    def parse(self, raw_response: str) -> HarnessGenerationResult:
        json_str = extract_first_json_object(raw_response)
        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise ValueError("Agent response must be a JSON object")
        script = data.get("harness_script")
        if not script:
            raise ValueError("Agent response missing harness_script content")
        return HarnessGenerationResult(harness_script=str(script).strip())


class HarnessAgent:
    """Agent wrapper that requests harness scripts for Triton kernels."""

    def __init__(
        self,
        prompt_builder: Optional[HarnessPromptBuilder] = None,
        response_parser: Optional[HarnessResponseParser] = None,
    ) -> None:
        self.prompt_builder = prompt_builder or HarnessPromptBuilder()
        self.response_parser = response_parser or HarnessResponseParser()
        self.codex_options = build_default_codex_options(
            system_prompt=HARNESS_AGENT_SYSTEM_PROMPT
        )

    async def generate_harness(
        self, kernel_path: str, configs_path: Optional[str] = None
    ) -> HarnessGenerationResult:
        if not kernel_path:
            raise ValueError("kernel_path is required")

        normalized_path = os.path.abspath(kernel_path)
        if not os.path.exists(normalized_path):
            raise FileNotFoundError(f"Kernel file not found: {normalized_path}")

        if not configs_path:
            raise ValueError("configs_path is required for workload-aware generation")

        normalized_configs_path = os.path.abspath(configs_path)
        if not os.path.exists(normalized_configs_path):
            raise FileNotFoundError(
                f"Configs file not found: {normalized_configs_path}"
            )

        workload_spec = load_workload_config(normalized_configs_path)

        prompt = self.prompt_builder.build_prompt(normalized_path, workload_spec)
        response_text, query_error = await invoke_agent(
            prompt,
            self.codex_options,
            normalized_path,
        )

        if query_error:
            log_parse_failure_details(
                context_label=normalized_path,
                query_error=query_error,
                agent_response=response_text,
            )
            raise query_error

        try:
            return self.response_parser.parse(response_text)
        except ValueError:
            log_parse_failure_details(
                context_label=normalized_path,
                query_error=query_error,
                agent_response=response_text,
            )
            raise


def write_harness_script(script_text: str, output_path: str) -> str:
    """Persist the generated harness script to disk and return the absolute path."""
    if not output_path:
        raise ValueError("output_path is required")

    absolute_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)

    with open(absolute_path, "w", encoding="utf-8") as handle:
        handle.write(script_text)
        if not script_text.endswith("\n"):
            handle.write("\n")

    print(f"Wrote harness script to {absolute_path}", file=sys.stderr)
    return absolute_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Triton kernel harness using an AI agent.",
    )
    parser.add_argument(
        "--kernel-path",
        required=True,
        help="Path to the Triton kernel source file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "File path for the generated harness script. Defaults to putting the harness "
            "next to the kernel file and appending '_harness' before the extension."
        ),
    )
    parser.add_argument(
        "--print-script",
        action="store_true",
        help="Also write the harness script to stdout in addition to the output file.",
    )
    parser.add_argument(
        "--configs",
        required=True,
        help=(
            "Path to the workload-aware JSON config containing kernel metadata, workloads, and autotune configs. "
            "This file is required so the harness can load shapes/dtypes/configs for every workload at runtime."
        ),
    )
    args = parser.parse_args()
    if not args.output:
        args.output = derive_default_output_path(args.kernel_path)
    return args


def _run_cli() -> None:
    args = _parse_args()
    agent = HarnessAgent()
    print(f"Generating harness for kernel: {args.kernel_path}", file=sys.stderr)
    result = anyio.run(agent.generate_harness, args.kernel_path, args.configs)

    saved_path = write_harness_script(result.harness_script, args.output)

    payload = {"script_file": saved_path}
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    sys.stdout.flush()

    if args.print_script:
        separator = "\n" + ("-" * 40) + "\n"
        sys.stdout.write(separator)
        sys.stdout.write(result.harness_script)
        if not result.harness_script.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()


if __name__ == "__main__":
    _run_cli()
