#!/usr/bin/env python3
"""Minimal config-generation agent for Triton kernels."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import anyio
from shared.agent_shared import (
    build_default_codex_options,
    extract_first_json_object,
    invoke_agent,
    log_parse_failure_details,
)


CONFIG_AGENT_SYSTEM_PROMPT = (
    "You are a principal GPU kernel performance engineer and Triton autotuning specialist. "
    "Always begin by opening the referenced Triton kernel with Codex MCP filesystem tools (read_file, rg) and study the code until every decorator, function parameter, tl.constexpr knob, loop, and tl.load/store is understood. "
    "Base every conclusion strictly on facts visible in the source—do not speculate about compiler lowering, cache behavior, or register usage unless the code states it explicitly.\n"
    "Methodology:\n"
    "1. Catalog all tunable/constexpr parameters (BLOCK_SIZE, TILE_M/N/K, GROUP_SIZE, VECTOR_WIDTH, num_warps, num_stages, etc.), their defaults, and any constraints expressed in the code (assertions, divisibility, min/max values).\n"
    "2. Identify the algorithm type (matmul, reduction, elementwise, attention, etc.) and summarize the control flow, indexing math, and data shapes purely from what the kernel computes.\n"
    "3. Note relationships between parameters and problem sizes that must hold (e.g., BLOCK_M divides M, BLOCK_SIZE matches stride requirements) and capture any tl.static_assert or boundary-guard logic.\n"
    "4. Apply practical GPU heuristics tied to the stated architecture—warp granularity (32 NVIDIA / 64 AMD), preference for powers of two, balanced tile aspect ratios, reasonable num_warps/num_stages combos—only insofar as they align with the visible code constraints.\n"
    "5. Determine an appropriate number of configs (typically 5-20) by weighing the count of tunable knobs, the breadth of their valid ranges, kernel complexity, and expected autotuning coverage.\n"
    "6. Design that many diverse candidate configs (small/medium/large tiles, varying warp/stage counts) that all satisfy the kernel's requirements and would provide meaningful coverage for autotuning.\n"
    "7. Explain the intent of the configs using references to the kernel's logic (e.g., loop bounds, tensor blocking) rather than unverifiable hardware speculation.\n"
    "Response requirements:\n"
    "- Produce exactly one JSON object containing a `configs` list and an optional `notes` string summarizing the explored trade-offs.\n"
    "- Each config dictionary must map the kernel's compile-time parameters to concrete integers that respect the kernel constraints and basic hardware heuristics.\n"
    "- Justify the chosen config count inside the `notes` field and ensure the number of emitted configs matches that justification.\n"
    "- Do not include markdown fences, commentary outside the JSON object, or extra keys."
)


def derive_default_output_path(kernel_path: str) -> str:
    """Return default path `<kernel_name>_configs.json` next to the kernel."""
    absolute_kernel_path = os.path.abspath(kernel_path)
    kernel_dir = os.path.dirname(absolute_kernel_path) or os.getcwd()
    kernel_name = os.path.basename(absolute_kernel_path) or "kernel"
    stem = os.path.splitext(kernel_name)[0] or kernel_name
    return os.path.join(kernel_dir, f"{stem}_configs.json")


@dataclass
class ConfigGenerationResult:
    """In-memory representation of the config agent response."""

    configs: List[Dict[str, Any]]
    notes: str = ""


class ConfigPromptBuilder:
    """Builds prompts for the config-generation agent."""

    def build_prompt(self, kernel_path: str, gpu_arch: str) -> str:
        kernel_path = os.path.abspath(kernel_path)
        template = (
            "Prepare Triton tuning configs for the specified kernel.\n"
            f"Target kernel path: {kernel_path}\n"
            f"Target GPU architecture: {gpu_arch}\n"
            "Follow this step-by-step process:\n"
            "1. Use Codex MCP filesystem tools (read_file, read_directory, rg) to open the kernel at the provided path. Confirm the function signature, @triton.jit decorator arguments, and every tl.constexpr parameter or module-level constant.\n"
            "2. List every compile-time/tunable knob the kernel exposes (BLOCK_SIZE, TILE_M/N/K, GROUP_SIZE_M, VECTOR_WIDTH, num_warps, num_stages, etc.) along with any constraints stated in the code (assertions, tl.static_assert, divisibility rules, boundary guards).\n"
            "3. Describe the algorithm purely from the visible source: tensor shapes, indexing math, loop structure, tl.load/tl.store usage, and synchronization that appears in the code. Avoid speculation about compiler-inserted behavior.\n"
            "4. Derive valid ranges or preferred patterns for each tunable using the observed relationships plus basic GPU heuristics for the specified architecture (warp multiples of 32/64, powers-of-two tile sizes, practical num_warps/num_stages combinations).\n"
            "5. Decide how many configs to emit (typically 5-20) by weighing the number of tunables, the breadth of their valid ranges, kernel complexity, and expected autotuning coverage. Do not guess—justify the count using facts from the source and practical heuristics.\n"
            "6. Produce that many distinct configs that respect every visible constraint and cover diverse tile scales or warp/stage selections so the tuner can explore different regions.\n"
            "7. Summarize notable trade-offs or assumptions in a short notes string, explicitly explaining why the selected config count is appropriate for this kernel.\n"
            "Do not attempt to infer register counts, cache hit rates, or memory-stage movement beyond what the source explicitly expresses.\n"
            "Output instructions:\n"
            "- Return exactly one JSON object with:\n"
            "{\n"
            '  "configs": [<config dicts equal to the justified count>],\n'
            '  "notes": "Brief rationale covering trade-offs, knob constraints, and the reasoning behind the chosen count."\n'
            "}\n"
            "- Each config dictionary must fill in every relevant compile-time knob with concrete integers tailored to the kernel; omit knobs that are not declared.\n"
            "- The number of configs emitted must match the count defended in `notes`.\n"
            "- Do not include markdown fences, commentary, or extra keys."
        )
        return template


class ConfigResponseParser:
    """Parses the JSON payload emitted by the config agent."""

    def parse(self, raw_response: str) -> ConfigGenerationResult:
        json_str = extract_first_json_object(raw_response)
        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise ValueError("Agent response must be a JSON object")

        configs = data.get("configs")
        if not isinstance(configs, list) or not configs:
            raise ValueError("Agent response must include a non-empty 'configs' list")

        normalized_configs: List[Dict[str, Any]] = []
        for entry in configs:
            if not isinstance(entry, dict) or not entry:
                raise ValueError("Each config must be a non-empty JSON object")
            normalized_configs.append({str(k): entry[k] for k in entry})

        notes = data.get("notes", "")
        if notes is None:
            notes = ""

        return ConfigGenerationResult(
            configs=normalized_configs,
            notes=str(notes).strip(),
        )


class ConfigAgent:
    """Agent wrapper that requests config candidates for Triton kernels."""

    def __init__(
        self,
        prompt_builder: Optional[ConfigPromptBuilder] = None,
        response_parser: Optional[ConfigResponseParser] = None,
    ) -> None:
        self.prompt_builder = prompt_builder or ConfigPromptBuilder()
        self.response_parser = response_parser or ConfigResponseParser()
        self.codex_options = build_default_codex_options(
            system_prompt=CONFIG_AGENT_SYSTEM_PROMPT
        )

    async def generate_configs(
        self, kernel_path: str, gpu_arch: str
    ) -> ConfigGenerationResult:
        if not kernel_path:
            raise ValueError("kernel_path is required")
        if not gpu_arch:
            raise ValueError("gpu_arch is required")

        normalized_path = os.path.abspath(kernel_path)
        if not os.path.exists(normalized_path):
            raise FileNotFoundError(f"Kernel file not found: {normalized_path}")

        prompt = self.prompt_builder.build_prompt(normalized_path, gpu_arch)
        response_text, query_error = await invoke_agent(
            prompt, self.codex_options, normalized_path
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


def write_config_file(configs: List[Dict[str, Any]], output_path: str) -> str:
    """Write the configs list to disk and return the absolute path."""
    if not output_path:
        raise ValueError("output_path is required")
    if not isinstance(configs, list) or not configs:
        raise ValueError("configs must be a non-empty list")

    absolute_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)

    with open(absolute_path, "w", encoding="utf-8") as handle:
        json.dump(configs, handle, indent=2)
        handle.write("\n")

    print(f"Wrote config list to {absolute_path}", file=sys.stderr)
    return absolute_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Triton kernel tuning configs using an AI agent.",
    )
    parser.add_argument(
        "--kernel-path",
        required=True,
        help="Path to the Triton kernel source file.",
    )
    parser.add_argument(
        "--gpu-arch",
        required=True,
        help="GPU architecture identifier (e.g., rdna4, rdna3, mi300) to target tuning heuristics.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "File path for the generated config list. Defaults to placing "
            "a '<kernel>_configs.json' file next to the kernel."
        ),
    )
    parser.add_argument(
        "--print-configs",
        action="store_true",
        help="Print the config list to stdout after writing it to disk.",
    )
    args = parser.parse_args()
    if not args.output:
        args.output = derive_default_output_path(args.kernel_path)
    return args


def _run_cli() -> None:
    args = _parse_args()
    agent = ConfigAgent()
    print(
        (
            f"Generating configs for kernel: {args.kernel_path} "
            f"(GPU arch: {args.gpu_arch})"
        ),
        file=sys.stderr,
    )
    result = anyio.run(
        agent.generate_configs,
        args.kernel_path,
        args.gpu_arch,
    )

    saved_path = write_config_file(result.configs, args.output)

    payload = {
        "config_file": saved_path,
        "config_count": len(result.configs),
        "notes": result.notes,
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    sys.stdout.flush()

    if args.print_configs:
        separator = "\n" + ("-" * 40) + "\n"
        sys.stdout.write(separator)
        json.dump(result.configs, sys.stdout, indent=2)
        sys.stdout.write("\n")
        sys.stdout.flush()


if __name__ == "__main__":
    _run_cli()
