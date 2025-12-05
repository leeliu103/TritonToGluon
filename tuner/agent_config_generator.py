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
    invoke_agent,
    log_parse_failure_details,
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

    def build_prompt(self, kernel_path: str) -> str:
        kernel_path = os.path.abspath(kernel_path)
        template = (
            "You are an expert Triton tuning engineer tasked with proposing high-quality config "
            "candidates for a kernel.\n"
            f"The target kernel file path is: {kernel_path}\n"
            "Use Codex MCP tools to open and analyze the kernel. Inspect:\n"
            "  • Tunable parameters such as BLOCK_SIZE, TILE_M/N/K, etc.\n"
            "  • launch configuration requirements (num_warps, num_stages, num_ctas).\n"
            "  • Kernel arguments that influence tiling, vectorization, or shared-memory usage.\n"
            "Produce multiple candidate configs that are realistic for the given kernel. Each config "
            "must be a JSON object with concrete integer values for the relevant parameters.\n"
            "Return a single JSON object that matches this schema exactly:\n"
            "{\n"
            '  \"configs\": [\n'
            "    {\"BLOCK_SIZE\": 256, \"num_warps\": 4, \"num_stages\": 2},\n"
            "    {\"BLOCK_SIZE\": 512, \"num_warps\": 8, \"num_stages\": 3}\n"
            "  ],\n"
            '  \"notes\": \"Optional brief explanation of assumptions\"\n'
            "}\n"
            "Do NOT include markdown fences, commentary, or extra keys."
        )
        return template


class ConfigResponseParser:
    """Parses the JSON payload emitted by the config agent."""

    def parse(self, raw_response: str) -> ConfigGenerationResult:
        data = json.loads(raw_response.strip())
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
        self.codex_options = build_default_codex_options()

    async def generate_configs(self, kernel_path: str) -> ConfigGenerationResult:
        if not kernel_path:
            raise ValueError("kernel_path is required")

        normalized_path = os.path.abspath(kernel_path)
        if not os.path.exists(normalized_path):
            raise FileNotFoundError(f"Kernel file not found: {normalized_path}")

        prompt = self.prompt_builder.build_prompt(normalized_path)
        response_text, query_error = await invoke_agent(
            prompt, self.codex_options, normalized_path
        )

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
    print(f"Generating configs for kernel: {args.kernel_path}", file=sys.stderr)
    result = anyio.run(agent.generate_configs, args.kernel_path)

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
