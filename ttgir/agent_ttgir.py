#!/usr/bin/env python3
"""Minimal TTGIR metadata extraction agent."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

import anyio
from shared.agent_shared import (
    build_default_codex_options,
    extract_first_json_object,
    invoke_agent,
    log_parse_failure_details,
)


def derive_default_output_path(kernel_path: str) -> str:
    """Return default path `<kernel_name>_metadata.json` next to the kernel."""
    absolute_kernel_path = os.path.abspath(kernel_path)
    kernel_dir = os.path.dirname(absolute_kernel_path) or os.getcwd()
    kernel_name = os.path.basename(absolute_kernel_path) or "kernel"
    stem = os.path.splitext(kernel_name)[0] or kernel_name
    return os.path.join(kernel_dir, f"{stem}_metadata.json")


@dataclass
class TTGIRNodeMetadataResult:
    """In-memory representation of the TTGIR metadata response."""

    node_metadata: Dict[str, Any]


class TTGIRPromptBuilder:
    """Builds prompts for extracting TTGIR node metadata."""

    def build_prompt(
        self,
        kernel_path: str,
        mlir_log_path: str,
        node_location: str,
    ) -> str:
        kernel_path = os.path.abspath(kernel_path)
        mlir_log_path = os.path.abspath(mlir_log_path)
        template = (
            "You are an expert TTGIR/MLIR engineer tasked with extracting metadata for one "
            "operation emitted from a Triton kernel.\n"
            f"Triton kernel source path: {kernel_path}\n"
            f"MLIR / TTGIR log path: {mlir_log_path}\n"
            f"Target NodeLocation identifier: {node_location}\n"
            "Use Codex MCP tools (Read, Grep, etc.) to:\n"
            "  • Open and understand the Triton kernel to gain context on argument shapes and tunables.\n"
            "  • Read the MLIR log file in detail.\n"
            "  • Locate the operation that matches the provided NodeLocation string.\n"
            "  • Extract precise metadata for that operation, including op name, operand/result types, "
            "attributes, location info, and any other relevant properties visible in the MLIR.\n"
            "Return a single JSON object that matches this schema exactly:\n"
            "{\n"
            '  "node_metadata": {\n'
            '    "operation": "tt.load or arith.addf",\n'
            '    "operands": [{"name": "%x", "type": "tensor<...>"}],\n'
            '    "result_types": ["tensor<...>"],\n'
            '    "attributes": {"cache": "cg"},\n'
            '    "location": "file.py:123:4",\n'
            '    "notes": "Optional clarifications or observations."\n'
            "  }\n"
            "}\n"
            "You may add additional key-value pairs inside node_metadata if the MLIR contains "
            "useful information (e.g., encodings, layouts, constants). Do NOT include markdown fences, "
            "explanations outside of node_metadata, or extraneous top-level keys."
        )
        return template


class TTGIRResponseParser:
    """Parses the JSON payload emitted by the TTGIR agent."""

    def parse(self, raw_response: str) -> TTGIRNodeMetadataResult:
        json_str = extract_first_json_object(raw_response)
        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise ValueError("Agent response must be a JSON object")

        metadata = data.get("node_metadata")
        if not isinstance(metadata, dict) or not metadata:
            raise ValueError("Agent response must include a non-empty 'node_metadata' object")

        normalized_metadata = {str(key): metadata[key] for key in metadata}
        return TTGIRNodeMetadataResult(node_metadata=normalized_metadata)


class TTGIRAgent:
    """Agent wrapper that extracts TTGIR metadata for a specific node."""

    def __init__(
        self,
        prompt_builder: Optional[TTGIRPromptBuilder] = None,
        response_parser: Optional[TTGIRResponseParser] = None,
    ) -> None:
        self.prompt_builder = prompt_builder or TTGIRPromptBuilder()
        self.response_parser = response_parser or TTGIRResponseParser()
        self.codex_options = build_default_codex_options()

    async def extract_node_metadata(
        self,
        kernel_path: str,
        mlir_log_path: str,
        node_location: str,
    ) -> TTGIRNodeMetadataResult:
        if not kernel_path:
            raise ValueError("kernel_path is required")
        if not mlir_log_path:
            raise ValueError("mlir_log_path is required")
        if not node_location:
            raise ValueError("node_location is required")

        normalized_kernel = os.path.abspath(kernel_path)
        normalized_log = os.path.abspath(mlir_log_path)

        if not os.path.exists(normalized_kernel):
            raise FileNotFoundError(f"Kernel file not found: {normalized_kernel}")
        if not os.path.exists(normalized_log):
            raise FileNotFoundError(f"MLIR log file not found: {normalized_log}")

        prompt = self.prompt_builder.build_prompt(
            normalized_kernel,
            normalized_log,
            node_location,
        )
        context_label = f"{normalized_log}:{node_location}"
        response_text, query_error = await invoke_agent(
            prompt,
            self.codex_options,
            context_label,
        )

        if query_error:
            log_parse_failure_details(
                context_label=context_label,
                query_error=query_error,
                agent_response=response_text,
            )
            raise query_error

        try:
            return self.response_parser.parse(response_text)
        except ValueError:
            log_parse_failure_details(
                context_label=context_label,
                query_error=query_error,
                agent_response=response_text,
            )
            raise


def write_metadata_file(node_metadata: Dict[str, Any], output_path: str) -> str:
    """Persist the metadata JSON to disk and return the absolute file path."""
    if not output_path:
        raise ValueError("output_path is required")
    if not isinstance(node_metadata, dict) or not node_metadata:
        raise ValueError("node_metadata must be a non-empty dictionary")

    absolute_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)

    with open(absolute_path, "w", encoding="utf-8") as handle:
        json.dump({"node_metadata": node_metadata}, handle, indent=2)
        handle.write("\n")

    print(f"Wrote node metadata to {absolute_path}", file=sys.stderr)
    return absolute_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract TTGIR node metadata for a Triton kernel using an AI agent.",
    )
    parser.add_argument(
        "--kernel-path",
        required=True,
        help="Path to the Triton kernel source file.",
    )
    parser.add_argument(
        "--mlir-log",
        required=True,
        help="Path to the TTGIR/MLIR log containing the NodeLocation.",
    )
    parser.add_argument(
        "--node-location",
        required=True,
        help="NodeLocation string that identifies the target operation.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "File path for the extracted metadata. Defaults to placing "
            "a '<kernel>_metadata.json' file next to the kernel."
        ),
    )
    args = parser.parse_args()
    if not args.output:
        args.output = derive_default_output_path(args.kernel_path)
    return args


def _run_cli() -> None:
    args = _parse_args()
    agent = TTGIRAgent()
    print(
        "Extracting TTGIR node metadata:\n"
        f"  kernel: {args.kernel_path}\n"
        f"  mlir log: {args.mlir_log}\n"
        f"  node: {args.node_location}",
        file=sys.stderr,
    )
    result = anyio.run(
        agent.extract_node_metadata,
        args.kernel_path,
        args.mlir_log,
        args.node_location,
    )

    saved_path = write_metadata_file(result.node_metadata, args.output)

    payload = {
        "node_metadata": result.node_metadata,
        "metadata_file": saved_path,
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    _run_cli()
