#!/usr/bin/env python3
"""Minimal harness-generation agent for Triton kernels."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import anyio
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
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
class HarnessGenerationResult:
    """In-memory representation of the harness agent response."""

    harness_script: str
    assumptions: str = ""


class HarnessPromptBuilder:
    """Builds prompts for the harness-generation agent."""

    def build_prompt(self, kernel_path: str) -> str:
        kernel_path = os.path.abspath(kernel_path)
        template = (
            "You are an expert Triton engineer tasked with creating a runnable harness for a kernel.\n"
            f"The kernel source file is located at: {kernel_path}\n"
            "Use Codex MCP tools to read and understand the kernel file so you know how to launch it.\n"
            "Produce a minimal, human-readable Python script that:\n"
            "  1. Imports the kernel via importlib from the provided path.\n"
            "  2. Builds simple sample tensors or scalars that satisfy the kernel signature.\n"
            "  3. Chooses a reasonable launch grid/block configuration.\n"
            "  4. Invokes the kernel once inside a `main()` function guarded by `if __name__ == \"__main__\":`.\n"
            "  5. Only depends on Triton and PyTorch (no extra libraries).\n"
            "Include helpful comments when you must make assumptions about tensor shapes or argument values.\n"
            "Return a single JSON object that matches exactly this schema:\n"
            "{\n"
            '  \"harness_script\": \"#!/usr/bin/env python3\\n... full script ...\",\n'
            '  \"assumptions\": \"Optional notes about placeholders or launch params\"\n'
            "}\n"
            "Do NOT include markdown fences, explanations, or additional fields."
        )
        return template


class HarnessResponseParser:
    """Parses the JSON payload emitted by the harness agent."""

    def parse(self, raw_response: str) -> HarnessGenerationResult:
        data = json.loads(raw_response.strip())
        if not isinstance(data, dict):
            raise ValueError("Agent response must be a JSON object")
        script = data.get("harness_script")
        if not script:
            raise ValueError("Agent response missing harness_script content")
        assumptions = data.get("assumptions", "")
        return HarnessGenerationResult(
            harness_script=str(script).strip(),
            assumptions=str(assumptions).strip(),
        )


class HarnessAgent:
    """Agent wrapper that requests harness scripts for Triton kernels."""

    def __init__(
        self,
        prompt_builder: Optional[HarnessPromptBuilder] = None,
        response_parser: Optional[HarnessResponseParser] = None,
    ) -> None:
        self.prompt_builder = prompt_builder or HarnessPromptBuilder()
        self.response_parser = response_parser or HarnessResponseParser()
        self.codex_options = ClaudeAgentOptions(
            allowed_tools=["mcp__codex", "Read", "Grep"],
            mcp_servers={
                "codex": {
                    "type": "stdio",
                    "command": "npx",
                    "args": [
                        "-y",
                        "@openai/codex",
                        "-c",
                        'model_provider="amd-openai"',
                        "mcp-server",
                    ],
                }
            },
            system_prompt=(
                "You are a careful systems engineer. "
                "Respond with a single JSON object that matches the caller's schema. "
                "Do not include markdown fences or commentary."
            ),
            max_turns=15,
        )

    async def generate_harness(self, kernel_path: str) -> HarnessGenerationResult:
        if not kernel_path:
            raise ValueError("kernel_path is required")

        normalized_path = os.path.abspath(kernel_path)
        if not os.path.exists(normalized_path):
            raise FileNotFoundError(f"Kernel file not found: {normalized_path}")

        prompt = self.prompt_builder.build_prompt(normalized_path)
        response_text, query_error = await self._invoke_agent(prompt, normalized_path)

        try:
            return self.response_parser.parse(response_text)
        except ValueError as exc:
            if query_error:
                print(
                    f"Warning: Agent query failed for {normalized_path}: {query_error}",
                    file=sys.stderr,
                )
            snippet = response_text.replace("\n", " ")[:400] if response_text else ""
            if snippet:
                print(f"Agent response snippet: {snippet}...", file=sys.stderr)
            raise

    async def _invoke_agent(self, prompt: str, kernel_path: str) -> Tuple[str, Optional[Exception]]:
        response_text = ""
        query_error: Optional[Exception] = None

        try:
            async for message in query(prompt=prompt, options=self.codex_options):
                if isinstance(message, AssistantMessage):
                    blocks = getattr(message, "content", None) or []
                    response_text += "".join(
                        block.text for block in blocks if isinstance(block, TextBlock)
                    )
                elif isinstance(message, ResultMessage):
                    blocks = getattr(message, "content", None) or []
                    response_text += "".join(
                        block.text for block in blocks if isinstance(block, TextBlock)
                    )
                    payload = getattr(message, "result", None)
                    if payload:
                        if isinstance(payload, str):
                            response_text += payload
                        else:
                            response_text += json.dumps(payload)
        except Exception as exc:  # pragma: no cover - guard against agent issues
            query_error = exc
            print(f"Warning: Agent streaming failed for {kernel_path}: {exc}", file=sys.stderr)

        return response_text, query_error


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
    args = parser.parse_args()
    if not args.output:
        args.output = derive_default_output_path(args.kernel_path)
    return args


def _run_cli() -> None:
    args = _parse_args()
    agent = HarnessAgent()
    print(f"Generating harness for kernel: {args.kernel_path}", file=sys.stderr)
    result = anyio.run(agent.generate_harness, args.kernel_path)

    saved_path = write_harness_script(result.harness_script, args.output)

    payload = {
        "script_file": saved_path,
        "assumptions": result.assumptions,
    }
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
