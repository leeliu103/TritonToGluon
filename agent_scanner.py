#!/usr/bin/env python3
"""Agent-based scanner using Claude Agent SDK and Codex MCP.

Instead of manually parsing files, this uses an AI agent with Codex to:
1. Scan Triton Gluon source files
2. Extract builtin operations
3. Categorize them intelligently
"""

import argparse
import os
import sys
import json
import anyio
from typing import Dict, List, Optional, Set, Tuple

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    query,
)

from structures import BuiltinOp
from config import (
    BUILTIN_FILES,
    AMD_BUILTIN_DIRS,
    OUTPUT_DIR,
)
from agent_common import (
    MalformedAgentResponseError,
    log_status,
    parse_concatenated_json_arrays,
    render_text_blocks,
)

DEFAULT_ALL_OPS_OUTPUT = os.path.join(OUTPUT_DIR, "all_gluon_ops.json")
SCAN_EXCLUDED_FILES: Set[str] = set()


class AgentBasedScanner:
    """Scanner that uses Claude Agent SDK + Codex MCP to discover builtins."""

    def __init__(self):
        self.failure_count = 0
        self.failed_targets: List[str] = []
        self.codex_options = ClaudeAgentOptions(
            # Allow Codex MCP tools
            allowed_tools=["mcp__codex", "Read", "Grep"],

            # Register Codex as MCP server
            mcp_servers={
                "codex": {
                    "type": "stdio",
                    "command": "npx",
                    "args": [
                        "-y", "@openai/codex",
                        "-c", 'model_provider="amd-openai"',
                        "mcp-server"
                    ]
                }
            },

            # ClaudeAgentOptions has no dedicated JSON response_format toggle, so rely on the
            # system prompt to ensure strict JSON output from the agent.
            system_prompt=(
                "You are an expert code analyzer specializing in Gluon builtin discovery.\n"
                "Your response MUST be a single valid JSON array with no characters before or after it.\n"
                "Any non-JSON output will cause this tool to fail. No prose, code fences, or commentary are allowed.\n"
                "The first character must be '[' and the last must be ']'. Return [] when no builtins are found."
            ),

            max_turns=10,
        )

    def _add_ops_to_category(
        self,
        category: str,
        ops: List[BuiltinOp],
        categorized_ops: Dict[str, List[BuiltinOp]]
    ) -> None:
        """Record operations under a category."""
        if not ops:
            return

        bucket = categorized_ops.setdefault(category, [])
        # Track already-added ops so we don't double count when the agent
        # repeats the same JSON payload (common when the response is streamed).
        seen_identities = {(existing.name, existing.file_path) for existing in bucket}

        for op in ops:
            identity = (op.name, op.file_path)
            if identity in seen_identities:
                continue
            setattr(op, "category", category)
            bucket.append(op)
            seen_identities.add(identity)

    @staticmethod
    def _collect_unique_ops(categorized_ops: Dict[str, List[BuiltinOp]]):
        """Flatten categorized ops into a unique, serializable list."""
        seen = set()
        serialized_ops = []

        for category, ops in categorized_ops.items():
            for op in ops:
                identity = (op.name, op.file_path)
                if identity in seen:
                    continue
                seen.add(identity)
                serialized_ops.append({
                    "name": op.name,
                    "category": category,
                    "file_path": op.file_path,
                })

        serialized_ops.sort(key=lambda item: item["name"])
        return serialized_ops

    def _record_failure(self, target: str) -> None:
        """Track scan failures so the CLI can exit with a non-zero status."""
        self.failure_count += 1
        self.failed_targets.append(target)

    def write_discovered_ops(
        self,
        categorized_ops: Dict[str, List[BuiltinOp]],
        output_path: str
    ) -> str:
        """Persist discovered operations to JSON and return the absolute path."""
        if not output_path:
            return ""

        absolute_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(absolute_path), exist_ok=True)

        serialized_ops = self._collect_unique_ops(categorized_ops)
        with open(absolute_path, "w", encoding="utf-8") as json_file:
            json.dump(serialized_ops, json_file, indent=2)

        log_status(f"Wrote {len(serialized_ops)} Gluon ops to {absolute_path}")
        return absolute_path

    async def scan_file_with_agent(self, file_path: str) -> List[BuiltinOp]:
        """Use agent to scan a single file for builtins."""

        normalized_path = os.path.abspath(file_path)

        if normalized_path in SCAN_EXCLUDED_FILES:
            log_status(f"Skipping excluded file: {normalized_path}")
            return []

        if not os.path.exists(normalized_path):
            log_status(f"Warning: File not found: {normalized_path}")
            self._record_failure(normalized_path)
            return []

        file_path = normalized_path

        prompt = (
            f"Scan the Python file at {file_path} for builtin operations.\n"
            "Use Codex MCP tools to do any coding-related task (analysis, generate, search, read files, etc.).\n"
            "\n"
            "Definition patterns to look for:\n"
            "1. Functions decorated with @builtin.\n"
            "2. Assignments using builtin(...).\n"
            "3. Methods inside classes decorated with @builtin.\n"
            "\n"
            "Return a JSON array of objects with these keys:\n"
            "{\n"
            '  "name": "operation_name",\n'
            f'  "file_path": "{file_path}"\n'
            "}\n"
            "\n"
            "Important: reply with JSON only. Do not include any text before or after the array."
        )

        ops = []
        json_response = ""

        query_error: Optional[Exception] = None
        failure_occurred = False

        try:
            async for message in query(prompt=prompt, options=self.codex_options):
                if isinstance(message, AssistantMessage):
                    json_response += render_text_blocks(message.content)
                elif isinstance(message, ResultMessage):
                    json_response += render_text_blocks(getattr(message, "content", None))
                    result_payload = message.result
                    if result_payload:
                        if isinstance(result_payload, str):
                            json_response += result_payload
                        else:
                            json_response += json.dumps(result_payload)
        except Exception as exc:  # pragma: no cover - protect against agent failures
            query_error = exc
            failure_occurred = True
            log_status(f"Warning: Agent scan failed for {file_path}: {exc}")

        # Parse JSON response
        try:
            ops_data = parse_concatenated_json_arrays(json_response)

            for index, op_data in enumerate(ops_data):
                try:
                    op = BuiltinOp(
                        name=op_data["name"],
                        file_path=op_data["file_path"],
                    )
                    ops.append(op)
                except KeyError as missing_key:
                    log_status(
                        f"Warning: Skipping malformed builtin entry #{index} for {file_path}: missing {missing_key}"
                    )
                except Exception as build_error:
                    log_status(
                        f"Warning: Skipping malformed builtin entry #{index} for {file_path}: {build_error}"
                    )

        except MalformedAgentResponseError as e:
            if query_error:
                log_status(f"Warning: Agent query error for {file_path}: {query_error}")
            log_status(f"Warning: Malformed agent response for {file_path}: {e}")
            snippet = json_response[:200].replace("\n", " ")
            if snippet:
                log_status(f"Response snippet: {snippet}...")
            failure_occurred = True
        except Exception as e:
            if query_error:
                log_status(f"Warning: Agent query error for {file_path}: {query_error}")
            log_status(f"Warning: Failed to parse agent response for {file_path}: {e}")
            snippet = json_response[:200].replace("\n", " ")
            if snippet:
                log_status(f"Response snippet: {snippet}...")
            failure_occurred = True

        if failure_occurred:
            self._record_failure(file_path)
        return ops

    async def scan_directory_with_agent(self, dir_path: str) -> List[BuiltinOp]:
        """Use agent to scan a directory for builtins."""

        normalized_dir = os.path.abspath(dir_path)

        if not os.path.exists(normalized_dir):
            log_status(f"Warning: Directory not found: {normalized_dir}")
            self._record_failure(normalized_dir)
            return []

        dir_path = normalized_dir

        # Collect all Python files
        python_files = []
        for root, dirs, files in os.walk(dir_path):
            dirs[:] = [d for d in dirs if d != '__pycache__']
            for file in files:
                if file.endswith('.py') and not file.startswith('_test'):
                    full_path = os.path.join(root, file)
                    normalized_path = os.path.abspath(full_path)
                    if normalized_path in SCAN_EXCLUDED_FILES:
                        continue
                    python_files.append(normalized_path)

        # Scan each file with agent
        all_ops = []
        for file_path in python_files:
            ops = await self.scan_file_with_agent(file_path)
            all_ops.extend(ops)

        return all_ops

    async def scan_all(self) -> Dict[str, List[BuiltinOp]]:
        """Scan all configured files and directories for builtins using agents."""

        self.failure_count = 0
        self.failed_targets = []

        categorized_ops: Dict[str, List[BuiltinOp]] = {}

        log_status("Scanning with AI agents...")

        for file_path in BUILTIN_FILES:
            log_status(f"Scanning builtin file {os.path.basename(file_path)}...")
            ops = await self.scan_file_with_agent(file_path)
            filename = os.path.basename(file_path)
            category_name, _ = os.path.splitext(filename)
            category_name = category_name.lstrip('_') or 'builtin'
            self._add_ops_to_category(category_name, ops, categorized_ops)

        for dir_path in AMD_BUILTIN_DIRS:
            log_status(f"Scanning AMD directory {os.path.basename(dir_path)}...")
            ops = await self.scan_directory_with_agent(dir_path)
            self._add_ops_to_category('amd', ops, categorized_ops)

        return categorized_ops


async def scan_with_agents(
    output_path: Optional[str] = DEFAULT_ALL_OPS_OUTPUT
) -> Tuple[Dict[str, List[BuiltinOp]], int]:
    """Main entry point for agent-based scanning."""
    scanner = AgentBasedScanner()
    categorized_ops = await scanner.scan_all()
    if output_path:
        scanner.write_discovered_ops(categorized_ops, output_path)
    return categorized_ops, scanner.failure_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan Gluon sources with AI agents.")
    parser.add_argument(
        "--output",
        default=DEFAULT_ALL_OPS_OUTPUT,
        help=(
            "Path to write the discovered Gluon operations JSON file. "
            f"Defaults to {DEFAULT_ALL_OPS_OUTPUT}."
        ),
    )
    cli_args = parser.parse_args()

    categorized, failure_count = anyio.run(scan_with_agents, cli_args.output)

    serialized_ops = AgentBasedScanner._collect_unique_ops(categorized)

    if failure_count:
        log_status(f"Agent scan completed with {failure_count} failure(s).")
    else:
        log_status("Agent scan complete.")

    log_status(
        f"Discovered {len(serialized_ops)} builtin operations across {len(categorized)} categories."
    )

    if categorized:
        log_status("Summary statistics:")
        for category, ops in categorized.items():
            log_status(f"  - {category}: {len(ops)} ops")

    log_status(f"JSON results written to: {os.path.abspath(cli_args.output)}")

    json.dump(serialized_ops, sys.stdout, indent=2)
    sys.stdout.write("\n")
    sys.stdout.flush()

    if failure_count:
        sys.exit(1)
