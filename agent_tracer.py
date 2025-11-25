#!/usr/bin/env python3
"""Agent-based tracer that emits structured JSON summaries for Gluon builtins."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import anyio

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    query,
)

from config import (
    ALL_GLUON_EXAMPLE_FILES,
    AMD_PRE_SEMANTIC_FILES,
    CPP_FILES,
    OUTPUT_DIR,
    SEMANTIC_FILES,
)
from agent_common import (
    MalformedAgentResponseError,
    extract_first_json_object,
    log_status,
    render_text_blocks,
)
from structures import BuiltinOp

DEFAULT_ALL_OPS_JSON = os.path.join(OUTPUT_DIR, "all_gluon_ops.json")


def default_gluon_interface(description: str = "") -> str:
    """Return a default Gluon interface string."""
    return description if description else "result = op_name(...)"


@dataclass
class TraceResult:
    """In-memory representation of the agent response."""

    gluon_op: str
    ttgir_ops: List[str]
    semantic: str
    lowering_summary: str
    gluon_op_interface: str
    code_snippet: str


class TracePromptBuilder:
    """Constructs a tracing prompt for a single builtin op."""

    def __init__(
        self,
        semantic_files: Sequence[str],
        cpp_files: Sequence[str],
        amd_pre_semantic_files: Sequence[str],
        example_files: Sequence[str],
    ) -> None:
        self.semantic_files = list(semantic_files)
        self.cpp_files = list(cpp_files)
        self.amd_pre_semantic_files = list(amd_pre_semantic_files)
        self.example_files = list(example_files)

    @staticmethod
    def _format_path_block(title: str, paths: Sequence[str]) -> str:
        if not paths:
            return ""
        entries = "\n".join(f"  - {path}" for path in paths)
        return f"{title}:\n{entries}\n"

    def build_prompt(self, op: BuiltinOp) -> str:
        analysis_goals = (
            "Deeply trace this single Gluon builtin across the lowering stack:\n"
            "Use Codex MCP tools to do any coding-related task (analysis, generate, search, read files, etc.).\n"
            "  1. Read the builtin definition carefully and capture the user-facing function interface as a code string.\n"
            "     Include only the parameters kernel authors actually write: required args, optional args with defaults,\n"
            "     keyword-only args, and any *args/**kwargs. Exclude compiler-injected or internal parameters such as\n"
            "     `_semantic`, `_generator`, or any other args beginning with `_` unless the public docs explicitly show\n"
            "     users providing them. If the op has both a function form and an operator sugar form, include both\n"
            "     (e.g., `result = op_name(...) OR result = x op y`); otherwise include the single available form.\n"
            "     Use this user-facing interface string for `gluon_op_interface`.\n"
            "  2. Follow the lowering into the semantic layers listed above.\n"
            "     AMD-specific ops must go through AMD_PRE_SEMANTIC_FILES first.\n"
            "  3. Inspect the builder logic inside the C++ files to understand how\n"
            "     TTGIR ops are emitted and how operands/attributes are computed.\n"
            "  4. List the TTGIR operations that define the semantic behavior, in execution order, as `dialect.operation`\n"
            "     entries (e.g. `ttg.make_range`). Only include the invariant core ops that always implement the semantic\n"
            "     (e.g. `arith.addi`, `arith.addf`, `tt.addptr` for add). Exclude optional helper/scaffolding ops such as\n"
            "     type conversions (`arith.extui`, `arith.trunci`), bounds/mask logic (`arith.cmpi`, `arith.andi`, `tt.splat`),\n"
            "     debug/assert ops (`tt.assert`), and constant/default builders (`arith.constant`).\n"
            "  5. Write the `lowering_summary` as a concise semantic map from the Gluon op to the TTGIR ops it emits.\n"
            "     Describe the core decision logic (e.g., dtype checks, broadcasting, specialization branches) that governs\n"
            "     which TTGIR constructs appear and why. Emphasize the essential semantics and relationships so someone\n"
            "     could later infer the reverse mapping from TTGIR back to Gluon. Avoid surface details such as file paths,\n"
            "     helper function names, or incidental implementation trivia—only mention names when they are indispensable\n"
            "     to the semantic explanation.\n"
            "  6. Search ONLY the ALL_GLUON_EXAMPLE_FILES listed below for the\n"
            "     most typical real kernel usage of this builtin.\n"
            "     Return a snippet ONLY if it calls the EXACT builtin symbol being traced.\n"
            "     For example, if tracing ttgl.amd.rdna3.wmma, the snippet must show ttgl.amd.rdna3.wmma,\n"
            "     not ttgl.amd.gfx1250.wmma or any other variant.\n"
            "     If no matching snippet exists in the allowed files, leave code_snippet empty.\n"
            "     Prefer examples from tutorial files over test files.\n"
            "     Choose a snippet that shows the op in a meaningful context (5-10 lines),\n"
            "     including relevant surrounding operations that make the usage clear.\n"
            "     DO NOT include source file path comments or line numbers.\n"
            "     Just provide clean, well-formatted code showing how to use the op.\n"
        )

        output_format = (
            "Respond with a single JSON object containing ONLY these keys:\n"
            "{\n"
            '  "gluon_op": "name",\n'
            '  "gluon_op_interface": "result = op_name(x, y, param=default) OR result = x op y",\n'
            '  "ttgir_ops": ["ttg.make_range"],\n'
            '  "semantic": "Concise semantic description",\n'
            '  "lowering_summary": "Detailed multi-sentence explanation of how the lowering works.",\n'
            '  "code_snippet": "Clean, well-formatted code snippet from ALL_GLUON_EXAMPLE_FILES (no source comments)"\n'
            "}\n"
            "No markdown, commentary, or additional fields are allowed.\n"
        )

        sections = [
            "You are an expert compiler engineer.",
            f"Builtin name: {op.name}",
            f"Source file: {op.file_path}",
            analysis_goals,
            "Files you must inspect:",
            self._format_path_block("AMD_PRE_SEMANTIC_FILES", sorted(self.amd_pre_semantic_files)),
            self._format_path_block("SEMANTIC_FILES", self.semantic_files),
            self._format_path_block("CPP_FILES", self.cpp_files),
            self._format_path_block("ALL_GLUON_EXAMPLE_FILES", self.example_files),
            output_format,
        ]
        return "\n".join(section for section in sections if section)


class TraceResponseParser:
    """Parses and normalizes the JSON emitted by the agent."""

    def __init__(self, json_loader=json.loads):
        self._json_loader = json_loader

    def parse(self, raw_response: str, fallback_name: str) -> TraceResult:
        if not raw_response or not raw_response.strip():
            raise MalformedAgentResponseError("Empty response from agent")

        payload = extract_first_json_object(raw_response)
        try:
            data = self._json_loader(payload)
        except json.JSONDecodeError as exc:
            raise MalformedAgentResponseError(f"Invalid JSON object: {exc}") from exc

        gluon_op = str(data.get("gluon_op") or fallback_name).strip()
        semantic = str(data.get("semantic") or f"Gluon builtin: {gluon_op}").strip()
        lowering_summary = str(data.get("lowering_summary") or "").strip()
        normalized_ops = self._normalize_ttgir_ops(data.get("ttgir_ops"))
        gluon_interface = self._normalize_gluon_interface(data.get("gluon_op_interface"))
        code_snippet = self._normalize_code_snippet(data.get("code_snippet"))

        return TraceResult(
            gluon_op=gluon_op,
            ttgir_ops=normalized_ops,
            semantic=semantic,
            lowering_summary=lowering_summary or "Lowering summary unavailable.",
            gluon_op_interface=gluon_interface,
            code_snippet=code_snippet,
        )

    @staticmethod
    def _normalize_ttgir_ops(value: Any) -> List[str]:
        if value is None:
            return []

        entries: Iterable[Any]
        if isinstance(value, str):
            entries = [value]
        elif isinstance(value, dict):
            entries = [value]
        elif isinstance(value, Iterable):
            entries = value
        else:
            raise MalformedAgentResponseError("ttgir_ops must be a list of strings")

        normalized: List[str] = []
        for entry in entries:
            if isinstance(entry, str):
                clean = entry.strip()
                if clean:
                    normalized.append(clean)
                continue

            if isinstance(entry, dict):
                dialect = str(entry.get("dialect", "")).strip()
                op_name = str(
                    entry.get("operation")
                    or entry.get("op")
                    or entry.get("name")
                    or entry.get("full_name", "")
                ).strip()
                if dialect and op_name:
                    normalized.append(f"{dialect}.{op_name}")
                elif op_name or dialect:
                    normalized.append(op_name or dialect)
                continue

        return normalized

    @staticmethod
    def _normalize_gluon_interface(value: Any) -> str:
        """Normalize gluon_op_interface to a simple code string."""
        if value is None:
            return default_gluon_interface()

        if isinstance(value, str):
            return value.strip() if value.strip() else default_gluon_interface()

        # If it's a dict (old format), try to extract a description
        if isinstance(value, dict):
            description = str(value.get("description") or "").strip()
            return description if description else default_gluon_interface()

        # For any other type, convert to string
        return str(value).strip() if str(value).strip() else default_gluon_interface()

    @staticmethod
    def _normalize_code_snippet(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            snippet = value.get("code") or value.get("snippet") or value.get("text")
            if snippet:
                return str(snippet).strip()
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            joined = "\n".join(str(item).strip() for item in value if item)
            return joined.strip()
        return str(value).strip()


class AgentBasedTracer:
    """Tracer that uses Claude Agent SDK + Codex MCP to summarize a builtin."""

    def __init__(
        self,
        prompt_builder: Optional[TracePromptBuilder] = None,
        response_parser: Optional[TraceResponseParser] = None,
    ) -> None:
        prompt_builder = prompt_builder or TracePromptBuilder(
            semantic_files=SEMANTIC_FILES,
            cpp_files=CPP_FILES,
            amd_pre_semantic_files=AMD_PRE_SEMANTIC_FILES,
            example_files=ALL_GLUON_EXAMPLE_FILES,
        )
        self.prompt_builder = prompt_builder
        self.response_parser = response_parser or TraceResponseParser()

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
                "You are an expert compiler engineer analyzing Triton Gluon operations.\n"
                "Your response MUST be a single valid JSON object that matches the caller's schema.\n"
                "Do not include prose, code fences, or commentary. Use empty arrays/strings when unsure."
            ),
            max_turns=30,
        )

    async def trace_operation(self, op: Union[BuiltinOp, Dict[str, Any]]) -> Dict[str, Any]:
        builtin = self._normalize_op(op)
        prompt = self.prompt_builder.build_prompt(builtin)
        raw_response, query_error = await self._invoke_agent(prompt, builtin.name)

        try:
            trace_result = self.response_parser.parse(raw_response, builtin.name)
            return self._result_to_json(trace_result, builtin)
        except MalformedAgentResponseError as exc:
            self._log_warning(builtin.name, exc, raw_response, query_error)
            return self._fallback_json(builtin, str(exc))
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._log_warning(builtin.name, exc, raw_response, query_error)
            return self._fallback_json(builtin, "Agent tracing failed")

    async def _invoke_agent(self, prompt: str, op_name: str) -> tuple[str, Optional[Exception]]:
        response_text = ""
        query_error: Optional[Exception] = None

        try:
            async for message in query(prompt=prompt, options=self.codex_options):
                if isinstance(message, AssistantMessage):
                    response_text += render_text_blocks(message.content)
                elif isinstance(message, ResultMessage):
                    response_text += render_text_blocks(getattr(message, "content", None))
                    payload = getattr(message, "result", None)
                    if payload:
                        if isinstance(payload, str):
                            response_text += payload
                        else:
                            response_text += json.dumps(payload)
        except Exception as exc:  # pragma: no cover - guard against agent issues
            query_error = exc
            log_status(f"Warning: Agent streaming failed for {op_name}: {exc}")

        return response_text, query_error

    @staticmethod
    def _normalize_op(op: Union[BuiltinOp, Dict[str, Any]]) -> BuiltinOp:
        if isinstance(op, BuiltinOp):
            return op

        if isinstance(op, dict):
            name = str(op.get("name", "unknown"))
            file_path = str(op.get("file_path", "unknown"))
            return BuiltinOp(name=name, file_path=file_path)

        raise TypeError(f"Unsupported op type for tracing: {type(op)}")

    @staticmethod
    def _result_to_json(result: TraceResult, op: BuiltinOp) -> Dict[str, Any]:
        clean_ops = [entry for entry in (s.strip() for s in result.ttgir_ops) if entry]
        return {
            "gluon_op": result.gluon_op or op.name,
            "ttgir_ops": clean_ops,
            "semantic": result.semantic or f"Gluon builtin: {op.name}",
            "lowering_summary": result.lowering_summary or "Lowering summary unavailable.",
            "gluon_op_interface": result.gluon_op_interface or default_gluon_interface(),
            "code_snippet": result.code_snippet or "",
        }

    @staticmethod
    def _fallback_json(op: BuiltinOp, reason: str) -> Dict[str, Any]:
        interface = default_gluon_interface(f"Interface unavailable: {reason}")
        return {
            "gluon_op": op.name,
            "ttgir_ops": [],
            "semantic": f"Gluon builtin: {op.name}",
            "lowering_summary": reason,
            "gluon_op_interface": interface,
            "code_snippet": "",
        }

    @staticmethod
    def _log_warning(
        op_name: str,
        error: Exception,
        response: Optional[str],
        query_error: Optional[Exception],
    ) -> None:
        if query_error:
            log_status(f"Warning: Agent query failed for {op_name}: {query_error}")
        log_status(f"Warning: Failed to parse agent response for {op_name}: {error}")
        if response:
            snippet = response.replace("\n", " ")[:400]
            if snippet:
                log_status(f"Response snippet: {snippet}...")


async def trace_builtin_with_agent(
    op: Union[BuiltinOp, Dict[str, Any]],
    tracer: Optional[AgentBasedTracer] = None,
) -> Dict[str, Any]:
    tracer = tracer or AgentBasedTracer()
    return await tracer.trace_operation(op)


def _load_op_from_json(
    op_name: str,
    json_path: str = DEFAULT_ALL_OPS_JSON,
) -> BuiltinOp:
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Gluon ops file not found: {json_path}. Run the scanner to generate it."
        )

    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    def _match(entry: Dict[str, Any]) -> Optional[BuiltinOp]:
        if entry.get("name") == op_name:
            return AgentBasedTracer._normalize_op(entry)
        return None

    if isinstance(payload, list):
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            candidate = _match(entry)
            if candidate:
                return candidate
    elif isinstance(payload, dict):
        candidate = _match(payload)
        if candidate:
            return candidate
        for value in payload.values():
            if isinstance(value, list):
                for entry in value:
                    if not isinstance(entry, dict):
                        continue
                    candidate = _match(entry)
                    if candidate:
                        return candidate

    raise ValueError(f"Operation '{op_name}' not found in {json_path}")


def _load_builtin_op(
    op_name: str,
    ops_file: str = DEFAULT_ALL_OPS_JSON,
    op_file_path: Optional[str] = None,
) -> BuiltinOp:
    if op_file_path:
        return BuiltinOp(name=op_name, file_path=os.path.abspath(op_file_path))
    return _load_op_from_json(op_name, ops_file)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace a Gluon builtin through lowering and emit JSON summary.",
    )
    parser.add_argument("--op", required=True, help="Name of the Gluon op to trace.")
    parser.add_argument(
        "--ops-file",
        default=DEFAULT_ALL_OPS_JSON,
        help=(
            "Path to the JSON file containing discovered Gluon operations. "
            f"Defaults to {DEFAULT_ALL_OPS_JSON}."
        ),
    )
    parser.add_argument(
        "--op-file-path",
        help=(
            "Optional absolute path to the builtin definition. "
            "When provided, the tracer skips the ops index and uses this path directly."
        ),
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()

    builtin = _load_builtin_op(
        op_name=args.op,
        ops_file=args.ops_file,
        op_file_path=args.op_file_path,
    )

    log_status(f"Tracing Gluon builtin: {builtin.name} ({builtin.file_path})")
    tracer = AgentBasedTracer()
    result = anyio.run(tracer.trace_operation, builtin)
    log_status("Agent trace complete.")

    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    sys.stdout.flush()
