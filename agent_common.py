"""Shared helpers for agent-based scanner and tracer workflows."""

from __future__ import annotations

import json
import re
import sys
from typing import Any, List, Optional, Sequence

from claude_agent_sdk import TextBlock

__all__ = [
    "MalformedAgentResponseError",
    "extract_first_json_object",
    "log_status",
    "parse_concatenated_json_arrays",
    "render_text_blocks",
    "strip_markdown_code_fences",
]


class MalformedAgentResponseError(RuntimeError):
    """Raised when an agent response cannot be parsed as JSON."""


def log_status(message: str) -> None:
    """Emit operational logs to stderr so stdout can remain JSON-only."""
    print(message, file=sys.stderr)


_CODE_FENCE_PATTERN = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\s*(.*?)```", re.DOTALL)


def strip_markdown_code_fences(raw_text: str) -> str:
    """Remove markdown style code fences while preserving the inner payload."""
    if "```" not in raw_text:
        return raw_text

    def _replace(match: re.Match) -> str:
        inner = match.group(1)
        return inner.strip() if inner else ""

    stripped = _CODE_FENCE_PATTERN.sub(_replace, raw_text)
    return stripped.replace("```", "").strip()


def render_text_blocks(blocks: Optional[Sequence[TextBlock]]) -> str:
    """Collapse TextBlock content into a single string."""
    if not blocks:
        return ""

    parts: List[str] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            parts.append(block.text)
    return "".join(parts)


def parse_concatenated_json_arrays(raw_text: str) -> List[Any]:
    """Parse agent responses that may contain fences or concatenated arrays."""
    if not raw_text or not raw_text.strip():
        raise MalformedAgentResponseError("Empty response from agent")

    sanitized = strip_markdown_code_fences(raw_text).strip()
    if not sanitized:
        raise MalformedAgentResponseError(
            "Response only contained markdown fences without JSON content"
        )

    decoder = json.JSONDecoder()
    idx = 0
    merged: List[Any] = []
    parsed_arrays = 0

    while idx < len(sanitized):
        while idx < len(sanitized) and sanitized[idx].isspace():
            idx += 1
        if idx >= len(sanitized):
            break
        if sanitized[idx] != "[":
            raise MalformedAgentResponseError(
                "Agent response contained unexpected non-JSON content before '['"
            )
        try:
            payload, end_idx = decoder.raw_decode(sanitized, idx)
        except json.JSONDecodeError as exc:
            raise MalformedAgentResponseError(f"Invalid JSON array: {exc}") from exc

        if not isinstance(payload, list):
            raise MalformedAgentResponseError(
                f"Agent response must be a JSON array, received {type(payload).__name__}"
            )

        merged.extend(payload)
        parsed_arrays += 1
        idx = end_idx

    if parsed_arrays == 0:
        raise MalformedAgentResponseError("Agent response did not contain a JSON array")

    return merged


def extract_first_json_object(raw_text: str) -> str:
    """Locate the first JSON object in an agent response after fence stripping."""
    sanitized = strip_markdown_code_fences(raw_text).strip()
    if not sanitized:
        raise MalformedAgentResponseError("Agent response only contained markdown fences")

    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(sanitized):
        char = sanitized[idx]
        if char.isspace():
            idx += 1
            continue
        if char != "{":
            idx += 1
            continue
        try:
            _, end_idx = decoder.raw_decode(sanitized, idx)
            return sanitized[idx:end_idx]
        except json.JSONDecodeError:
            idx += 1

    raise MalformedAgentResponseError("No JSON object found in agent response")
