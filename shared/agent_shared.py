"""Shared utilities for Codex-backed agents."""

from __future__ import annotations

import json
import sys
from typing import List, Optional, Tuple

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful systems engineer. "
    "Respond with a single JSON object that matches the caller's schema. "
    "Do not include markdown fences or commentary."
)


def build_default_codex_options(
    allowed_tools: Optional[List[str]] = None,
    max_turns: int = 15,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> ClaudeAgentOptions:
    """Return a standard ClaudeAgentOptions configuration for Codex MCP usage."""
    tools = allowed_tools or ["mcp__codex", "Read", "Grep"]
    return ClaudeAgentOptions(
        allowed_tools=tools,
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
        system_prompt=system_prompt,
        max_turns=max_turns,
    )


async def invoke_agent(
    prompt: str,
    codex_options: ClaudeAgentOptions,
    context_label: str,
) -> Tuple[str, Optional[Exception]]:
    """Stream agent output into a single string while capturing any query errors."""
    response_text = ""
    query_error: Optional[Exception] = None

    try:
        async for message in query(prompt=prompt, options=codex_options):
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
        print(f"Warning: Agent streaming failed for {context_label}: {exc}", file=sys.stderr)

    return response_text, query_error


def log_parse_failure_details(
    context_label: str,
    query_error: Optional[Exception],
    agent_response: str,
) -> None:
    """Emit consistent diagnostics when response parsing fails."""
    if query_error:
        print(
            f"Warning: Agent query failed for {context_label}: {query_error}",
            file=sys.stderr,
        )
    snippet = agent_response.replace("\n", " ")[:400] if agent_response else ""
    if snippet:
        print(f"Agent response snippet: {snippet}...", file=sys.stderr)
