#!/usr/bin/env python3
"""Agent that scans Gluon builtins and traces each one into semantic summaries."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import anyio

from agent_common import log_status
from agent_scanner import AgentBasedScanner
from agent_tracer import AgentBasedTracer
from config import OUTPUT_DIR

DEFAULT_SEMANTICS_OUTPUT = os.path.join(OUTPUT_DIR, "all_semantics.json")


@dataclass
class TraceStatistics:
    total: int = 0
    success: int = 0
    failures: int = 0

    def record_success(self) -> None:
        self.success += 1

    def record_failure(self) -> None:
        self.failures += 1


async def _discover_builtins(scanner: AgentBasedScanner) -> Sequence[Any]:
    """Run the scanner and flatten categorized ops."""
    log_status("Starting builtin discovery with AgentBasedScanner...")
    categorized = await scanner.scan_all()

    flattened: List[Any] = []
    if categorized:
        log_status(f"Discovered {len(categorized)} category bucket(s):")
        for category, ops in categorized.items():
            count = len(ops)
            log_status(f"  - {category}: {count} builtin(s)")
            flattened.extend(ops)
    else:
        log_status("Scanner returned no categories.")

    log_status(f"Total builtins discovered: {len(flattened)}")

    if scanner.failure_count:
        log_status(
            f"Warning: {scanner.failure_count} scan target(s) failed. "
            "Continuing to tracing phase."
        )

    return flattened


def _fallback_trace_payload(op: Any, reason: str) -> Dict[str, Any]:
    """Return a placeholder trace entry when tracing fails."""
    op_name = str(getattr(op, "name", "unknown"))
    return {
        "gluon_op": op_name,
        "ttgir_ops": [],
        "semantic": f"Gluon builtin: {op_name}",
        "lowering_summary": reason,
    }


async def _trace_all_builtins(
    builtins: Sequence[Any],
    max_concurrency: int,
) -> tuple[List[Dict[str, Any]], TraceStatistics]:
    """Spawn tracing subagents for every builtin and collect their JSON outputs."""
    stats = TraceStatistics(total=len(builtins))
    if not builtins:
        return [], stats

    semaphore = anyio.Semaphore(max(1, max_concurrency))
    stats_lock = anyio.Lock()
    results: List[Optional[Dict[str, Any]]] = [None] * stats.total

    async def trace_single(index: int, builtin: Any) -> None:
        tracer = AgentBasedTracer()
        op_name = str(getattr(builtin, "name", f"builtin_{index}"))

        async with semaphore:
            log_status(f"[{index + 1}/{stats.total}] Tracing {op_name}...")
            try:
                trace_payload = await tracer.trace_operation(builtin)
                async with stats_lock:
                    stats.record_success()
            except Exception as exc:  # pragma: no cover - defensive safeguard
                async with stats_lock:
                    stats.record_failure()
                log_status(f"Warning: Trace subagent failed for {op_name}: {exc}")
                trace_payload = _fallback_trace_payload(
                    builtin, f"Agent trace failed: {exc}"
                )

        results[index] = trace_payload

    log_status(
        f"Tracing {stats.total} builtins with up to {max(1, max_concurrency)}"
        " concurrent subagents..."
    )

    async with anyio.create_task_group() as tg:
        for idx, builtin in enumerate(builtins):
            tg.start_soon(trace_single, idx, builtin)

    serialized_results = [payload for payload in results if payload is not None]
    return serialized_results, stats


def _persist_results(results: Sequence[Dict[str, Any]], output_path: str) -> str:
    """Write aggregated semantics to disk."""
    absolute_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)

    with open(absolute_path, "w", encoding="utf-8") as handle:
        json.dump(list(results), handle, indent=2)
        handle.write("\n")

    log_status(f"Wrote {len(results)} trace result(s) to {absolute_path}")
    return absolute_path


async def build_semantics_database(
    output_path: str,
    max_concurrency: int,
) -> tuple[List[Dict[str, Any]], TraceStatistics, int]:
    """High-level orchestration for scanning + tracing."""
    scanner = AgentBasedScanner()
    builtins = await _discover_builtins(scanner)
    traces, stats = await _trace_all_builtins(builtins, max_concurrency)

    _persist_results(traces, output_path)
    return traces, stats, scanner.failure_count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan and trace all Gluon builtins into semantic summaries."
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_SEMANTICS_OUTPUT,
        help=(
            "Path to write the aggregated trace JSON. "
            f"Defaults to {DEFAULT_SEMANTICS_OUTPUT}."
        ),
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Maximum number of tracing subagents to run concurrently.",
    )
    return parser.parse_args()


async def _main_async(output_path: str, max_concurrency: int) -> tuple[List[Dict[str, Any]], TraceStatistics, int]:
    """Async wrapper so anyio.run can orchestrate execution."""
    log_status("Agent semantics builder starting...")
    traces, stats, scan_failures = await build_semantics_database(
        output_path=output_path,
        max_concurrency=max_concurrency,
    )
    log_status(
        "Tracing complete: "
        f"total={stats.total}, success={stats.success}, failures={stats.failures}"
    )
    if scan_failures:
        log_status(f"Scan reported {scan_failures} failure(s).")
    return traces, stats, scan_failures


if __name__ == "__main__":
    cli_args = _parse_args()
    safe_concurrency = max(1, cli_args.max_concurrency)

    try:
        trace_results, statistics, scan_failures = anyio.run(
            _main_async,
            cli_args.output,
            safe_concurrency,
        )
    except KeyboardInterrupt:
        log_status("Agent semantics builder interrupted by user.")
        sys.exit(1)

    json.dump(trace_results, sys.stdout, indent=2)
    sys.stdout.write("\n")
    sys.stdout.flush()

    exit_code = 0
    if scan_failures or statistics.failures:
        exit_code = 1
        log_status(
            f"Completed with issues. scan_failures={scan_failures}, "
            f"trace_failures={statistics.failures}"
        )
    else:
        log_status("Agent semantics builder finished successfully.")

    sys.exit(exit_code)
