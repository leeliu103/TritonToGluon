#!/usr/bin/env python3
"""CLI utility to extract MLIR content from a Triton compiler log."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MARKER = "// -----// IR Dump Before SCFToControlFlowPass"
MARKER_PREFIX = "// -----// IR Dump Before"
LOC_PATTERN = re.compile(r"\s*#loc\d+\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the MLIR dump that appears after the first occurrence of "
            f'"{MARKER}" in a Triton compiler log.'
        )
    )
    parser.add_argument(
        "log_path",
        type=Path,
        help="Path to the Triton compiler MLIR log file",
    )
    return parser.parse_args()


def extract_mlir_content(log_path: Path) -> list[str]:
    """Return MLIR lines that follow the marker, skipping #loc definitions."""
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    try:
        with log_path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError as exc:
        raise RuntimeError(f"Unable to read {log_path}: {exc}") from exc

    marker_index = next(
        (idx for idx, line in enumerate(lines) if MARKER in line),
        None,
    )
    if marker_index is None:
        raise ValueError(f'Marker "{MARKER}" not found in {log_path}')

    next_marker_index = next(
        (
            idx
            for idx in range(marker_index + 1, len(lines))
            if lines[idx].startswith(MARKER_PREFIX)
        ),
        len(lines),
    )

    mlir_lines = [
        line
        for line in lines[marker_index + 1 : next_marker_index]
        if not LOC_PATTERN.match(line)
    ]
    return mlir_lines


def main() -> int:
    args = parse_args()
    try:
        mlir_lines = extract_mlir_content(args.log_path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if mlir_lines:
        sys.stdout.writelines(mlir_lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
