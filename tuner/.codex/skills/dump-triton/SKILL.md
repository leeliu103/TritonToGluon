---
name: dump-triton
description: Run a Triton harness with dumping enabled and keep every generated file under a dedicated dump directory.
metadata:
  short-description: Execute harness + preserve full Triton dumps
---

# Dump Triton
Run the generated `<base>-harness.py` with `<base>-config.json` and `<base>-shape.json` to produce Triton dump artifacts.

## Inputs
- folder (required): folder containing exactly one `<base>-harness.py`, `<base>-shape.json` and `<base>-config.json`.

If the base names differ or unexpected files are present, stop and tell the user exactly what to fix.

## Outputs
- A dump directory with all Triton artifacts (ttgir, hsaco, etc.) untouched.
- Harness stdout timing JSON (from the harness).

## Quick start
```bash
bash .codex/skills/dump-triton/scripts/dump-triton.sh --folder "<path>"
```

## Behavior
- Before each run, deletes any existing `<base>-dump` directory, then recreates it to keep the dump set clean.
- Sets `TRITON_DUMP_DIR`, `TRITON_ALWAYS_COMPILE=1`, and `TRITON_KERNEL_DUMP=1` for the run.
- Runs the harness and exits non-zero if it fails.
- Writes all artifacts under `<base>-dump` and leaves them intact after the run.
