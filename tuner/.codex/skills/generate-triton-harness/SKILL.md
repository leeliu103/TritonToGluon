---
name: generate-triton-harness
description: Generate shape/config JSON files and a Triton harness from a folder containing a single Triton kernel.
metadata:
  short-description: Build shape+config JSON and harness from one Triton kernel
---

# Generate Triton Harness
Given a folder with exactly one Triton kernel `<base>.py`, create `<base>-shape.json`, `<base>-config.json`, and `<base>-triton-harness.py` using the reference pattern in `references/`.

## Inputs
- folder (required): path that must contain a single Triton kernel `<base>.py` file (no other files except optional `__pycache__/`). If extra files are present, stop and report them.

## Outputs
- `<base>-shape.json`: tensor shapes/grid/seed (see `references/matrix-multiplication-shape.json`)
- `<base>-config.json`: meta/launch/run (see `references/matrix-multiplication-config.json`)
- `<base>-triton-harness.py`: runnable harness (see `references/matrix-multiplication-triton-harness.py`):
  - Loads shape/config JSON, seeds RNG, builds tensors per shape spec.
  - Uses grid from shape (`eval` on the lambda string).
  - Passes `meta` to the kernel and `num_warps/num_stages` from `launch`.
  - `--benchmark` flag to enable warmup/rep; default is a single iteration.

## Quick start
- Use the `references/` files as a template for field names and structure.
- After generation, for verification run:
  - `python <base>-triton-harness.py` (single run)
  - `python <base>-triton-harness.py --benchmark` (warmup/rep from config)

## Steps
1) Validate folder contents (one `<base>.py`, nothing else besides `__pycache__/`); if invalid, list offending files and exit.
2) Derive `<base>` from the kernel filename.
3) Create `<base>-shape.json` and `<base>-config.json` following the reference schemas.
4) Write `<base>-triton-harness.py` matching the reference harness behavior.
5) Refuse to overwrite existing generated files, warn if they already exist.
