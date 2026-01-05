---
name: generate-conversion-harness
description: Generate a Triton-vs-Gluon conversion harness using existing shape/config JSON and the Triton harness.
metadata:
  short-description: Build conversion harness from shape/config + Triton harness
---

# Generate Conversion Harness
Given a folder that already has `<base>.py`, `<base>-shape.json`, `<base>-config.json`, and `<base>-triton-harness.py`, create `<base>-conversion-harness.py` that runs Triton vs Gluon with correctness + perf checks.

## Inputs
- folder: contains a single Triton kernel `<base>.py` plus the matching shape/config JSON and Triton harness from the tuner step.

## Outputs
- `<base>-conversion-harness.py`: launches Triton and Gluon kernels using the shape/config metadata, checks correctness, and benchmarks both.

## Workflow
1) Confirm the folder has `<base>.py`, `<base>-shape.json`, `<base>-config.json`, `<base>-triton-harness.py`; stop if files are missing or multiple bases exist.
2) Generate `<base>-conversion-harness.py` following the pattern in `references/matrix-multiplication-conversion-harness.py`.
