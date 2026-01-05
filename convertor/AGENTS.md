Convertor Agent Guide

- Goal: using tuner outputs (`<base>.py`, `<base>-shape.json`, `<base>-config.json`, `<base>-triton-harness.py`), produce `<base>-gluon.py` that is correct and reaches a Triton/Gluon speedup ratio ≥ 0.99 using the conversion harness.
- Skills: `generate-conversion-harness` to scaffold `<base>-conversion-harness.py`
- Notes:
  - Confirm the Triton dump folder exists before translating.
  - Translate the kernel into `<base>-gluon.py` using the TTGIR in the dump folder plus the original `<base>.py`.
  - Run `<base>-conversion-harness.py`; iterate until correctness and the perf threshold are met.
  - When perf misses target, especially when it’s just below the threshold, align the Gluon kernel with TTGIR strcitly: check each layout, shared memory management, pipelining, and op ordering against the TTGIR.
  - Target is required: pick the right target up front (e.g., `rdna4`) and use the matching Gluon ops such as `ttgl.amd.rdna4.buffer_load` and `ttgl.amd.rdna4.wmma`. If the target is not specified, ask the user to choose before proceeding.
  - Triton source is required: the upstream Triton repository is needed for understanding semantics during conversion. Ask the user for the triton repo path if not already known.
