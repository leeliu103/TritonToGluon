Triton Config Tuner Agent Guide

- Goal: given a folder with one Triton kernel `<base>.py`, produce `<base>-shape.json`, `<base>-config.json`, `<base>-triton-harness.py`, then tune the config for best runtime and keep dumps in sync.
- Skills: `generate-triton-harness` (initial shape/config/harness), `dump-triton` (run harness and capture dumps).
- Notes:
  - After generating the initial shape/config/harness, ask the user if they want to edit `<base>-shape.json`, honor their changes.
  - Run `<base>-triton-harness.py --benchmark` for timing.
  - Use `dump-triton` to produce dumps.
  - Use dumps (especially the amdgcn file, which shows register spills/lds usage/occupancy) to guide `<base>-config.json` changes.
  - Change any meta or launch parameters in `<base>-config.json`.
  - Avoid repeating the same config.
  - Iterate up to 10 configs, track the best runtime.
  - Use `dump-triton` to get up-to-date dumps for the current config in each iteration after getting timing.
  - Persist best config in `<base>-config.json`, and finish with a `dump-triton` run so dumps match the best config.
