# ARCHITECTURE_DESIGN.md

> Architecture summary for the streamlined `src/` tree after removing the IR stage (commit `d517314ffdeea04e4e5111ce6e2b32c41f5bd873` + simplification plan).

## 1. Overview

### Purpose of the project

TritonToGluon translates Triton JIT kernels into Gluon kernels so existing Triton workloads can be executed on NVIDIA's experimental Gluon backend. The project acts as a research harness for understanding how to combine deterministic tooling and future LLM agents to bridge semantic gaps (layouts, tensor metadata, and warp scheduling details) between both ecosystems.

### Key design goals

- **Three-layer pipeline** – A lean frontend → TTGIR stub → mapping/codegen stack that mirrors the upstream translator’s behavior without introducing intermediate IR packages.
- **Deterministic skeletons** – Provide concrete, unit-testable Python entry points for every stage so future LLM or heuristic components have well-defined seams to plug into.
- **Mapping via Python functions** – Replace YAML/spec-based rules with explicit Python helpers registered in a `MappingFunctionRegistry`, matching the procedural style of `translator_helpers.py`.
- **Room for growth** – TTGIR extraction is currently a stub that returns empty metadata but exposes the API surface required for MLIR log ingestion or LLM-assisted layout reconstruction later.

### High-level architecture diagram

```
+------------------+      +-------------------+      +----------------------+
|  Frontend (AST)  | ---> |   TTGIR stub      | ---> |  Mapping + CodeGen   |
|  parser + scopes |      |  (extractor)      |      |  (Python functions)  |
+------------------+      +-------------------+      +----------------------+
        |                         |                             |
        v                         v                             v
  ParsedKernel             TTGIROutput (empty)          GluonModule (AST text)
```

## 2. Pipeline Flow

1. **Frontend (unchanged)** – `src/frontend/parser.py` parses Triton kernels into `ParsedKernel` objects that include the AST, source text, filename, globals, constexpr metadata, and a lightweight call graph. This stage preserves enough context for constexpr/global reconstruction later.
2. **TTGIR extraction (stub)** – `src/ttgir/extractor.py` exposes `TTGIRExtractor.extract`. Today it simply returns `TTGIROutput(spec_matches={}, layouts={})`, but the method signature already accepts optional MLIR dumps and `ParsedKernel`s so real extraction logic can be plugged in later.
3. **Mapping registry** – `src/mapping/function_registry.py` provides `MappingFunctionRegistry`, a decorator-based registry for Python mapping helpers. Files under `src/mapping/functions/` (e.g., `arange.py`, `program_id.py`, `dot.py`) register **actual Gluon helpers** via `@registry.register("<op>")`. The registry stores callable references so the code generator can import those helpers directly.
4. **Code generation** – `src/codegen/generator.py` implements `CodeGenerator`. It deep-copies the Triton AST, uses `_CallMappingTransformer` to rewrite Triton builtins by consulting the registry, injects `@gluon.jit` decorators, and `ast.unparse`s function definitions into `GluonKernel.source`. `CodeEmitter` (`src/codegen/emitter.py`) prepends the canonical Gluon import block, adds any helper imports requested by the transformer, and writes the kernel definitions into `.gluon.py` files.

## 3. Component Details

### Frontend (`src/frontend`)

- **Purpose** – Produce `ParsedKernel` dataclasses from Triton callables, capture globals/constexpr metadata, and expose the AST needed for mapping.
- **Inputs/Outputs** – Input is a Triton `JITFunction` or plain Python kernel. Output is a `ParsedKernel` with `tree`, `source`, `filename`, `globals_map`, `call_graph`, and `constexpr_metadata`.
- **Notes** – No structural changes were required for the simplification, so the frontend continues to own parsing and scope tracking responsibilities.

### TTGIR (`src/ttgir`)

- **Purpose** – Provide data structures (`LayoutMetadata`, `BlockedLayout`, `SliceLayout`) and a future-proof extraction API (`TTGIRExtractor`, `TTGIROutput`).
- **Key classes** – `TTGIROutput` groups `spec_matches`, `layouts`, and placeholder diagnostics. `TTGIRExtractor.extract` accepts a `ParsedKernel` plus an optional MLIR dump and returns empty metadata for now.
- **Notes** – Because TTGIR is a stub, consumers must tolerate missing layout information. The simplified pipeline still threads `TTGIROutput` through mapping helpers so the interface remains stable when extraction becomes real.

### Mapping (`src/mapping`)

- **Purpose** – Capture Triton → Gluon lowering logic using explicit Python functions.
- **Key classes** – `MappingFunctionRegistry` is a thin decorator-based registry that only stores callable references plus enough information to derive import statements.
- **Inputs/Outputs** – The registry maps a Triton op name (e.g., `"arange"`) to a runnable Gluon helper such as `tl_arange`. Helpers accept TTGIR metadata via explicit keyword parameters (`layout`, `layout_a`, etc.), so they can be imported and called directly in emitted kernels.
- **Notes** – Default helpers live under `src/mapping/functions/`, are decorated with `@gluon.jit`, and reuse upstream translator helpers (`translator_helpers.tl_dot`, etc.) until native Gluon implementations exist. Adding a new builtin simply means registering another helper function.

### Codegen (`src/codegen`)

- **Purpose** – Rewrite the Triton AST by invoking mapping helpers and emit Gluon source text.
- **Key classes** – `CodeGenerator`, `_CallMappingTransformer`, `GluonKernel`, `GluonModule`, and `CodeEmitter`.
- **Inputs/Outputs** – `CodeGenerator.generate(parsed_kernel, ttgir_output)` returns a `GluonModule` whose `imports` list only contains helper import statements plus a `kernels` list of `GluonKernel` objects. `CodeEmitter.emit` writes a `.gluon.py` stub by prepending standard Gluon imports, adding helper imports, and appending kernel definitions.
- **Notes** – The transformer mirrors the upstream translator: it inspects `ast.Call` nodes, recognizes `tl.*` builtins, rewrites those calls to the registered helper function names (while threading any TTGIR metadata as keyword args), and injects `@gluon.jit` decorators. Because lowering happens directly on the AST, no IR package is needed.

## 4. Data Structures

| Structure | Location | Key fields | Role |
|-----------|----------|------------|------|
| `ParsedKernel` | `src/frontend/parser.py` | `kernel`, `tree`, `source`, `globals_map`, `call_graph`, `constexpr_metadata` | Canonical representation of the Triton kernel fed into the backend pipeline. |
| `TTGIROutput` | `src/ttgir/extractor.py` | `spec_matches`, `layouts`, `diagnostics` | Placeholder container for extracted metadata; currently empty but plumbed through the pipeline. |
| `GluonKernel` | `src/codegen/generator.py` | `name`, `source` | One Gluon function produced by `ast.unparse`. |
| `GluonModule` | `src/codegen/generator.py` | `imports`, `kernels` | Aggregates helper import statements plus the generated kernel sources for emission. |

## 5. Design Decisions

- **Remove IR layer entirely** – The previous IR package provided annotated nodes that were primarily placeholders. Dropping it shortens the feedback loop and aligns the project with the proven upstream translator flow while leaving room to reintroduce a richer IR only if future requirements demand it.
- **Python function-based mappings** – Instead of YAML specs, mapping logic now lives in ordinary Python modules with decorator registration. This choice unlocks full Python expressiveness (control flow, helper imports, docstrings) and mirrors developer expectations set by `translator_helpers.py`. It also makes experimentation easy—developers can set breakpoints directly inside mapping functions.
- **TTGIR stub with future-proof API** – Even though extraction returns empty structures today, the API already accepts parsed kernels and MLIR dumps. When layout extraction or LLM integration arrives, only the implementation of `TTGIRExtractor.extract` needs to change; downstream consumers already receive a `TTGIROutput`.
- **AST-first code generation** – Walking and rewriting the AST keeps the implementation close to the reference translator and avoids text-based string munging. Using `ast.unparse` ensures emitted Gluon functions stay syntactically correct while still allowing the emitter to prepend canonical imports and helper snippets.

## 6. Comparison with Reference Implementation

The upstream translator (`/app/triton/python/triton/tools/triton_to_gluon_translater/translator.py` and `translator_helpers.py`) performs an all-in-one AST rewrite: it visits every call, rewrites Triton builtins to Gluon equivalents, injects decorators, reconstructs constexpr globals, and emits source strings in a single pass.

Our simplified architecture mirrors that behavior but introduces explicit seams:

- The AST transformer (`CodeGenerator`) is smaller and easier to reason about because builtin rewrites are delegated to dedicated mapping helpers rather than hard-coded dictionaries.
- Mapping helpers can reuse the existing Gluon helper ecosystem (e.g., `tl_dot`, `tl_dot_decomposed_scale_to_16`) just like the reference translator while still accepting TTGIR metadata explicitly through their parameters, enabling richer conditional logic without runtime context objects.
- TTGIR extraction remains a stub but is now a first-class stage, which means future layout metadata (from MLIR dumps or LLM agents) can start influencing mapping helpers without touching the frontend or code generator.

Overall, the new pipeline keeps the deterministic, AST-centric feel of the reference translator while shedding the unused IR layer and adopting explicit Python mapping functions, setting the stage for incremental feature work without over-engineering.
