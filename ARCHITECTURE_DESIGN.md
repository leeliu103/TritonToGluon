# ARCHITECTURE_DESIGN.md

> Architecture summary for the `src/` tree in commit `d517314ffdeea04e4e5111ce6e2b32c41f5bd873`.

## 1. Overview

### Purpose of the project

TritonToGluon is a research harness that translates Triton JIT kernels into Gluon kernels so that existing Triton workloads can be exercised on the experimental Gluon backend. The long-term objective is to understand how much compiler automation is required to cross the Triton/Gluon semantic gap, and to prepare infrastructure that can be assisted by an LLM agent when heuristics are insufficient.

### Key design goals

- Preserve Triton-level semantics (constexpr binding, program structure, tensor metadata) so Gluon emits functionally equivalent programs.
- Keep stages modular (frontend → TTGIR → IR → mapping → codegen) to enable rapid iteration and independent testing.
- Make lowering decisions data-driven via declarative specs rather than hard-coded rewrites.
- Create space for future MLIR/TTGIR ingestion (possibly guided by an LLM agent) without entangling it with parsing or codegen.

### High-level architecture diagram

```
+------------------+     +-----------------+     +----------------+     +------------------+     +-----------------+
|  Frontend (AST)  | --> | TTGIR metadata  | --> | Annotated  IR  | --> | Mapping registry | --> |   Codegen/emit   |
| parser + scopes  |     |  (layouts, etc) |     | (AST + layout) |     | (templated ops)  |     | (Gluon module)   |
+------------------+     +-----------------+     +----------------+     +------------------+     +-----------------+
         |                        |                      |                       |                        |
         v                        v                      v                       v                        v
   SymbolTable            LayoutMetadata          AnnotatedNode          MappingSpec                GluonModule
```

## 2. Pipeline Flow

1. **Frontend (parse AST + build symbol table)** – `src/frontend/parser.py` and `src/frontend/scope.py` extract `ParsedKernel` objects containing the AST, source text, filename, and globals. `ScopeAnalyzer` builds a lightweight `SymbolTable` so constexpr/global bindings can be recovered later.
2. **TTGIR extraction** – `src/ttgir/layouts.py` defines immutable layout dataclasses (`LayoutMetadata`, `BlockedLayout`, `SliceLayout`) that will be filled by future log parsers or LLM agents reading compiled MLIR dumps such as `extracted_mlir.txt`. Today this step is a stub to be supplied by tooling like `semantics_builder.py`.
3. **IR annotation** – `src/ir/builder.py` walks the Python AST and produces an `AnnotatedKernel`. Each `AnnotatedNode` stores the AST statement, inferred op kind, and whatever layout info is available from TTGIR metadata.
4. **Mapping** – `src/mapping/registry.py` loads declarative lowering rules (currently `arange.yml`, `program_id.yml`). These `MappingSpec`s describe how a Triton op expands into Gluon statements and what prerequisites (e.g., layout) are needed. The `LoweringPipeline` consults this registry per node.
5. **Codegen** – `src/codegen/lowering.py` turns annotated nodes plus mapping specs into `GluonInstruction`s, assembles them into `GluonKernel`s and a `GluonModule`, and `src/codegen/emitter.py` serializes the module into `*.gluon.py` stubs for downstream execution.

## 3. Component Details

### Frontend (`src/frontend`)
- **Purpose** – Convert Triton kernels into `ParsedKernel` objects and collect binding information needed for constexpr reconstruction.
- **Key classes/functions** – `KernelParser.parse_kernel`, `_get_source`, `_get_filename`; `ParsedKernel`, `CallDependency`, and `ConstexprMetadata` dataclasses; `ScopeAnalyzer.analyze`; `SymbolTable.define/lookup`; `SymbolBinding` and `SymbolType` for binding classification.
- **Inputs/Outputs** – Input is a Triton callable (`triton.runtime.JITFunction` or plain Python function). Output is a `ParsedKernel` that now bundles the AST, source text, globals, a breadth-first transitive `call_graph: Dict[str, Sequence[CallDependency]]` keyed by qualified names, and `constexpr_metadata: Sequence[ConstexprMetadata]`, plus a `SymbolTable` of `SymbolBinding` entries.
- **Current state** – Parses successfully but loses formatting comments/docstrings and does not evaluate assignment RHS expressions (bindings are recorded as `None`).
- **Future work** – Preserve trivia for high-fidelity re-emission, safely evaluate constexpr expressions, and thread nonlocal/import scopes.

The frontend stores richer binding metadata: each `SymbolTable` entry wraps its value in a `SymbolBinding` that records both the Python object and its `SymbolType` (one of `VARIABLE`, `BUILTIN`, `JIT_FUNCTION`, `CONSTEXPR`, or `DTYPE`). The call graph captures `CallDependency` edges with both simple and fully qualified names so kernels from different modules do not collide, and `ConstexprMetadata` keeps provenance for constexpr annotations and assignments.

### TTGIR package (`src/ttgir`)
- **Purpose** – Data model for layout/type metadata coming from TTGIR/MLIR dumps or future LLM agents.
- **Key classes** – `LayoutMetadata` base class plus `BlockedLayout` and `SliceLayout` for the concrete layouts used today.
- **Inputs/Outputs** – Expected input is a parsed MLIR/TTGIR log or LLM-produced summary keyed by SSA value id. Output is a `LayoutMap` consumed by `IRBuilder`.
- **Current state** – Pure dataclasses; no parser yet.
- **Future work** – Build extraction pipelines (regex-based first, then LLM-assisted) that populate `LayoutMetadata` instances with accurate tensor shapes and blocking details.

### IR package (`src/ir`)
- **Purpose** – Bridge AST statements with TTGIR metadata so downstream passes can reason about layout-sensitive lowering.
- **Key classes/functions** – `AnnotatedKernel.add_node/walk`, `AnnotatedNode`, `IRBuilder.build` and `_scan_ast`.
- **Inputs/Outputs** – Input is `ParsedKernel` plus an optional `LayoutMap`. Output is an `AnnotatedKernel` containing ordered `AnnotatedNode`s.
- **Current state** – `_scan_ast` emits one node per top-level statement with synthetic ids (`n0`, `n1`, …) and attaches a layout only if it finds a matching key; no fine-grained AST traversal yet.
- **Future work** – Implement true AST/TTGIR alignment, carry dtype/opcode metadata, and create sub-node annotations (e.g., per call or per tensor operation).

### Mapping package (`src/mapping`)
- **Purpose** – Store human-editable lowering specs so Triton ops can be translated without hard-coding Python rewrites.
- **Key classes/functions** – `MappingSpec`, `MappingRegistry.lookup/_load_specs/_parse_stub_yaml`.
- **Inputs/Outputs** – YAML specs under `src/mapping/specs/` (e.g., `arange.yml`, `program_id.yml`) are parsed into `MappingSpec`s. `LoweringPipeline` queries the registry by `AnnotatedNode.op`.
- **Current state** – Minimal spec set, and the lowering pipeline only records which spec matched (no template expansion yet).
- **Future work** – Expand spec coverage, allow parameter binding (e.g., map AST arguments into `{start}`, `{stop}` templates), validate requirements, and version specs alongside Gluon intrinsics.

### Codegen (`src/codegen`)
- **Purpose** – Transform mapped instructions into runnable Gluon modules and emit them to disk.
- **Key classes/functions** – `LoweringPipeline.lower/_lower_node`, `GluonInstruction`, `GluonKernel`, `GluonModule`; `CodeEmitter.emit`.
- **Inputs/Outputs** – Input is an `AnnotatedKernel` plus access to `MappingRegistry`. Output is a serialized module (`Path` to `*.gluon.py`).
- **Current state** – `_lower_node` emits `pass` instructions with TODO comments, and the emitter writes placeholder bodies.
- **Future work** – Interpret `MappingSpec.gluon_template`, generate real `ttgl.*` calls, attach decorators/imports per kernel, and pipe the generated source into the runtime for validation.

## 4. Data Structures

| Structure | Location | Key fields | Role |
|-----------|----------|------------|------|
| `ParsedKernel` | `src/frontend/parser.py` | `kernel`, `tree`, `source`, `filename`, `globals_map`, `call_graph`, `constexpr_metadata` | Captures everything known about a Triton entry point; `call_graph` is a breadth-first adjacency map keyed by qualified kernel names and populated with `CallDependency` entries, while `constexpr_metadata` tracks constexpr bindings and their provenance.
| `SymbolTable` | `src/frontend/scope.py` | `bindings: Dict[str, SymbolBinding]` | Lightweight scope map used to record assignments and globals seen during AST traversal, now tagging each binding with a `SymbolBinding` wrapper to describe its `SymbolType`.
| `CallDependency` | `src/frontend/parser.py` | `name`, `qualified_name`, `module`, `obj` | Represents an edge in the kernel call graph, retaining both the friendly name and fully qualified identifier so similarly named kernels from different modules remain distinct.
| `ConstexprMetadata` | `src/frontend/parser.py` | `name`, `module`, `assignment` | Records constexpr bindings discovered during parsing along with the module path or assignment string used to define them.
| `SymbolBinding` | `src/frontend/scope.py` | `value`, `symbol_type: SymbolType` | Wraps each recorded symbol; `SymbolType` enumerates `VARIABLE`, `BUILTIN`, `JIT_FUNCTION`, `CONSTEXPR`, and `DTYPE` to keep downstream analyses aware of binding intent.
| `LayoutMetadata` | `src/ttgir/layouts.py` | `kind` plus layout-specific fields (`size_per_thread`, etc.) | Normalized TTGIR annotations describing how tensors are distributed.
| `BlockedLayout` / `SliceLayout` | `src/ttgir/layouts.py` | Blocked: per-thread/thread-per-warp metadata; Slice: `dimension`, `parent` | Specific layout forms referenced by TTGIR dumps and eventually by the Gluon lowering.
| `AnnotatedNode` | `src/ir/annotated.py` | `id`, `ast_node`, `op`, `operands`, `dtype`, `layout`, `ttgir_value` | Glue node that pairs AST statements with TTGIR metadata so mapping can reason about intent.
| `AnnotatedKernel` | `src/ir/annotated.py` | `name`, `source`, ordered `nodes` | Container for all annotated statements in a Triton kernel.
| `MappingSpec` | `src/mapping/registry.py` | `name`, `gluon_template`, `requirements` | Declarative lowering recipe consumed by the mapping layer.
| `GluonInstruction` | `src/codegen/lowering.py` | `op`, `args`, `comment` | Intermediate representation of a single Gluon statement.
| `GluonKernel` | `src/codegen/lowering.py` | `name`, `instructions`, `decorators` | Holds the instructions for each lowered kernel.
| `GluonModule` | `src/codegen/lowering.py` | `kernels`, `imports` | Top-level artifact passed to `CodeEmitter` for serialization.

## 5. Design Decisions

- **Separating frontend / TTGIR / IR / mapping / codegen** – Each stage has distinct inputs and can evolve independently. For example, the frontend only needs Python reflection, while TTGIR parsing operates on MLIR dumps. Downstream passes can be unit-tested by injecting synthetic `AnnotatedKernel`s without needing real Triton kernels.
- **Declarative YAML specs** – Specs keep lowering logic data-driven and auditable. They allow non-Python stakeholders (e.g., kernel authors) to describe how a Triton op should appear in Gluon form, and make it easy to diff/test changes. The stub parser in `MappingRegistry` intentionally tolerates `.yml` content even without `pyyaml` so specs remain lightweight during prototyping.
- **Deferring TTGIR extraction to an LLM agent** – MLIR dumps are noisy; bootstrapping with a rule-based parser would slow iteration. By defining `LayoutMetadata` upfront and isolating TTGIR ingestion, we create a clear seam where an agent (via `agent_scanner.py` / `agent_tracer.py`) can read traces and populate layouts before IR construction.
- **Trade-offs** – Modularity introduces more moving pieces than the monolithic reference translator. Today each stage is intentionally skeletal, so the system produces placeholder outputs until TTGIR, mapping, and lowering are implemented. This slows end-to-end utility but dramatically reduces the amount of code that must be rewritten once real metadata becomes available.

## 6. Comparison with Reference Implementation

The reference translator in upstream Triton (`/app/triton/python/triton/tools/triton_to_gluon_translater/translator.py` plus `translator_helpers.py`) performs direct AST-to-AST rewriting:

- A single `TritonToGluonTransformer` walks the Triton AST, rewrites builtins (e.g., `tl.arange`, `tl.load`) into Gluon calls, injects decorators, and eagerly converts layouts via helper utilities.
- Runtime metadata (layouts, tensor descriptors, dot-product lowering strategies, etc.) is hard-coded in procedural helpers such as `tl_dot`, `default_blocked_layout`, and TMA helpers in `translator_helpers.py`.

**How our design differs**

- Instead of mutating the AST in place, we parse once, enrich the tree with TTGIR metadata, and defer lowering to declarative specs executed later in the pipeline.
- Layout knowledge is modeled as data (`LayoutMetadata`) rather than implicit helper logic, making it possible to swap sources (MLIR logs, profiling traces, or LLM summaries).
- Mapping decisions are externalized (YAML) so alternative lowering strategies can coexist without editing transformer code.

**Advantages of the new architecture**

- Better observability: every stage exposes serializable dataclasses, enabling debugging and caching.
- Extensibility: new ops can be added by dropping a spec or a TTGIR extractor without touching the frontend.
- Agent-friendly: because TTGIR ingestion is isolated, an LLM can focus on a single, well-scoped task (produce `LayoutMetadata`), while deterministic code handles the rest.

**Added complexity / trade-offs**

- The staged pipeline requires orchestrating more artifacts before any Gluon code is emitted, which increases the amount of scaffolding and state management.
- Until TTGIR extraction and mapping templates are implemented, the system produces placeholder `pass` instructions, whereas the reference translator already emits functional albeit inefficient kernels.
- Keeping declarative specs in sync with real Gluon semantics demands validation infrastructure that the reference implementation implicitly gets by reusing Gluon helper functions.
