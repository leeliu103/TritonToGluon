# TritonToGluon

An agentic framework for converting Triton kernels to Gluon kernels and optimizing the performance based on Gluon kernels.

## Prerequisites

Set up [AgentTemplate](https://github.com/leeliu103/AgentTemplate) before using this project.

## Usage

1. Create a folder containing your Triton kernel (e.g., `./examples/01-vector-add`)

2. Run the tuner agent:
   ```bash
   cd tuner
   codex -c model_provider="amd-openai"
   ```
   Then prompt: `tune the triton kernel for ../examples/01-vector-add`

3. Run the converter agent:
   ```bash
   cd converter
   codex -c model_provider="amd-openai"
   ```
   Then prompt: `convert the triton kernel to gluon kernel for ../examples/01-vector-add`

## Agents

### Tuner Agent

Tunes Triton kernel configurations for best performance. Generates shape/config JSON files and harness, then iterates on configs using IR dumps to optimize runtime.

### Converter Agent

Converts tuned Triton kernel to Gluon kernel. Uses TTGIR dumps to translate the kernel and iterates until correctness and performance targets are met.

## Architecture

Each agent is powered by the Codex CLI. Each subfolder represents a dedicated agent, containing an `AGENTS.md` file and a collection of skills.

To contribute, update the `AGENTS.md` file or add/update skills in the corresponding agent folder.
