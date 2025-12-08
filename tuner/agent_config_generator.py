#!/usr/bin/env python3
"""Minimal config-generation agent for Triton kernels."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import anyio
from shared.agent_shared import (
    build_default_codex_options,
    extract_first_json_object,
    invoke_agent,
    log_parse_failure_details,
)


CONFIG_AGENT_SYSTEM_PROMPT = (
    "CRITICAL: You MUST use Codex MCP for ALL analysis and generation."
    "Never rely on assumptions or prior knowledge—read the actual source code using Codex tools before making ANY decisions about parameters, shapes, or configs. "
    "Every conclusion must be grounded in observable facts from the files you read with Codex tools.\n"
    "You are a principal GPU kernel performance engineer and Triton autotuning specialist. "
    "Always begin by opening the referenced Triton kernel with Codex MCP filesystem tools (read_file, rg) and study the code until every decorator, function parameter, tl.constexpr knob, loop, and tl.load/store is understood. "
    "Base every conclusion strictly on facts visible in the source—do not speculate about compiler lowering, cache behavior, or register usage unless the code states it explicitly. "
    "For each config, specify ALL compile-time parameters: BLOCK_SIZE, num_stages, num_warps, and any other kernel-specific tunables. "
    "Do NOT omit num_warps—it is a critical launch parameter that affects performance.\n"
    "Methodology:\n"
    "1. Catalog every tunable/constexpr parameter along with explicit constraints from the source (assertions, divisibility, boundary guards).\n"
    "2. Summarize the algorithm, tensor shapes, indexing math, and synchronization exactly as implemented—no hypothetical behavior.\n"
    "3. Relate tunable knobs to problem sizes so you know when tiles, BLOCK_SIZE, or num_warps should scale with batch/sequence dimensions.\n"
    "4. Apply GPU-specific heuristics tied to the requested architecture (e.g., RDNA warp granularity of 64, powers-of-two shapes, realistic num_warps/num_stages combinations).\n"
    "5. Design 3-5 representative workloads that cover distinct real-world scenarios (small/medium/large batches, varying sequence lengths, etc.) and articulate how each stresses the kernel differently.\n"
    "6. For every workload, craft several configs that satisfy kernel constraints, align with the workload's tensor sizes, and leverage the target GPU architecture.\n"
    "Response requirements:\n"
    "- Return exactly one JSON object with keys: `kernel_name`, `gpu_arch`, and `workloads`.\n"
    "- Each workload must include: `name`, `input_shapes`, `output_shapes`, `dtypes`, and a `configs` list containing architecture-aware knob dictionaries.\n"
    "- BLOCK_SIZE (or analogous tiling knobs) should be adapted to the workload's problem dimensions so that small workloads use smaller tiles than large ones.\n"
    "- Do not include markdown fences, commentary outside the JSON object, or extra keys."
)


def derive_default_output_path(kernel_path: str) -> str:
    """Return default path `<kernel_name>_configs.json` next to the kernel."""
    absolute_kernel_path = os.path.abspath(kernel_path)
    kernel_dir = os.path.dirname(absolute_kernel_path) or os.getcwd()
    kernel_name = os.path.basename(absolute_kernel_path) or "kernel"
    stem = os.path.splitext(kernel_name)[0] or kernel_name
    return os.path.join(kernel_dir, f"{stem}_configs.json")


@dataclass
class WorkloadSpec:
    """Description of a representative workload and its tailored configs."""

    name: str
    input_shapes: Dict[str, List[int]]
    output_shapes: Dict[str, List[int]]
    dtypes: Dict[str, str]
    configs: List[Dict[str, Any]]
    best_config: Optional[Dict[str, Any]] = None  # Added by harness after autotuning; reset to None when regenerating configs.


@dataclass
class ConfigGenerationResult:
    """In-memory representation of the workload-aware response."""

    kernel_name: str
    gpu_arch: str
    workloads: List[WorkloadSpec]


class ConfigPromptBuilder:
    """Builds prompts for the config-generation agent."""

    def build_prompt(self, kernel_path: str, gpu_arch: str) -> str:
        kernel_path = os.path.abspath(kernel_path)
        template = (
            "Prepare workload-aware Triton tuning configs for the specified kernel.\n"
            f"Target kernel path: {kernel_path}\n"
            f"Target GPU architecture: {gpu_arch}\n"
            "Follow this process:\n"
            "1. Use Codex MCP filesystem tools (read_file, rg) to fully inspect the kernel:\n"
            "   - Read the kernel source file completely\n"
            "   - Identify ALL @triton.jit functions\n"
            "   - List ALL tl.constexpr parameters (BLOCK_SIZE, num_stages, num_warps, etc.)\n"
            "   - Find ALL assertions, guards, or constraints in the code\n"
            "   CRITICAL: Do this using Codex tools—never assume or guess the kernel structure.\n"
            "2. Describe how tensor dimensions flow through the kernel (loops, indexing, blocking, tl.load/tl.store) purely from source evidence. Pay attention to how BLOCK_SIZE, TILE sizes, num_warps, and num_stages interact with batch/sequence lengths.\n"
            "3. Derive ranges or preferred multiples for each tunable knob using code constraints plus GPU-specific heuristics for the requested architecture (warp granularity, LDS usage limits, etc.).\n"
            "4. Propose 3-5 representative workloads that capture distinct real deployments (e.g., small_batch, medium_batch, long_sequence). Each workload must highlight different tensor shapes or batch/sequence lengths so autotuning can reason about scaling.\n"
            "5. For every workload, specify: a descriptive name, input_shapes (dict of parameter -> shape list), output_shapes, and dtypes using torch dtype strings. Then craft 2-4 configs whose BLOCK_SIZE/tiling choices are a tight fit for that workload's tensor dimensions and GPU architecture. Larger workloads should prefer larger tiles than smaller workloads.\n"
            "6. Reference the kernel's actual tunables only; do not invent knobs. Every config must include num_warps (typically 2, 4, 8, or 16 for RDNA GPUs) and you must choose num_warps based on occupancy and register pressure for each BLOCK_SIZE. Reject configs that violate tl.static_assert or divisibility requirements.\n"
            "Output requirements:\n"
            "- Emit exactly one JSON object matching:\n"
            "{\n"
            '  "kernel_name": "fused_softmax_kernel",\n'
            '  "gpu_arch": "' + gpu_arch + '",\n'
            '  "workloads": [\n'
            '    {\n'
            '      "name": "small_batch",\n'
            '      "input_shapes": {"x": [1024, 256]},\n'
            '      "output_shapes": {"y": [1024, 256]},\n'
            '      "dtypes": {"x": "torch.float16", "y": "torch.float16"},\n'
            '      "configs": [\n'
            '        {"BLOCK_SIZE": 128, "num_stages": 2, "num_warps": 2},\n'
            '        {"BLOCK_SIZE": 256, "num_stages": 3, "num_warps": 4}\n'
            '      ]\n'
            '    },\n'
            '    {\n'
            '      "name": "long_sequence",\n'
            '      "input_shapes": {"q": [4096, 512], "k": [4096, 512]},\n'
            '      "output_shapes": {"attn": [4096, 512]},\n'
            '      "dtypes": {"q": "torch.float16", "k": "torch.float16", "attn": "torch.float16"},\n'
            '      "configs": [\n'
            '        {"BLOCK_SIZE": 256, "num_stages": 2, "num_warps": 4},\n'
            '        {"BLOCK_SIZE": 512, "num_stages": 3, "num_warps": 8},\n'
            '        {"BLOCK_SIZE": 1024, "num_stages": 3, "num_warps": 16}\n'
            '      ]\n'
            '    }\n'
            '  ]\n'
            "}\n"
            "- Do not use ellipses (...) or placeholder text in the JSON output.\n"
            "- Every list must contain complete, concrete objects.\n"
            "- kernel_name must match the actual Triton function implemented at the provided path.\n"
            "- workload configs must be GPU-specific, satisfy kernel constraints, and should align BLOCK_SIZE (or analogous tiles) with the workload's tensor sizes.\n"
            "- Every config dictionary must include BLOCK_SIZE, num_stages, num_warps, and any other kernel-specific compile-time knobs derived directly from the code. Do NOT omit num_warps—it must come from Codex-driven evidence.\n"
            "- Do not include markdown fences, commentary, or extra keys."
        )
        return template


class ConfigResponseParser:
    """Parses the JSON payload emitted by the config agent."""

    def parse(self, raw_response: str) -> ConfigGenerationResult:
        json_str = extract_first_json_object(raw_response)
        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise ValueError("Agent response must be a JSON object")

        kernel_name = self._validate_non_empty_str(data.get("kernel_name"), "kernel_name")
        gpu_arch = self._validate_non_empty_str(data.get("gpu_arch"), "gpu_arch")

        workloads_data = data.get("workloads")
        if not isinstance(workloads_data, list) or not workloads_data:
            raise ValueError("Agent response must include a non-empty 'workloads' list")

        workloads: List[WorkloadSpec] = []
        for idx, workload_entry in enumerate(workloads_data):
            if not isinstance(workload_entry, dict):
                raise ValueError(f"Workload #{idx} must be a JSON object")

            workload_name = self._validate_non_empty_str(
                workload_entry.get("name"), f"workloads[{idx}].name"
            )
            input_shapes = self._validate_shapes_map(
                workload_entry.get("input_shapes"),
                f"workloads[{idx}].input_shapes",
            )
            output_shapes = self._validate_shapes_map(
                workload_entry.get("output_shapes"),
                f"workloads[{idx}].output_shapes",
            )
            dtypes = self._validate_dtypes_map(
                workload_entry.get("dtypes"), f"workloads[{idx}].dtypes"
            )

            max_extent = self._max_tensor_extent(input_shapes, output_shapes)
            configs = self._validate_configs_list(
                workload_entry.get("configs"),
                f"workloads[{idx}].configs",
                workload_name,
                max_extent,
            )
            workloads.append(
                WorkloadSpec(
                    name=workload_name,
                    input_shapes=input_shapes,
                    output_shapes=output_shapes,
                    dtypes=dtypes,
                    configs=configs,
                )
            )

        return ConfigGenerationResult(
            kernel_name=kernel_name,
            gpu_arch=gpu_arch,
            workloads=workloads,
        )

    @staticmethod
    def _validate_non_empty_str(value: Any, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"'{field_name}' must be a non-empty string")
        return value.strip()

    @staticmethod
    def _validate_shapes_map(
        value: Any, field_name: str
    ) -> Dict[str, List[int]]:
        if not isinstance(value, dict) or not value:
            raise ValueError(f"'{field_name}' must be a non-empty object")
        normalized: Dict[str, List[int]] = {}
        for tensor_name, dims in value.items():
            if not isinstance(tensor_name, str) or not tensor_name.strip():
                raise ValueError(f"{field_name} keys must be non-empty strings")
            if not isinstance(dims, list) or not dims:
                raise ValueError(
                    f"{field_name} -> '{tensor_name}' must be a non-empty list of integers"
                )
            normalized_dims: List[int] = []
            for dim in dims:
                if not isinstance(dim, int) or dim <= 0:
                    raise ValueError(
                        f"{field_name} -> '{tensor_name}' entries must be positive integers"
                    )
                normalized_dims.append(dim)
            normalized[tensor_name.strip()] = normalized_dims
        return normalized

    @staticmethod
    def _validate_dtypes_map(value: Any, field_name: str) -> Dict[str, str]:
        if not isinstance(value, dict) or not value:
            raise ValueError(f"'{field_name}' must be a non-empty object")
        normalized: Dict[str, str] = {}
        for tensor_name, dtype in value.items():
            if not isinstance(tensor_name, str) or not tensor_name.strip():
                raise ValueError(f"{field_name} keys must be non-empty strings")
            if not isinstance(dtype, str) or not dtype.strip():
                raise ValueError(
                    f"{field_name} -> '{tensor_name}' values must be dtype strings"
                )
            normalized[tensor_name.strip()] = dtype.strip()
        return normalized

    def _validate_configs_list(
        self,
        value: Any,
        field_name: str,
        workload_name: str,
        max_extent: Optional[int],
    ) -> List[Dict[str, Any]]:
        if not isinstance(value, list) or not value:
            raise ValueError(f"'{field_name}' must be a non-empty list")
        normalized: List[Dict[str, Any]] = []
        for idx, config in enumerate(value):
            if not isinstance(config, dict) or not config:
                raise ValueError(
                    f"{field_name}[{idx}] must be a non-empty object with tuning knobs"
                )
            normalized_config = {str(k): config[k] for k in config}
            self._validate_block_size(
                normalized_config, workload_name, max_extent, idx
            )
            normalized.append(normalized_config)
        return normalized

    @staticmethod
    def _max_tensor_extent(*shape_maps: Dict[str, List[int]]) -> Optional[int]:
        max_extent = 0
        for shape_map in shape_maps:
            for dims in shape_map.values():
                extent = 1
                for dim in dims:
                    extent *= dim
                max_extent = max(max_extent, extent)
        return max_extent or None

    @staticmethod
    def _validate_block_size(
        config: Dict[str, Any],
        workload_name: str,
        max_extent: Optional[int],
        config_index: int,
    ) -> None:
        if max_extent is None or max_extent <= 0:
            return
        if "BLOCK_SIZE" not in config:
            return
        block_size = config["BLOCK_SIZE"]
        if not isinstance(block_size, int):
            raise ValueError(
                f"workload '{workload_name}' config #{config_index} has non-integer BLOCK_SIZE"
            )
        if block_size <= 0:
            raise ValueError(
                f"workload '{workload_name}' config #{config_index} must use a positive BLOCK_SIZE"
            )
        if block_size > max_extent * 4:
            raise ValueError(
                f"workload '{workload_name}' config #{config_index} BLOCK_SIZE is too large for the declared tensor sizes"
            )


class ConfigAgent:
    """Agent wrapper that requests config candidates for Triton kernels."""

    def __init__(
        self,
        prompt_builder: Optional[ConfigPromptBuilder] = None,
        response_parser: Optional[ConfigResponseParser] = None,
    ) -> None:
        self.prompt_builder = prompt_builder or ConfigPromptBuilder()
        self.response_parser = response_parser or ConfigResponseParser()
        self.codex_options = build_default_codex_options(
            system_prompt=CONFIG_AGENT_SYSTEM_PROMPT
        )

    async def generate_configs(
        self, kernel_path: str, gpu_arch: str
    ) -> ConfigGenerationResult:
        if not kernel_path:
            raise ValueError("kernel_path is required")
        if not gpu_arch:
            raise ValueError("gpu_arch is required")

        normalized_path = os.path.abspath(kernel_path)
        if not os.path.exists(normalized_path):
            raise FileNotFoundError(f"Kernel file not found: {normalized_path}")

        prompt = self.prompt_builder.build_prompt(normalized_path, gpu_arch)
        response_text, query_error = await invoke_agent(
            prompt,
            self.codex_options,
            normalized_path,
        )

        if query_error:
            log_parse_failure_details(
                context_label=normalized_path,
                query_error=query_error,
                agent_response=response_text,
            )
            raise query_error

        try:
            return self.response_parser.parse(response_text)
        except ValueError:
            log_parse_failure_details(
                context_label=normalized_path,
                query_error=query_error,
                agent_response=response_text,
            )
            raise


def write_config_file(result: ConfigGenerationResult, output_path: str) -> str:
    """Write the workload-aware response to disk and return the absolute path."""
    if not output_path:
        raise ValueError("output_path is required")
    if not isinstance(result, ConfigGenerationResult):
        raise ValueError("result must be a ConfigGenerationResult")
    if not result.workloads:
        raise ValueError("result must include at least one workload")

    absolute_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)

    payload = asdict(result)
    for workload in payload.get("workloads", []):
        if workload.get("best_config") is None:
            workload.pop("best_config", None)

    with open(absolute_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print(
        f"Wrote workload-aware tuning plan to {absolute_path}",
        file=sys.stderr,
    )
    return absolute_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Triton kernel tuning configs using an AI agent.",
    )
    parser.add_argument(
        "--kernel-path",
        required=True,
        help="Path to the Triton kernel source file.",
    )
    parser.add_argument(
        "--gpu-arch",
        required=True,
        help="GPU architecture identifier (e.g., rdna4, rdna3, mi300) to target tuning heuristics.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "File path for the generated config list. Defaults to placing "
            "a '<kernel>_configs.json' file next to the kernel."
        ),
    )
    parser.add_argument(
        "--print-configs",
        action="store_true",
        help="Print the config list to stdout after writing it to disk.",
    )
    args = parser.parse_args()
    if not args.output:
        args.output = derive_default_output_path(args.kernel_path)
    return args


def _run_cli() -> None:
    args = _parse_args()
    agent = ConfigAgent()
    print(
        (
            f"Generating configs for kernel: {args.kernel_path} "
            f"(GPU arch: {args.gpu_arch})"
        ),
        file=sys.stderr,
    )
    result = anyio.run(
        agent.generate_configs,
        args.kernel_path,
        args.gpu_arch,
    )

    saved_path = write_config_file(result, args.output)

    summary_payload = {
        "config_file": saved_path,
        "kernel_name": result.kernel_name,
        "gpu_arch": result.gpu_arch,
        "workload_count": len(result.workloads),
        "workload_names": [workload.name for workload in result.workloads],
    }
    json.dump(summary_payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    sys.stdout.flush()

    if args.print_configs:
        separator = "\n" + ("-" * 40) + "\n"
        sys.stdout.write(separator)
        json.dump(asdict(result), sys.stdout, indent=2)
        sys.stdout.write("\n")
        sys.stdout.flush()


if __name__ == "__main__":
    _run_cli()
