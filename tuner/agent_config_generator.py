#!/usr/bin/env python3
"""Minimal config-generation agent for Triton kernels."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import textwrap
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

# Ensure the project root (sibling of this file's directory) is importable even when
# the script is invoked from arbitrary working directories.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import anyio
from shared.agent_shared import (
    build_default_codex_options,
    extract_first_json_object,
    invoke_agent,
    log_parse_failure_details,
)


CONFIG_AGENT_SYSTEM_PROMPT = textwrap.dedent(
    """
    CRITICAL: Use Codex MCP filesystem tools to inspect the Triton source directly before reasoning. Every workload, dimension, and knob MUST be justified by explicit code you can point to.

    Role & scope:
    - Act as a principal GPU performance engineer preparing fully-specified autotuning plans for Triton kernels.
    - You are responsible for providing runtime parameter wiring, launch grid metadata, representative workloads, and candidate autotune configs. The harness will execute and measure them later.

    Evidence-first workflow:
    1. Read the kernel completely (all @triton.jit entry points, tl.constexpr arguments, helper functions, assertions, and comments explaining constraints).
    2. Enumerate every compile-time knob (BLOCK sizes, tile shapes, accumulator blocking, num_warps, num_stages, etc.) plus runtime parameters (tensor pointers, strides, scalar hyper-parameters, sizes).
    3. Derive how tensors flow through the kernel: identify logical dimensions (M/N/K, sequence_len, head_dim, etc.), strides used in tl.load/tl.store, and any reductions or guards affecting launch grids.
    4. Map those insights into a modular JSON plan that future tooling can consume without any kernel-specific tweaks.

    JSON contract (return EXACTLY one object):
    {
      "kernel_name":   <string, exact @triton.jit name>,
      "gpu_arch":      <string, echo of requested target>,
      "parameters":    [
         {"name": <arg_name>, "kind": "tensor"|"stride"|"dimension"|"scalar"|"value", ...}
      ],
      "metadata": {
         "grid": <string of a lambda META: ... using dims/scalars> ,
         "autotune_keys": [<runtime fields whose values change across workloads>],
         "module_preamble": <optional helper definitions to exec before kernel lookup>
      },
      "workloads": [
         {
           "name": <string>,
           "dimensions": {"M": 4096, "K": 256, ...},
           "scalars": {"p": 0.3, "epsilon": 1e-5, ...},
           "buffers": {
             "x": {"kind": "dense", "shape": ["M", "K"], "dtype": "torch.float16", "init": "randn"},
             "descs": {"kind": "descriptor", "source": "x", "block_shape": ["BLOCK_M", "BLOCK_K"]},
             "ptrs": {"kind": "pointer_array", "source": "tensor_list_name"},
             "tiles": {"kind": "tensor_list", "dtype": "torch.float16", "shapes": [[128, 64], ...]}
           },
           "configs": [
             {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2},
             {"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 32, "num_warps": 8, "num_stages": 3}
           ]
         }
      ]
    }

    Required behavior:
    - Parameter list order MUST match the kernel signature (excluding tl.constexpr args) so harness wiring stays deterministic.
    - `grid` must be a lambda string whose body can reference `dims`, `scalars`, `META`, or Triton helpers exactly as written in the kernel.
    - Include every runtime buffer the kernel touches inside `buffers`, with concrete shapes/dtypes derived from the `dimensions` map.
    - Every workload must stress a different, realistic usage regime (small/medium/large batch, narrow vs. wide matrices, etc.). Minimum of 3 workloads, maximum of 5.
    - Each workload requires at least 2 candidate configs and each config must define ALL compile-time knobs mentioned in the kernel plus `num_warps` and `num_stages`.
    - Never emit cached tuning results, timing data, or placeholder text. All tensor sizes/dtypes must be concrete and actionable.
    - Do NOT wrap the JSON in markdown fences or add commentary outside of the JSON object.
    """
).strip()


def derive_default_output_path(kernel_path: str) -> str:
    """Return default path `<kernel_name>_configs.json` next to the kernel."""
    absolute_kernel_path = os.path.abspath(kernel_path)
    kernel_dir = os.path.dirname(absolute_kernel_path) or os.getcwd()
    kernel_name = os.path.basename(absolute_kernel_path) or "kernel"
    stem = os.path.splitext(kernel_name)[0] or kernel_name
    return os.path.join(kernel_dir, f"{stem}_configs.json")


@dataclass
class ConfigGenerationResult:
    """In-memory representation of the fully specified config payload."""

    kernel_name: str
    gpu_arch: str
    parameters: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    workloads: List[Dict[str, Any]]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kernel_name": self.kernel_name,
            "gpu_arch": self.gpu_arch,
            "parameters": self.parameters,
            "metadata": self.metadata,
            "workloads": self.workloads,
        }


class ConfigPromptBuilder:
    """Builds prompts for the config-generation agent."""

    def build_prompt(self, kernel_path: str, gpu_arch: str) -> str:
        kernel_path = os.path.abspath(kernel_path)
        template = textwrap.dedent(
            f"""
            Prepare a self-contained Triton autotuning plan.

            Kernel path: {kernel_path}
            Target GPU architecture: {gpu_arch}

            You must:
              1. Use Codex MCP filesystem tools only. Open the kernel, list the @triton.jit entry points, and document every runtime parameter (tensor ptrs, strides, dimensions, scalars) along with each tl.constexpr knob.
              2. Summarize launch/grid logic from source (program_id decomposition, cdiv math, loops) so you can reconstruct the `grid` lambda precisely.
              3. Identify all runtime buffers and how they map to logical dimensions. Provide dense/tensor_list/pointer_array/descriptor specs that the harness can materialize without editing future kernels.
              4. Design 3-5 workloads covering distinct problem sizes. Each workload must include `dimensions`, `scalars`, `buffers`, and 2-4 candidate `configs` that set every compile-time knob (plus `num_warps`/`num_stages`).

            JSON format to emit (no markdown, no commentary):
              {{
                "kernel_name": "<exact name>",
                "gpu_arch": "{gpu_arch}",
                "parameters": [
                  {{"name": "arg", "kind": "tensor", "buffer": "x"}},
                  {{"name": "n_cols", "kind": "dimension", "symbol": "N"}},
                  {{"name": "keep_prob", "kind": "scalar", "key": "p"}},
                  {{"name": "seed", "kind": "value", "value": 0}}
                ],
                "metadata": {{
                  "grid": "lambda META: (triton.cdiv(dims['M'], META['BLOCK_M']), dims['N'] // META['BLOCK_N'])",
                  "autotune_keys": ["M", "N"],
                  "module_preamble": "optional helper defs"
                }},
                "workloads": [
                  {{
                    "name": "small_batch",
                    "dimensions": {{"M": 1024, "N": 256}},
                    "scalars": {{"p": 0.2}},
                    "buffers": {{
                      "x": {{"kind": "dense", "shape": ["M", "N"], "dtype": "torch.float16", "init": "randn"}},
                      "bias": {{"kind": "dense", "shape": ["N"], "dtype": "torch.float16", "init": "randn"}},
                      "tiles": {{"kind": "tensor_list", "dtype": "torch.float16", "shapes": [[128, 128], [128, 64]]}},
                      "tile_ptrs": {{"kind": "pointer_array", "source": "tiles"}},
                      "x_desc": {{"kind": "descriptor", "source": "x", "block_shape": ["BLOCK_M", "BLOCK_N"]}}
                    }},
                    "configs": [
                      {{"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2}},
                      {{"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 3}}
                    ]
                  }}
                ]
              }}

            Validation checklist BEFORE responding:
              - Parameter order exactly matches the kernel signature (ignoring tl.constexprs).
              - All buffers have concrete shapes/dtypes derived from the dimension symbols.
              - Every config sets identical knob keys across workloads and respects constraints noted in source (divisibility, limits, etc.).
              - `autotune_keys` only references runtime dimension/scalar names defined above.
              - No field is redundant or vague. If the kernel has custom helper lambdas (pid math, descriptor makers), capture them in `module_preamble` so harnesses stay deterministic.
            """
        ).strip()
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

        parameters = self._validate_parameters(data.get("parameters"))
        metadata = self._validate_metadata(data.get("metadata"))

        workloads: List[Dict[str, Any]] = []
        for idx, workload_entry in enumerate(workloads_data):
            workloads.append(self._validate_workload(idx, workload_entry))

        workloads = self._expand_descriptor_workloads(workloads)

        return ConfigGenerationResult(
            kernel_name=kernel_name,
            gpu_arch=gpu_arch,
            parameters=parameters,
            metadata=metadata,
            workloads=workloads,
        )

    @staticmethod
    def _validate_non_empty_str(value: Any, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"'{field_name}' must be a non-empty string")
        return value.strip()

    def _validate_parameters(self, value: Any) -> List[Dict[str, Any]]:
        if not isinstance(value, list) or not value:
            raise ValueError("Agent response must include a non-empty 'parameters' list")
        normalized: List[Dict[str, Any]] = []
        allowed_kinds = {"tensor", "stride", "dimension", "scalar", "value", "descriptor"}
        for idx, entry in enumerate(value):
            if not isinstance(entry, dict):
                raise ValueError(f"parameters[{idx}] must be a JSON object")
            normalized_entry = dict(entry)
            name = self._validate_non_empty_str(entry.get("name"), f"parameters[{idx}].name")
            kind = self._validate_non_empty_str(entry.get("kind"), f"parameters[{idx}].kind").lower()
            if kind not in allowed_kinds:
                raise ValueError(
                    f"parameters[{idx}].kind '{kind}' is not supported; expected one of {sorted(allowed_kinds)}"
                )
            normalized_entry["name"] = name
            normalized_entry["kind"] = kind
            if kind in {"tensor", "stride", "descriptor"}:
                buffer_name = self._validate_non_empty_str(
                    entry.get("buffer"), f"parameters[{idx}].buffer"
                )
                normalized_entry["buffer"] = buffer_name
            if kind == "stride":
                axis = entry.get("axis")
                if not isinstance(axis, int):
                    raise ValueError(f"parameters[{idx}].axis must be an integer")
                normalized_entry["axis"] = axis
            if kind == "dimension":
                symbol = self._validate_non_empty_str(
                    entry.get("symbol"), f"parameters[{idx}].symbol"
                )
                normalized_entry["symbol"] = symbol
            if kind == "scalar":
                key_name = self._validate_non_empty_str(entry.get("key"), f"parameters[{idx}].key")
                normalized_entry["key"] = key_name
            if kind == "value":
                if "value" not in entry:
                    raise ValueError(f"parameters[{idx}] of kind 'value' must include 'value'")
            normalized.append(normalized_entry)
        return normalized

    def _validate_metadata(self, value: Any) -> Dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("Agent response must include a 'metadata' object")
        normalized = dict(value)
        grid_src = self._validate_non_empty_str(normalized.get("grid"), "metadata.grid")
        autotune_raw = normalized.get("autotune_keys", [])
        if not isinstance(autotune_raw, list):
            raise ValueError("metadata.autotune_keys must be a list")
        autotune_keys = [
            self._validate_non_empty_str(key, f"metadata.autotune_keys[{idx}]")
            for idx, key in enumerate(autotune_raw)
        ]
        module_preamble = normalized.get("module_preamble", "")
        if module_preamble is None:
            module_preamble = ""
        elif not isinstance(module_preamble, str):
            module_preamble = str(module_preamble)
        module_preamble = self._sanitize_module_preamble(module_preamble)
        normalized["grid"] = grid_src
        normalized["autotune_keys"] = autotune_keys
        normalized["module_preamble"] = module_preamble
        return normalized

    def _validate_workload(self, index: int, workload_entry: Any) -> Dict[str, Any]:
        if not isinstance(workload_entry, dict):
            raise ValueError(f"Workload #{index} must be a JSON object")
        name = self._validate_non_empty_str(workload_entry.get("name"), f"workloads[{index}].name")
        dimensions = self._validate_dimensions(index, workload_entry.get("dimensions"))
        scalars = self._validate_scalars(index, workload_entry.get("scalars"))
        buffers = self._validate_buffers(index, workload_entry.get("buffers"))
        configs = self._validate_configs(index, workload_entry.get("configs"))

        normalized = {
            key: workload_entry[key]
            for key in workload_entry
            if key not in {"name", "dimensions", "scalars", "buffers", "configs"}
        }
        normalized.update(
            {
                "name": name,
                "dimensions": dimensions,
                "scalars": scalars,
                "buffers": buffers,
                "configs": configs,
            }
        )
        return normalized

    def _validate_dimensions(self, index: int, value: Any) -> Dict[str, Any]:
        field = f"workloads[{index}].dimensions"
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(f"{field} must be an object")
        normalized: Dict[str, Any] = {}
        for dim_name, dim_value in value.items():
            name = self._validate_non_empty_str(dim_name, f"{field} key")
            normalized[name] = self._validate_dim_expression(
                dim_value, f"{field}['{name}']", allow_zero=False
            )
        return normalized

    def _validate_scalars(self, index: int, value: Any) -> Dict[str, Any]:
        field = f"workloads[{index}].scalars"
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(f"{field} must be an object")
        normalized: Dict[str, Any] = {}
        for scalar_name, scalar_value in value.items():
            name = self._validate_non_empty_str(scalar_name, f"{field} key")
            normalized[name] = self._validate_scalar_value(
                scalar_value, f"{field}['{name}']"
            )
        return normalized

    def _validate_buffers(self, index: int, value: Any) -> Dict[str, Any]:
        field = f"workloads[{index}].buffers"
        if not isinstance(value, dict) or not value:
            raise ValueError(f"{field} must be a non-empty object")
        normalized: Dict[str, Any] = {}
        for buffer_name, spec in value.items():
            name = self._validate_non_empty_str(buffer_name, f"{field} key")
            normalized[name] = self._validate_buffer_spec(name, spec, index)
        return normalized

    def _validate_buffer_spec(self, buffer_name: str, spec: Any, workload_index: int) -> Dict[str, Any]:
        field = f"workloads[{workload_index}].buffers['{buffer_name}']"
        if not isinstance(spec, dict):
            raise ValueError(f"{field} must be an object describing the buffer")
        kind = spec.get("kind", "dense")
        kind = self._validate_non_empty_str(kind, f"{field}.kind").lower()
        validators = {
            "dense": self._validate_dense_buffer,
            "tensor_list": self._validate_tensor_list_buffer,
            "pointer_array": self._validate_pointer_array_buffer,
            "descriptor": self._validate_descriptor_buffer,
        }
        if kind not in validators:
            raise ValueError(
                f"{field}.kind '{kind}' is not supported; expected one of {sorted(validators)}"
            )
        return validators[kind](spec, buffer_name, workload_index)

    def _validate_dense_buffer(self, spec: Dict[str, Any], buffer_name: str, workload_index: int) -> Dict[str, Any]:
        field = f"workloads[{workload_index}].buffers['{buffer_name}']"
        normalized = dict(spec)
        shape = normalized.get("shape")
        dtype = normalized.get("dtype", "torch.float32")
        normalized["kind"] = "dense"
        normalized["shape"] = self._validate_shape_sequence(shape, f"{field}.shape")
        normalized["dtype"] = self._validate_non_empty_str(dtype, f"{field}.dtype")
        return normalized

    def _validate_tensor_list_buffer(
        self, spec: Dict[str, Any], buffer_name: str, workload_index: int
    ) -> Dict[str, Any]:
        field = f"workloads[{workload_index}].buffers['{buffer_name}']"
        normalized = dict(spec)
        dtype = normalized.get("dtype", "torch.float32")
        normalized["kind"] = "tensor_list"
        normalized["dtype"] = self._validate_non_empty_str(dtype, f"{field}.dtype")
        if "elements" in normalized:
            elements = normalized["elements"]
            if not isinstance(elements, list) or not elements:
                raise ValueError(f"{field}.elements must be a non-empty list")
            normalized_elements = []
            for elem_idx, element in enumerate(elements):
                if not isinstance(element, dict):
                    raise ValueError(f"{field}.elements[{elem_idx}] must be an object")
                normalized_element = dict(element)
                shape = normalized_element.get("shape")
                normalized_element["shape"] = self._validate_shape_sequence(
                    shape, f"{field}.elements[{elem_idx}].shape"
                )
                normalized_elements.append(normalized_element)
            normalized["elements"] = normalized_elements
        elif "shapes" in normalized:
            shapes = normalized["shapes"]
            if not isinstance(shapes, list) or not shapes:
                raise ValueError(f"{field}.shapes must be a non-empty list")
            normalized["shapes"] = [
                self._validate_shape_sequence(shape, f"{field}.shapes[{idx}]")
                for idx, shape in enumerate(shapes)
            ]
        else:
            raise KeyError(f"{field} must define 'elements' or 'shapes'.")
        return normalized

    def _validate_pointer_array_buffer(
        self, spec: Dict[str, Any], buffer_name: str, workload_index: int
    ) -> Dict[str, Any]:
        field = f"workloads[{workload_index}].buffers['{buffer_name}']"
        normalized = dict(spec)
        normalized["kind"] = "pointer_array"
        source = self._validate_non_empty_str(normalized.get("source"), f"{field}.source")
        normalized["source"] = source
        dtype = normalized.get("dtype")
        if dtype is not None:
            normalized["dtype"] = self._validate_non_empty_str(dtype, f"{field}.dtype")
        return normalized

    def _validate_descriptor_buffer(
        self, spec: Dict[str, Any], buffer_name: str, workload_index: int
    ) -> Dict[str, Any]:
        field = f"workloads[{workload_index}].buffers['{buffer_name}']"
        normalized = dict(spec)
        normalized["kind"] = "descriptor"
        source = self._validate_non_empty_str(normalized.get("source"), f"{field}.source")
        normalized["source"] = source
        normalized["block_shape"] = self._validate_shape_sequence(
            normalized.get("block_shape"), f"{field}.block_shape"
        )
        if "shape" in normalized:
            normalized["shape"] = self._validate_shape_sequence(
                normalized.get("shape"), f"{field}.shape"
            )
        if "strides" in normalized:
            normalized["strides"] = self._validate_shape_sequence(
                normalized.get("strides"), f"{field}.strides", allow_zero=True
            )
        padding = normalized.get("padding")
        if padding is not None and not isinstance(padding, str):
            normalized["padding"] = str(padding)
        return normalized

    def _validate_shape_sequence(
        self, value: Any, field_name: str, allow_zero: bool = False
    ) -> List[Any]:
        if not isinstance(value, list) or not value:
            raise ValueError(f"{field_name} must be a non-empty list")
        normalized: List[Any] = []
        for idx, entry in enumerate(value):
            normalized.append(
                self._validate_dim_expression(
                    entry, f"{field_name}[{idx}]", allow_zero=allow_zero
                )
            )
        return normalized

    def _validate_configs(self, workload_index: int, value: Any) -> List[Dict[str, Any]]:
        field = f"workloads[{workload_index}].configs"
        if not isinstance(value, list) or len(value) < 2:
            raise ValueError(f"{field} must include at least two config entries")
        normalized: List[Dict[str, Any]] = []
        expected_keys: Optional[set[str]] = None
        for idx, entry in enumerate(value):
            if not isinstance(entry, dict) or not entry:
                raise ValueError(f"{field}[{idx}] must be a non-empty object")
            if "num_warps" not in entry or "num_stages" not in entry:
                raise KeyError(
                    f"{field}[{idx}] must include both 'num_warps' and 'num_stages'"
                )
            normalized_entry: Dict[str, Any] = {}
            for key, knob_value in entry.items():
                normalized_entry[key] = self._validate_config_value(
                    key, knob_value, workload_index, idx
                )
            key_set = set(normalized_entry.keys())
            if expected_keys is None:
                expected_keys = key_set
            elif key_set != expected_keys:
                raise ValueError(
                    f"{field}[{idx}] must define the same knob keys as previous configs: {sorted(expected_keys)}"
                )
            normalized.append(normalized_entry)
        return normalized

    def _validate_dim_expression(
        self, value: Any, field_name: str, allow_zero: bool
    ) -> Any:
        if isinstance(value, bool):
            raise ValueError(f"{field_name} must not be a boolean")
        if isinstance(value, int):
            if not allow_zero and value <= 0:
                raise ValueError(f"{field_name} must be greater than zero")
            if allow_zero and value < 0:
                raise ValueError(f"{field_name} must be non-negative")
            return value
        if isinstance(value, float):
            if not allow_zero and value <= 0:
                raise ValueError(f"{field_name} must be greater than zero")
            if allow_zero and value < 0:
                raise ValueError(f"{field_name} must be non-negative")
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise ValueError(f"{field_name} strings must be non-empty")
            return text
        raise ValueError(f"{field_name} must be an int, float, or expression string")

    def _validate_scalar_value(self, value: Any, field_name: str) -> Any:
        if isinstance(value, (int, float, bool)):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise ValueError(f"{field_name} strings must be non-empty")
            return text
        raise ValueError(f"{field_name} must be numeric, boolean, or expression string")

    def _validate_config_value(
        self,
        key: str,
        value: Any,
        workload_index: int,
        config_index: int,
    ) -> Any:
        field = f"workloads[{workload_index}].configs[{config_index}].{key}"
        if key in {"num_warps", "num_stages"}:
            return self._require_positive_int(value, field)
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise ValueError(f"{field} strings must be non-empty")
            return text
        raise ValueError(f"{field} must be numeric, boolean, or string")

    @staticmethod
    def _require_positive_int(value: Any, field_name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field_name} must be an integer")
        if value <= 0:
            raise ValueError(f"{field_name} must be greater than zero")
        return value

    @staticmethod
    def _sanitize_module_preamble(source: str) -> str:
        if not source:
            return ""
        sanitized = source.replace("triton.language.cuda", "triton.language.extra")
        return sanitized.strip("\n")

    def _expand_descriptor_workloads(self, workloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        expanded: List[Dict[str, Any]] = []
        for workload in workloads:
            tokens = self._descriptor_tokens(workload.get("buffers", {}))
            configs = workload.get("configs", [])
            if not tokens or not configs:
                expanded.append(workload)
                continue
            token_list = sorted(tokens)
            groups: Dict[tuple, List[Dict[str, Any]]] = {}
            for config in configs:
                key = tuple(config.get(token) for token in token_list)
                groups.setdefault(key, []).append(config)
            if len(groups) == 1:
                scalars = dict(workload.get("scalars", {}))
                key = next(iter(groups))
                for token, value in zip(token_list, key):
                    if value is not None:
                        scalars.setdefault(token, value)
                workload["scalars"] = scalars
                expanded.append(workload)
                continue
            base_workload = copy.deepcopy(workload)
            for idx, (key, grouped_configs) in enumerate(groups.items(), start=1):
                clone = copy.deepcopy(base_workload)
                clone["configs"] = grouped_configs
                scalars = dict(clone.get("scalars", {}))
                for token, value in zip(token_list, key):
                    if value is not None:
                        scalars[token] = value
                clone["scalars"] = scalars
                if idx > 1:
                    clone["name"] = f"{clone['name']}_variant{idx}"
                expanded.append(clone)
        return expanded

    def _descriptor_tokens(self, buffers: Dict[str, Any]) -> List[str]:
        tokens: set[str] = set()
        pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
        for spec in buffers.values():
            if not isinstance(spec, dict):
                continue
            if spec.get("kind") != "descriptor":
                continue
            for field in ("block_shape", "shape", "strides"):
                entries = spec.get(field)
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if isinstance(entry, str):
                        for match in pattern.findall(entry):
                            if match.isupper() or match.startswith("rep_"):
                                tokens.add(match)
        return list(tokens)


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

    workload_names = [
        workload.get("name", f"workload_{idx}")
        for idx, workload in enumerate(result.workloads)
    ]
    summary_payload = {
        "config_file": saved_path,
        "kernel_name": result.kernel_name,
        "gpu_arch": result.gpu_arch,
        "workload_count": len(result.workloads),
        "workload_names": workload_names,
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
