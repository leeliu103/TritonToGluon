from __future__ import annotations

import json

import pytest

from tuner.agent_config_generator import ConfigResponseParser


def _base_payload() -> dict:
    return {
        "kernel_name": "demo_kernel",
        "gpu_arch": "rdna4",
        "parameters": [
            {"name": "x_ptr", "kind": "tensor", "buffer": "x"},
            {"name": "y_ptr", "kind": "tensor", "buffer": "y"},
            {"name": "n_elements", "kind": "dimension", "symbol": "N"},
            {"name": "keep_prob", "kind": "scalar", "key": "p"},
            {"name": "seed", "kind": "value", "value": 0},
        ],
        "metadata": {
            "grid": "lambda META: (triton.cdiv(dims['N'], META['BLOCK_SIZE']),)",
            "autotune_keys": ["N"],
            "module_preamble": "",
        },
        "workloads": [
            {
                "name": "demo",
                "dimensions": {"N": 1024},
                "scalars": {"p": 0.2},
                "buffers": {
                    "x": {
                        "kind": "dense",
                        "shape": ["N"],
                        "dtype": "torch.float32",
                        "init": "randn",
                    },
                    "y": {
                        "kind": "dense",
                        "shape": ["N"],
                        "dtype": "torch.float32",
                        "init": "randn",
                    },
                    "out": {
                        "kind": "dense",
                        "shape": ["N"],
                        "dtype": "torch.float32",
                        "init": "zeros",
                    },
                },
                "configs": [
                    {"BLOCK_SIZE": 128, "num_warps": 2, "num_stages": 2},
                    {"BLOCK_SIZE": 256, "num_warps": 4, "num_stages": 2},
                ],
            }
        ],
    }


def test_parse_valid_payload_round_trips() -> None:
    parser = ConfigResponseParser()
    payload = _base_payload()
    result = parser.parse(json.dumps(payload))
    assert result.kernel_name == "demo_kernel"
    assert result.gpu_arch == "rdna4"
    assert len(result.workloads) == 1
    assert result.parameters[0]["name"] == "x_ptr"


def test_parse_requires_non_empty_buffers() -> None:
    parser = ConfigResponseParser()
    payload = _base_payload()
    payload["workloads"][0]["buffers"] = {}
    with pytest.raises(ValueError):
        parser.parse(json.dumps(payload))


def test_parse_requires_num_warps_and_stages() -> None:
    parser = ConfigResponseParser()
    payload = _base_payload()
    payload["workloads"][0]["configs"][0].pop("num_warps")
    with pytest.raises(KeyError):
        parser.parse(json.dumps(payload))
