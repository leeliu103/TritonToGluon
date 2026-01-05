import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Dict

import torch
import triton
import triton.language as tl

BASE_DIR = Path(__file__).parent
KERNEL_SRC = BASE_DIR / "matrix-multiplication.py"
SHAPE_PATH = BASE_DIR / "matrix-multiplication-shape.json"
CONFIG_PATH = BASE_DIR / "matrix-multiplication-config.json"


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def _load_triton_kernel():
    spec = importlib.util.spec_from_file_location("triton_matmul_kernel", KERNEL_SRC)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load Triton kernel from {KERNEL_SRC}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entry = getattr(module, "matmul_kernel", None)
    if entry is None:
        raise AttributeError("matmul_kernel not found in kernel module")
    return entry


def _make_tensor(spec: Dict[str, Any], device: torch.device, is_output: bool) -> torch.Tensor:
    dtype = _dtype_from_name(spec["dtype"])
    shape = spec["shape"]
    if is_output:
        init = spec.get("initialize", "random")
        if init == "zeros":
            return torch.zeros(tuple(shape), device=device, dtype=dtype)
        return torch.randn(tuple(shape), device=device, dtype=dtype)
    return torch.randn(tuple(shape), device=device, dtype=dtype)


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triton matmul harness driven by shape/config JSON.")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to config JSON (defaults to matrix-multiplication-config.json).",
    )
    parser.add_argument(
        "--shape",
        type=Path,
        default=SHAPE_PATH,
        help="Path to shape JSON (defaults to matrix-multiplication-shape.json).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable warmup/rep loops from config; otherwise run a single iteration.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    with args.shape.open("r") as f:
        shape_spec = json.load(f)
    with args.config.open("r") as f:
        config = json.load(f)

    device = triton.runtime.driver.active.get_active_torch_device()
    torch.manual_seed(shape_spec.get("seed", 0))

    inputs = [_make_tensor(spec, device, is_output=False) for spec in shape_spec["inputs"]]
    outputs = [_make_tensor(spec, device, is_output=True) for spec in shape_spec["outputs"]]

    a, b = inputs
    c = outputs[0]
    M, K = a.shape
    _, N = b.shape

    shape_map = {spec["name"]: spec["shape"] for spec in shape_spec["inputs"] + shape_spec["outputs"]}
    grid_expr = shape_spec.get("grid")
    if grid_expr is None:
        raise ValueError("No grid expression provided in config or shape JSON.")
    grid_fn = eval(grid_expr, {"triton": triton})
    meta = dict(config.get("meta", {}))
    launch = config.get("launch", {})
    num_warps = launch.get("num_warps", 4)
    num_stages = launch.get("num_stages", 2)
    run_cfg = {"warmup": 0, "rep": 1}
    if args.benchmark:
        run_cfg.update(config.get("run", {}))

    grid = grid_fn(meta, shape_map)

    kernel = _load_triton_kernel()

    # Warmup
    for _ in range(run_cfg["warmup"]):
        kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            **meta,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    _sync_device(device)

    times: list[float] = []
    for _ in range(run_cfg["rep"]):
        start = time.perf_counter()
        kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            **meta,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        _sync_device(device)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    mean_ms = sum(times) / len(times) if times else 0.0
    result = {
        "mean_ms": mean_ms,
        "rep": run_cfg["rep"],
        "warmup": run_cfg["warmup"],
        "device": str(device),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
