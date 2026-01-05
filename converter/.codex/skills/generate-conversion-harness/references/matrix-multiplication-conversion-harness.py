import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import torch
import triton
import triton.language as tl

BASE_DIR = Path(__file__).parent
BASE_NAME = "matrix-multiplication"
KERNEL_NAME = "matmul_kernel"
TRITON_SRC = BASE_DIR / f"{BASE_NAME}.py"
GLUON_SRC = BASE_DIR / f"{BASE_NAME}-gluon.py"
SHAPE_PATH = BASE_DIR / f"{BASE_NAME}-shape.json"
CONFIG_PATH = BASE_DIR / f"{BASE_NAME}-config.json"


def _load_kernel(module_path: Path, module_name: str, attr: str) -> Callable:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, attr):
        raise AttributeError(f"{module_path} does not define {attr}")
    return getattr(module, attr)


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


def _benchmark(fn: Callable, warmup: int, iters: int, device: torch.device) -> float:
    _sync_device(device)
    for _ in range(max(0, warmup)):
        fn()
    _sync_device(device)

    if iters <= 0:
        return 0.0

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync_device(device)
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def _reference_matmul(a: torch.Tensor, b: torch.Tensor, activation: str) -> torch.Tensor:
    ref = torch.matmul(a.float(), b.float())
    if activation == "leaky_relu":
        ref = torch.where(ref >= 0, ref, 0.01 * ref)
    return ref.to(a.dtype)


def _atol_rtol(dtype: torch.dtype) -> Tuple[float, float]:
    mapping = {
        torch.float16: (1e-2, 1e-2),
        torch.bfloat16: (1e-2, 1e-2),
        torch.float32: (1e-4, 1e-4),
    }
    return mapping.get(dtype, (1e-2, 1e-2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Triton and Gluon matmul using shape/config JSON."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to config JSON.")
    parser.add_argument("--shape", type=Path, default=SHAPE_PATH, help="Path to shape JSON.")
    parser.add_argument(
        "--gluon",
        type=Path,
        default=GLUON_SRC,
        help="Path to the Gluon kernel (defaults to <base>-gluon.py next to this file).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    with args.shape.open("r") as f:
        shape_spec = json.load(f)
    with args.config.open("r") as f:
        config = json.load(f)

    device = triton.runtime.driver.active.get_active_torch_device()
    if device.type != "cuda":
        raise RuntimeError(f"Expected a CUDA/ROCm device for Triton/Gluon kernels, got {device}")
    torch.manual_seed(shape_spec.get("seed", 0))

    inputs: List[torch.Tensor] = [
        _make_tensor(spec, device, is_output=False) for spec in shape_spec["inputs"]
    ]
    output_specs = shape_spec["outputs"]
    output_template: List[torch.Tensor] = [
        _make_tensor(spec, device, is_output=True) for spec in output_specs
    ]
    outputs_triton = [tensor.clone() for tensor in output_template]
    outputs_gluon = [tensor.clone() for tensor in output_template]

    if len(inputs) < 2 or not outputs_triton:
        raise ValueError("Expected at least two inputs and one output in shape JSON.")

    a, b = inputs[:2]
    c_triton = outputs_triton[0]
    c_gluon = outputs_gluon[0]
    M, K = a.shape
    _, N = b.shape

    shape_map = {spec["name"]: spec["shape"] for spec in shape_spec["inputs"] + shape_spec["outputs"]}
    grid_expr = shape_spec.get("grid")
    if grid_expr is None:
        raise ValueError("No grid expression provided in shape JSON.")
    grid_fn = eval(grid_expr, {"triton": triton})
    meta = dict(config.get("meta", {}))
    launch = config.get("launch", {})
    num_warps = launch.get("num_warps", 4)
    num_stages = launch.get("num_stages", 2)
    run_cfg = config.get("run", {"warmup": 0, "rep": 1})
    grid = grid_fn(meta, shape_map)

    triton_kernel = _load_kernel(TRITON_SRC, "matmul_triton_module", KERNEL_NAME)
    gluon_kernel = _load_kernel(args.gluon, "matmul_gluon_module", KERNEL_NAME)

    def triton_launcher():
        c_triton.zero_()
        triton_kernel[grid](
            a,
            b,
            c_triton,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c_triton.stride(0),
            c_triton.stride(1),
            **meta,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    def gluon_launcher():
        c_gluon.zero_()
        gluon_kernel[grid](
            a,
            b,
            c_gluon,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c_gluon.stride(0),
            c_gluon.stride(1),
            **meta,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    triton_launcher()
    gluon_launcher()
    _sync_device(device)

    activation = str(meta.get("ACTIVATION", "none"))
    ref = _reference_matmul(a, b, activation)
    atol, rtol = _atol_rtol(c_triton.dtype)
    torch.testing.assert_close(c_triton, ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(c_gluon, ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(c_gluon, c_triton, atol=atol, rtol=rtol)

    triton_ms = _benchmark(triton_launcher, run_cfg["warmup"], run_cfg["rep"], device)
    gluon_ms = _benchmark(gluon_launcher, run_cfg["warmup"], run_cfg["rep"], device)
    speedup = triton_ms / gluon_ms if gluon_ms else float("inf")

    print("Correctness: Triton and Gluon match the PyTorch reference within tolerance.")
    print(f"Triton: {triton_ms:.3f} ms   Gluon: {gluon_ms:.3f} ms   Triton/Gluon ratio: {speedup:.3f}x")


if __name__ == "__main__":
    main()
