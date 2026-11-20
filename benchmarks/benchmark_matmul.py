"""Matmul throughput: TFLOP/s = 2 * m * n * k / time, square sizes, vs torch.matmul (cuBLAS).

Precision is matched on both sides: FP32 runs compare IEEE FP32 with torch's TF32 disabled,
and TF32 runs compare with torch's TF32 enabled. Never compare FP16 against FP32.
"""

from __future__ import annotations

from functools import partial

import torch

from benchmarks.common import HAS_CUDA_EXT, Table, median_ms, require_gpu
from ops.matmul_op import matmul

SIZES = [512, 1024, 2048, 4096, 8192]
# (label, dtype, allow_tf32)
MODES = [
    ("fp16", torch.float16, False),
    ("bf16", torch.bfloat16, False),
    ("fp32 (IEEE)", torch.float32, False),
    ("fp32 (TF32)", torch.float32, True),
]


def _tflops(flops: int, fn) -> float:
    return flops / (median_ms(fn) * 1e-3) / 1e12


def run(sizes=SIZES, modes=MODES) -> Table:
    table = Table(
        title="matmul (square, m = n = k)",
        columns=["mode", "size", "torch TFLOP/s", "triton TFLOP/s", "triton % of torch"]
        + (["cuda (16×16 tiles) TFLOP/s"] if HAS_CUDA_EXT else []),
        notes=[
            "TFLOP/s = 2·m·n·k / median time. Triton block sizes are autotuned per shape;",
            "the first call per shape compiles and autotunes and is excluded by do_bench warm-up.",
        ],
    )
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        for label, dtype, tf32 in modes:
            torch.backends.cuda.matmul.allow_tf32 = tf32
            for s in sizes:
                a = torch.randn(s, s, device="cuda", dtype=dtype)
                b = torch.randn(s, s, device="cuda", dtype=dtype)
                flops = 2 * s * s * s
                t_torch = _tflops(flops, partial(matmul, a, b, backend="torch"))
                t_triton = _tflops(
                    flops, partial(matmul, a, b, backend="triton", allow_tf32=tf32)
                )
                row = [label, s, t_torch, t_triton, 100 * t_triton / t_torch]
                if HAS_CUDA_EXT:
                    run_cuda = dtype == torch.float32 and not tf32 and s <= 4096
                    cuda_fn = partial(matmul, a, b, backend="cuda")
                    row.append(_tflops(flops, cuda_fn) if run_cuda else "–")
                table.add(*row)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return table


if __name__ == "__main__":
    require_gpu()
    print(run().to_markdown())
