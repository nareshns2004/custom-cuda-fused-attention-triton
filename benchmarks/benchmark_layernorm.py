"""LayerNorm bandwidth: effective GB/s counting one read of x, one write of y, and gamma/beta.

The CUDA baseline actually reads x three times (audit C10); the formula counts the ideal
traffic so all backends are measured against the same yardstick.
"""

from __future__ import annotations

from functools import partial

import torch

from benchmarks.common import HAS_CUDA_EXT, Table, median_ms, require_gpu
from ops.layernorm_op import layernorm

ROWS = 4096
HIDDEN = [512, 1024, 2048, 4096, 8192, 16384]


def run(rows=ROWS, hidden_sizes=HIDDEN, dtype=torch.float32) -> Table:
    backends = ["torch", "triton"] + (["cuda"] if HAS_CUDA_EXT and dtype == torch.float32 else [])
    table = Table(
        title=f"layernorm ({str(dtype).removeprefix('torch.')}, {rows} rows)",
        columns=["hidden"] + [f"{b} GB/s" for b in backends],
        notes=["Effective bandwidth = (2 × rows × hidden + 2 × hidden) × element size / time."],
    )
    for hidden in hidden_sizes:
        x = torch.randn(rows, hidden, device="cuda", dtype=dtype)
        w = torch.randn(hidden, device="cuda", dtype=dtype)
        b = torch.randn(hidden, device="cuda", dtype=dtype)
        nbytes = (2 * rows * hidden + 2 * hidden) * x.element_size()
        row = [hidden]
        for backend in backends:
            ms = median_ms(partial(layernorm, x, w, b, backend=backend))
            row.append(nbytes / (ms * 1e-3) / 1e9)
        table.add(*row)
    return table


if __name__ == "__main__":
    require_gpu()
    for dtype in (torch.float32, torch.float16):
        print(run(dtype=dtype).to_markdown())
