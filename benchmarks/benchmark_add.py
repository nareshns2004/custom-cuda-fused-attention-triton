"""Vector add bandwidth: effective GB/s = 3 * n * bytes_per_element / time."""

from __future__ import annotations

from functools import partial

import torch

from benchmarks.common import HAS_CUDA_EXT, Table, median_ms, require_gpu
from ops.add_op import add

SIZES = [2**p for p in range(10, 29, 2)]  # 2^10 .. 2^28 elements


def run(sizes=SIZES, dtype=torch.float32) -> Table:
    backends = ["torch", "triton"] + (["cuda"] if HAS_CUDA_EXT and dtype == torch.float32 else [])
    table = Table(
        title=f"add ({str(dtype).removeprefix('torch.')})",
        columns=["n"] + [f"{b} GB/s" for b in backends],
        notes=["Effective bandwidth = 3 × n × element size / median time (2 reads, 1 write)."],
    )
    for n in sizes:
        a = torch.randn(n, device="cuda", dtype=dtype)
        b = torch.randn(n, device="cuda", dtype=dtype)
        nbytes = 3 * n * a.element_size()
        row = [n]
        for backend in backends:
            ms = median_ms(partial(add, a, b, backend=backend))
            row.append(nbytes / (ms * 1e-3) / 1e9)
        table.add(*row)
    return table


if __name__ == "__main__":
    require_gpu()
    print(run().to_markdown())
