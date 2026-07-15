from __future__ import annotations

import torch

from kernels.triton.matmul_kernel import launch_matmul as triton_launch_matmul

try:
    import cuda_kernels
except ImportError:  # pragma: no cover
    cuda_kernels = None


BACKENDS = ("cuda", "triton", "torch")


def _require_cuda_extension():
    if cuda_kernels is None:
        raise ImportError(
            "cuda_kernels extension is not built. Run `pip install -e .` or `make build`."
        )


def matmul(a: torch.Tensor, b: torch.Tensor, backend: str = "triton") -> torch.Tensor:
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend}. Choose from {BACKENDS}")

    if backend == "torch":
        return a @ b

    if backend == "cuda":
        _require_cuda_extension()
        return cuda_kernels.matmul(a, b)

    out = torch.empty(a.size(0), b.size(1), device=a.device, dtype=a.dtype)
    triton_launch_matmul(a, b, out)
    return out
