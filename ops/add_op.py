from __future__ import annotations

import torch

from kernels.triton.add_kernel import launch_add as triton_launch_add

try:
    import cuda_kernels
except ImportError:  # pragma: no cover - extension built separately
    cuda_kernels = None


BACKENDS = ("cuda", "triton", "torch")


def _require_cuda_extension():
    if cuda_kernels is None:
        raise ImportError(
            "cuda_kernels extension is not built. Run `pip install -e .` or `make build`."
        )


def add(a: torch.Tensor, b: torch.Tensor, backend: str = "triton") -> torch.Tensor:
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend}. Choose from {BACKENDS}")

    if backend == "torch":
        return a + b

    if backend == "cuda":
        _require_cuda_extension()
        return cuda_kernels.add(a, b)

    out = torch.empty_like(a)
    triton_launch_add(a, b, out)
    return out
