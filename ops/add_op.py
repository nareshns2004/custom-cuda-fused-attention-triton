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
            "cuda_kernels extension is not built. Build it on a GPU host with `make build-cuda`."
        )


def add(a: torch.Tensor, b: torch.Tensor, backend: str = "triton") -> torch.Tensor:
    """Elementwise a + b for same-shape, same-dtype tensors (no broadcasting)."""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend}. Choose from {BACKENDS}")
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        raise ValueError("add expects tensors with the same shape, dtype and device")

    if backend == "torch":
        return a + b

    if backend == "cuda":
        _require_cuda_extension()
        return cuda_kernels.add(a, b)

    # The Triton kernel indexes flat memory, so it needs contiguous operands (audit C13).
    # A non-contiguous input is copied first; that extra read+write is the honest cost of
    # supporting it in a memory-bound kernel. Output is always contiguous.
    a = a.contiguous()
    b = b.contiguous()
    out = torch.empty(a.shape, device=a.device, dtype=a.dtype)
    triton_launch_add(a, b, out)
    return out
