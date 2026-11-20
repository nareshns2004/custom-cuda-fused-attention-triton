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
            "cuda_kernels extension is not built. Build it on a GPU host with `make build-cuda`."
        )


def matmul(
    a: torch.Tensor, b: torch.Tensor, backend: str = "triton", allow_tf32: bool = False
) -> torch.Tensor:
    """a @ b for 2D tensors of the same dtype and device.

    allow_tf32 applies to FP32 inputs on the triton backend only; the default (False) gives
    IEEE FP32 results comparable with torch.matmul's default.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend}. Choose from {BACKENDS}")
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("matmul expects 2D inputs")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"inner dimensions differ: {tuple(a.shape)} @ {tuple(b.shape)}")
    if a.dtype != b.dtype or a.device != b.device:
        raise ValueError("matmul expects inputs with the same dtype and device")

    if backend == "torch":
        return a @ b

    if backend == "cuda":
        _require_cuda_extension()
        return cuda_kernels.matmul(a, b)

    out = torch.empty(a.size(0), b.size(1), device=a.device, dtype=a.dtype)
    triton_launch_matmul(a, b, out, allow_tf32=allow_tf32)
    return out
