from __future__ import annotations

import torch
import torch.nn.functional as F

from kernels.triton.layernorm_kernel import launch_layernorm as triton_launch_layernorm

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


def layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    backend: str = "triton",
) -> torch.Tensor:
    """LayerNorm over the last dimension of a 2D [rows, hidden] tensor."""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend}. Choose from {BACKENDS}")
    if x.dim() != 2:
        raise ValueError(f"layernorm expects a 2D [rows, hidden] input, got shape {tuple(x.shape)}")
    hidden = x.shape[1]
    if weight.shape != (hidden,) or bias.shape != (hidden,):
        raise ValueError("weight and bias must have shape (hidden,)")

    if backend == "torch":
        return F.layer_norm(x, (hidden,), weight, bias, eps)

    if backend == "cuda":
        _require_cuda_extension()
        return cuda_kernels.layernorm(x, weight, bias, eps)

    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    triton_launch_layernorm(x, weight, bias, out, eps)
    return out
