"""Per-dtype tolerances for comparing kernel output with an FP32 PyTorch reference.

The only place tolerances are defined. Never loosen one to make a test pass without
explaining why in the PR.

Kernels compute in FP32 internally; low-precision tolerances cover rounding of the inputs
and of the stored output, not accumulation error.
"""

import torch

# (atol, rtol) per output dtype, for elementwise and normalisation kernels.
ELEMENTWISE = {
    torch.float32: (1e-5, 1e-5),
    # fp16 has a 10-bit mantissa: one ulp is ~1e-3 relative; outputs of layernorm are O(1-5).
    torch.float16: (2e-3, 2e-3),
    # bf16 has a 7-bit mantissa: one ulp is ~8e-3 relative.
    torch.bfloat16: (1.6e-2, 1.6e-2),
}

# (atol, rtol) per output dtype for matmul. Different summation order vs the reference makes
# FP32 error grow with K; inputs here have K <= 512 and unit-variance entries.
MATMUL = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (8e-2, 2e-2),
}


def assert_close(actual: torch.Tensor, expected: torch.Tensor, table: dict) -> None:
    atol, rtol = table[actual.dtype]
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)
