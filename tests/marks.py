import os

import pytest
import torch

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"

# Triton's CPU interpreter runs on NumPy, which has no bfloat16: native bf16 arithmetic
# (a + b, tl.dot) on raw bf16 values produces garbage (verified: max error ~1e38 for a bf16
# add, exact after .to(tl.float32)). Kernels that do bf16 math natively are only testable on
# a GPU. Kernels that upcast to FP32 first (layernorm) are tested in bf16 everywhere.
SKIP_BF16_IN_INTERPRETER = pytest.mark.skipif(
    INTERPRET, reason="Triton interpreter can't do native bf16 arithmetic; runs on GPU"
)


def dtype_params(native_bf16_math: bool):
    bf16 = (
        pytest.param(torch.bfloat16, marks=SKIP_BF16_IN_INTERPRETER, id="bf16")
        if native_bf16_math
        else pytest.param(torch.bfloat16, id="bf16")
    )
    return [pytest.param(torch.float32, id="fp32"), pytest.param(torch.float16, id="fp16"), bf16]
