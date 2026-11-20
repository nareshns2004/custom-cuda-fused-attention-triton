import pytest
import torch
import torch.nn.functional as F
from marks import dtype_params
from tolerances import ELEMENTWISE, assert_close

from ops.layernorm_op import layernorm

DTYPES = dtype_params(native_bf16_math=False)
C1 = pytest.mark.xfail(
    strict=True, reason="C1: Triton layernorm only normalises the first BLOCK_SIZE=1024 columns"
)


def _inputs(rows, hidden, device, dtype):
    x = torch.randn(rows, hidden, device=device, dtype=dtype) * 3 + 1
    w = torch.randn(hidden, device=device, dtype=dtype)
    b = torch.randn(hidden, device=device, dtype=dtype)
    return x, w, b


def _reference(x, w, b, eps=1e-5):
    hidden = x.shape[-1]
    return F.layer_norm(x.float(), (hidden,), w.float(), b.float(), eps)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "hidden",
    [1, 100, 768, 1000, 1024, pytest.param(4096, marks=C1), pytest.param(8192, marks=C1)],
)
def test_layernorm_matches_torch(device, hidden, dtype):
    x, w, b = _inputs(7, hidden, device, dtype)
    assert_close(layernorm(x, w, b), _reference(x, w, b), ELEMENTWISE)


@C1
def test_layernorm_hidden_not_power_of_two_above_block(device):
    x, w, b = _inputs(3, 5000, device, torch.float32)
    assert_close(layernorm(x, w, b), _reference(x, w, b), ELEMENTWISE)


@pytest.mark.xfail(strict=True, reason="C2: Triton layernorm ignores strides")
def test_layernorm_transposed_view(device):
    hidden = 768
    x = torch.randn(hidden, 16, device=device).t()  # [16, hidden], non-contiguous
    w = torch.randn(hidden, device=device)
    b = torch.randn(hidden, device=device)
    assert not x.is_contiguous()
    assert_close(layernorm(x, w, b), _reference(x, w, b), ELEMENTWISE)


@pytest.mark.xfail(strict=True, reason="C2: Triton layernorm ignores strides")
def test_layernorm_row_slice(device):
    # Rows padded in memory: a column slice of a wider buffer keeps stride(1) == 1 but
    # stride(0) != hidden.
    hidden = 512
    x = torch.randn(9, hidden + 64, device=device)[:, :hidden]
    w = torch.randn(hidden, device=device)
    b = torch.randn(hidden, device=device)
    assert_close(layernorm(x, w, b), _reference(x, w, b), ELEMENTWISE)


def test_layernorm_constant_row_is_finite(device):
    # Zero variance: output must be beta, not NaN.
    hidden = 256
    x = torch.full((2, hidden), 3.0, device=device)
    w = torch.ones(hidden, device=device)
    b = torch.randn(hidden, device=device)
    out = layernorm(x, w, b)
    assert torch.isfinite(out).all()
    assert_close(out, _reference(x, w, b), ELEMENTWISE)
