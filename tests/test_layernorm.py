import pytest
import torch
import torch.nn.functional as F
from marks import dtype_params
from tolerances import ELEMENTWISE, assert_close

from kernels.triton.layernorm_kernel import launch_layernorm
from ops.layernorm_op import layernorm

DTYPES = dtype_params(native_bf16_math=False)


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
    # 4096 and 8192 are regressions for C1 (wrong above 1024 columns).
    [1, 100, 768, 1000, 1024, 4096, 8192],
)
def test_layernorm_matches_torch(device, hidden, dtype):
    x, w, b = _inputs(7, hidden, device, dtype)
    assert_close(layernorm(x, w, b), _reference(x, w, b), ELEMENTWISE)


def test_layernorm_hidden_not_power_of_two_above_block(device):
    x, w, b = _inputs(3, 5000, device, torch.float32)
    assert_close(layernorm(x, w, b), _reference(x, w, b), ELEMENTWISE)


def test_layernorm_transposed_view(device):
    # Regression for C2.
    hidden = 768
    x = torch.randn(hidden, 16, device=device).t()  # [16, hidden], non-contiguous
    w = torch.randn(hidden, device=device)
    b = torch.randn(hidden, device=device)
    assert not x.is_contiguous()
    assert_close(layernorm(x, w, b), _reference(x, w, b), ELEMENTWISE)


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


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hidden", [1024, 1025, 3000])
def test_layernorm_chunked_path(device, hidden, dtype):
    # Force rows wider than one block through the chunked loop, including a partial last chunk
    # and a non-contiguous input, without needing a 16k+ hidden size.
    x, w, b = _inputs(5, hidden, device, dtype)
    x = x.t().contiguous().t()  # column-major copy: stride(1) != 1
    out = torch.empty(x.shape, device=device, dtype=dtype)
    launch_layernorm(x, w, b, out, max_block_size=256)
    assert_close(out, _reference(x, w, b), ELEMENTWISE)


def test_layernorm_default_limit_takes_chunked_path(device):
    # 20000 FP32 columns exceed the 64 KiB single-pass limit (16384 elements).
    x, w, b = _inputs(2, 20_000, device, torch.float32)
    assert_close(layernorm(x, w, b), _reference(x, w, b), ELEMENTWISE)
