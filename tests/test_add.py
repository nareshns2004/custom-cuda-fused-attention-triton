import pytest
import torch
from marks import dtype_params
from tolerances import ELEMENTWISE, assert_close

from ops.add_op import add

DTYPES = dtype_params(native_bf16_math=True)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [1, 255, 256, 1000, 4097, 100_003])
def test_add_matches_torch(device, n, dtype):
    # Sizes cover: smaller than one block, exactly one block, not a multiple of the block,
    # many blocks.
    a = torch.randn(n, device=device, dtype=dtype)
    b = torch.randn(n, device=device, dtype=dtype)
    assert_close(add(a, b), a.float() + b.float(), ELEMENTWISE)


def test_add_2d_contiguous(device):
    a = torch.randn(33, 77, device=device)
    b = torch.randn(33, 77, device=device)
    out = add(a, b)
    assert out.shape == a.shape
    assert_close(out, a + b, ELEMENTWISE)


@pytest.mark.xfail(strict=True, reason="C13: Triton add uses flat offsets and ignores strides")
def test_add_mixed_layout(device):
    a = torch.randn(64, 48, device=device).t()  # non-contiguous view
    b = torch.randn(48, 64, device=device)  # contiguous
    assert_close(add(a, b), a + b, ELEMENTWISE)


@pytest.mark.xfail(strict=True, reason="C13: Triton add uses flat offsets and ignores strides")
def test_add_strided_1d(device):
    a = torch.randn(200, device=device)[::2]
    b = torch.randn(100, device=device)
    assert_close(add(a, b), a + b, ELEMENTWISE)


def test_add_rejects_bad_backend():
    with pytest.raises(ValueError):
        add(torch.ones(2), torch.ones(2), backend="nope")
