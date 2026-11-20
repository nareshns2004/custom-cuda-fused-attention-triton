import pytest
import torch
from marks import dtype_params
from tolerances import MATMUL, assert_close

from ops.matmul_op import matmul

SHAPES = [
    (1, 1, 1),
    (64, 64, 64),
    (100, 70, 130),  # nothing divisible by the block sizes
    (257, 129, 63),
    (256, 512, 128),
    (600, 40, 200),  # >8 tile rows at BLOCK_M=64: grouped ordering spans two groups
]


def _reference(a, b):
    return a.double() @ b.double()


@pytest.mark.parametrize("dtype", dtype_params(native_bf16_math=True))
@pytest.mark.parametrize("m,k,n", SHAPES)
def test_matmul_matches_torch(device, m, k, n, dtype):
    a = torch.randn(m, k, device=device, dtype=dtype)
    b = torch.randn(k, n, device=device, dtype=dtype)
    out = matmul(a, b)
    assert out.shape == (m, n) and out.dtype == dtype
    assert_close(out, _reference(a, b), MATMUL)


@pytest.mark.parametrize("transpose", ["a", "b", "both"])
def test_matmul_non_contiguous(device, transpose):
    m, k, n = 96, 80, 112
    a = torch.randn(k, m, device=device).t() if transpose in ("a", "both") else torch.randn(
        m, k, device=device
    )
    b = torch.randn(n, k, device=device).t() if transpose in ("b", "both") else torch.randn(
        k, n, device=device
    )
    assert_close(matmul(a, b), _reference(a, b), MATMUL)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_matmul_fp32_is_ieee_on_gpu():
    # Regression for C9. Tight FP32 tolerance: TF32 (10-bit mantissa) misses it by orders of
    # magnitude, so this fails if the kernel silently falls back to TF32.
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    torch.testing.assert_close(
        matmul(a, b), (a.double() @ b.double()).float(), atol=1e-4, rtol=1e-4
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_matmul_tf32_opt_in_on_gpu():
    # TF32 is opt-in; its error vs FP64 is ~1e-3 relative per product, so compare loosely.
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    ref = (a.double() @ b.double()).float()
    torch.testing.assert_close(matmul(a, b, allow_tf32=True), ref, atol=5e-2, rtol=1e-2)


def test_matmul_empty_and_zero_k(device):
    assert matmul(torch.randn(0, 5, device=device), torch.randn(5, 3, device=device)).shape == (0, 3)
    out = matmul(torch.randn(4, 0, device=device), torch.randn(0, 3, device=device))
    assert out.shape == (4, 3) and (out == 0).all()


def test_matmul_rejects_bad_inputs():
    with pytest.raises(ValueError):
        matmul(torch.ones(2, 2), torch.ones(2, 2), backend="nope")
    with pytest.raises(ValueError):
        matmul(torch.ones(2, 3), torch.ones(2, 3))
    with pytest.raises(ValueError):
        matmul(torch.ones(2, 2), torch.ones(2, 2, dtype=torch.float16))
