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
@pytest.mark.xfail(strict=False, reason="C9: FP32 tl.dot defaults to TF32 on Ampere+")
def test_matmul_fp32_is_ieee_on_gpu():
    # Tight FP32 tolerance: TF32 (10-bit mantissa) inputs miss this by orders of magnitude.
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    torch.testing.assert_close(matmul(a, b), (a.double() @ b.double()).float(), atol=1e-4, rtol=1e-4)


def test_matmul_rejects_bad_backend():
    with pytest.raises(ValueError):
        matmul(torch.ones(2, 2), torch.ones(2, 2), backend="nope")
