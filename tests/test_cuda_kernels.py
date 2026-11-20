"""Tests for the CUDA learning-set extension. GPU host only, after `make build-cuda`."""

import pytest
import torch
import torch.nn.functional as F
from tolerances import ELEMENTWISE, MATMUL, assert_close

cuda_kernels = pytest.importorskip("cuda_kernels", reason="build with `make build-cuda`")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]


@pytest.mark.parametrize("n", [1, 255, 256, 1000, 1 << 20])
def test_cuda_add(n):
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    assert_close(cuda_kernels.add(a, b), a + b, ELEMENTWISE)


@pytest.mark.parametrize("m,k,n", [(1, 1, 1), (64, 64, 64), (100, 70, 130), (257, 129, 63)])
def test_cuda_matmul(m, k, n):
    a = torch.randn(m, k, device="cuda")
    b = torch.randn(k, n, device="cuda")
    # CUDA-core FP32 FMA, no TF32: compare with a float64 reference.
    assert_close(cuda_kernels.matmul(a, b), a.double() @ b.double(), MATMUL)


@pytest.mark.parametrize("hidden", [1, 31, 33, 100, 768, 1024, 1025, 4096, 8192])
def test_cuda_layernorm(hidden):
    # Hidden sizes cover partial warps, a single warp, and the 1024-thread cap.
    x = torch.randn(64, hidden, device="cuda") * 3 + 1
    w = torch.randn(hidden, device="cuda")
    b = torch.randn(hidden, device="cuda")
    expected = F.layer_norm(x, (hidden,), w, b, 1e-5)
    assert_close(cuda_kernels.layernorm(x, w, b, 1e-5), expected, ELEMENTWISE)


def test_cuda_layernorm_repeated_is_deterministic():
    # C3 was a race between the two block reductions; a racy kernel can return different
    # results across runs. Not a proof of absence: `make racecheck` is.
    x = torch.randn(512, 4096, device="cuda")
    w = torch.randn(4096, device="cuda")
    b = torch.randn(4096, device="cuda")
    first = cuda_kernels.layernorm(x, w, b, 1e-5)
    for _ in range(20):
        torch.testing.assert_close(cuda_kernels.layernorm(x, w, b, 1e-5), first, atol=0, rtol=0)


def test_cuda_rejects_non_contiguous():
    x = torch.randn(64, 32, device="cuda").t()
    with pytest.raises(RuntimeError, match="contiguous"):
        cuda_kernels.add(x, x)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_cuda_runs_on_non_default_device():
    # Device guard: inputs on cuda:1 while cuda:0 is current.
    a = torch.randn(1000, device="cuda:1")
    b = torch.randn(1000, device="cuda:1")
    out = cuda_kernels.add(a, b)
    assert out.device == a.device
    assert_close(out, a + b, ELEMENTWISE)
