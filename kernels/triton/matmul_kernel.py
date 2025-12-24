import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for tile_k in range(0, k, BLOCK_K):
        k_offsets = tile_k + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak,
            mask=(offs_m[:, None] < m) & (k_offsets[None, :] < k),
            other=0.0,
        )
        b = tl.load(
            b_ptr + k_offsets[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k_offsets[:, None] < k) & (offs_n[None, :] < n),
            other=0.0,
        )
        acc += tl.dot(a, b)

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )


def launch_matmul(a, b, out, block_m: int = 64, block_n: int = 64, block_k: int = 32):
    m, k = a.shape
    _, n = b.shape
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]),
        triton.cdiv(n, meta["BLOCK_N"]),
    )
    matmul_kernel[grid](
        a,
        b,
        out,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
