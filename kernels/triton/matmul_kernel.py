import torch
import triton
import triton.language as tl


def _autotune_configs():
    # (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps). Configs that don't fit a GPU's
    # shared memory (e.g. large tiles in FP32) fail to compile and are skipped by the autotuner.
    shapes = [
        (128, 256, 64, 3, 8),
        (256, 128, 64, 3, 8),
        (128, 128, 64, 4, 4),
        (128, 128, 32, 4, 4),
        (128, 64, 32, 4, 4),
        (64, 128, 32, 4, 4),
        (64, 64, 32, 5, 2),
        (32, 64, 32, 5, 2),
    ]
    return [
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8},
            num_stages=stages,
            num_warps=warps,
        )
        for bm, bn, bk, stages, warps in shapes
    ]


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
    GROUP_M: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    # Grouped tile ordering: consecutive programs walk down GROUP_M rows of output tiles
    # before moving right, so the A row-panels and B column-panels they share stay in L2.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(m, BLOCK_M)
    num_pid_n = tl.cdiv(n, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

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
        # INPUT_PRECISION only matters for FP32 operands: "ieee" is true FP32, "tf32" uses
        # tensor cores with a 10-bit mantissa (Triton's default on Ampere+, audit C9).
        acc = tl.dot(a, b, acc, input_precision=INPUT_PRECISION)

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(out_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )


matmul_kernel_autotuned = triton.autotune(configs=_autotune_configs(), key=["m", "n", "k"])(
    matmul_kernel
)


def _grid(m, n):
    return lambda meta: (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)


def launch_matmul(a, b, out, allow_tf32: bool = False):
    """out = a @ b for 2D a [m, k], b [k, n]; any strides; FP32 accumulation.

    FP32 inputs use IEEE FP32 math unless allow_tf32=True (matches torch.matmul's default).
    On a GPU the block sizes are autotuned per (m, n, k, dtype); under the CPU interpreter a
    fixed config is used because autotuning there would only time the interpreter.
    """
    m, k = a.shape
    _, n = b.shape
    if m == 0 or n == 0:
        return
    # None = backend default; only FP32 operands have a precision choice.
    precision = None
    if a.dtype == torch.float32:
        precision = "tf32" if allow_tf32 else "ieee"
    args = (
        a, b, out, m, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), out.stride(0), out.stride(1),
    )
    if a.device.type == "cpu":
        matmul_kernel[_grid(m, n)](
            *args, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, GROUP_M=8, INPUT_PRECISION=precision
        )
    else:
        matmul_kernel_autotuned[_grid(m, n)](*args, INPUT_PRECISION=precision)
