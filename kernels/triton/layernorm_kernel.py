import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    input_ptr,
    gamma_ptr,
    beta_ptr,
    output_ptr,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * hidden_size
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size

    row = tl.load(input_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(row, axis=0) / hidden_size

    centered = tl.where(mask, row - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / hidden_size
    inv_std = tl.rsqrt(var + eps)

    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)
    out = centered * inv_std * gamma + beta
    tl.store(output_ptr + row_start + offsets, out, mask=mask)


def launch_layernorm(input, gamma, beta, out, eps: float = 1e-5, block_size: int = 1024):
    batch_size, hidden_size = input.shape
    grid = (batch_size,)
    layernorm_kernel[grid](
        input,
        gamma,
        beta,
        out,
        hidden_size,
        eps,
        BLOCK_SIZE=block_size,
    )
