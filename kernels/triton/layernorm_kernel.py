import triton
import triton.language as tl

# Largest row the single-pass path keeps in registers, in bytes (Triton tutorial's limit).
# Above this the kernel loops over the row in chunks.
_MAX_FUSED_BYTES = 65536


@triton.jit
def layernorm_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    n_cols,
    stride_x_row,
    stride_x_col,
    stride_out_row,
    stride_out_col,
    eps,
    BLOCK_SIZE: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
):
    # One program per row. Statistics in FP32 regardless of input dtype.
    row = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row * stride_x_row
    out_row = out_ptr + row * stride_out_row
    cols = tl.arange(0, BLOCK_SIZE)

    if SINGLE_PASS:
        # Whole row fits in one block: read x from memory once.
        mask = cols < n_cols
        x = tl.load(x_row + cols * stride_x_col, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / n_cols
        xc = tl.where(mask, x - mean, 0.0)
        var = tl.sum(xc * xc, axis=0) / n_cols
        rstd = tl.rsqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = xc * rstd * w + b
        tl.store(out_row + cols * stride_out_col, y.to(out_ptr.dtype.element_ty), mask=mask)
    else:
        # Row larger than one block: three passes over the row in BLOCK_SIZE chunks.
        # Two-pass mean/variance (not E[x^2] - E[x]^2) to avoid cancellation.
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for start in range(0, n_cols, BLOCK_SIZE):
            c = start + cols
            acc += tl.load(x_row + c * stride_x_col, mask=c < n_cols, other=0.0).to(tl.float32)
        mean = tl.sum(acc, axis=0) / n_cols

        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for start in range(0, n_cols, BLOCK_SIZE):
            c = start + cols
            mask = c < n_cols
            x = tl.load(x_row + c * stride_x_col, mask=mask, other=0.0).to(tl.float32)
            xc = tl.where(mask, x - mean, 0.0)
            acc += xc * xc
        rstd = tl.rsqrt(tl.sum(acc, axis=0) / n_cols + eps)

        for start in range(0, n_cols, BLOCK_SIZE):
            c = start + cols
            mask = c < n_cols
            x = tl.load(x_row + c * stride_x_col, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + c, mask=mask, other=0.0).to(tl.float32)
            b = tl.load(b_ptr + c, mask=mask, other=0.0).to(tl.float32)
            y = (x - mean) * rstd * w + b
            tl.store(out_row + c * stride_out_col, y.to(out_ptr.dtype.element_ty), mask=mask)


def launch_layernorm(x, weight, bias, out, eps: float = 1e-5, max_block_size: int | None = None):
    """LayerNorm over the last dim of a 2D tensor, any hidden size and any strides.

    weight and bias must be contiguous 1D. `max_block_size` caps the block (power of two);
    rows wider than it take the chunked path. Defaults to 64 KiB of input per row.
    """
    n_rows, n_cols = x.shape
    if n_rows == 0 or n_cols == 0:
        return
    if max_block_size is None:
        max_block_size = _MAX_FUSED_BYTES // x.element_size()
    block = min(triton.next_power_of_2(n_cols), triton.next_power_of_2(max_block_size))
    single_pass = block >= n_cols
    num_warps = min(max(block // 256, 1), 8)
    layernorm_kernel[(n_rows,)](
        x,
        weight,
        bias,
        out,
        n_cols,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        eps,
        BLOCK_SIZE=block,
        SINGLE_PASS=single_pass,
        num_warps=num_warps,
    )
