#include <cuda_runtime.h>
#include <cmath>

namespace cuda_kernels {

constexpr int kWarpSize = 32;
constexpr int kMaxThreads = 1024;

__device__ __forceinline__ float warp_reduce_sum(float value) {
    // Butterfly reduction: after the loop every lane holds the warp's sum.
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

// Sum `value` over the block and return the total to every thread.
// Requires blockDim.x to be a multiple of 32 (full warps for the shuffles), up to 1024.
//
// Safe to call repeatedly in one kernel (audit C3): the original version wrote shared[]
// at the start of the second call while slower warps could still be reading shared[0]
// from the first. Here the trailing __syncthreads() guarantees every thread has read
// `total` before any thread can enter the next call and overwrite shared memory.
__device__ float block_reduce_sum(float value) {
    __shared__ float warp_sums[kMaxThreads / kWarpSize];
    __shared__ float total;

    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const int num_warps = blockDim.x / kWarpSize;

    value = warp_reduce_sum(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        float v = lane < num_warps ? warp_sums[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) {
            total = v;
        }
    }
    __syncthreads();

    const float result = total;
    __syncthreads();
    return result;
}

// One block per row. Teaching baseline: reads each row three times from global memory
// (sum, squared deviations, normalise); see audit C10.
__global__ void layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int batch_size,
    int hidden_size,
    float eps
) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) {
        return;  // uniform across the block, so no thread skips a barrier
    }

    const size_t row_offset = static_cast<size_t>(batch_idx) * hidden_size;
    const float* row = input + row_offset;
    float* out_row = output + row_offset;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        local_sum += row[i];
    }
    const float mean = block_reduce_sum(local_sum) / static_cast<float>(hidden_size);

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const float diff = row[i] - mean;
        local_var += diff * diff;
    }
    const float inv_std =
        rsqrtf(block_reduce_sum(local_var) / static_cast<float>(hidden_size) + eps);

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        out_row[i] = (row[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

void launch_layernorm(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int hidden_size,
    float eps,
    cudaStream_t stream
) {
    if (batch_size == 0 || hidden_size == 0) {
        return;
    }
    // About 4 elements per thread, rounded up to whole warps, between 32 and 1024 threads.
    int threads = (hidden_size + 3) / 4;
    threads = ((threads + kWarpSize - 1) / kWarpSize) * kWarpSize;
    threads = threads < kWarpSize ? kWarpSize : (threads > kMaxThreads ? kMaxThreads : threads);
    layernorm_kernel<<<batch_size, threads, 0, stream>>>(
        input, gamma, beta, output, batch_size, hidden_size, eps
    );
}

}  // namespace cuda_kernels
