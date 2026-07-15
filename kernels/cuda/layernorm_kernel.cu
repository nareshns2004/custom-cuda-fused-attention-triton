#include <cuda_runtime.h>
#include <cmath>

namespace cuda_kernels {

__device__ float block_reduce_sum(float value) {
    __shared__ float shared[256];
    shared[threadIdx.x] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    return shared[0];
}

__global__ void layernorm_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int hidden_size,
    float eps
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) {
        return;
    }

    const float* row = input + batch_idx * hidden_size;
    float* out_row = output + batch_idx * hidden_size;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        local_sum += row[i];
    }

    float mean = block_reduce_sum(local_sum) / static_cast<float>(hidden_size);

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = row[i] - mean;
        local_var += diff * diff;
    }

    float inv_std = rsqrtf(block_reduce_sum(local_var) / static_cast<float>(hidden_size) + eps);

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = (row[i] - mean) * inv_std;
        out_row[i] = normalized * gamma[i] + beta[i];
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
    const int block_size = 256;
    layernorm_kernel<<<batch_size, block_size, 0, stream>>>(
        input, gamma, beta, output, batch_size, hidden_size, eps
    );
}

}  // namespace cuda_kernels
