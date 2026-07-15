#include <cuda_runtime.h>

namespace cuda_kernels {

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

void launch_add(const float* a, const float* b, float* out, int n, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    add_kernel<<<grid_size, block_size, 0, stream>>>(a, b, out, n);
}

}  // namespace cuda_kernels
