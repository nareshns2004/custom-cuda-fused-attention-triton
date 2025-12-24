#include <cuda_runtime.h>

namespace cuda_kernels {

constexpr int TILE_SIZE = 16;

__global__ void matmul_kernel(
    const float* a,
    const float* b,
    float* out,
    int m,
    int n,
    int k
) {
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;
    for (int tile = 0; tile < (k + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int a_col = tile * TILE_SIZE + threadIdx.x;
        int b_row = tile * TILE_SIZE + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] =
            (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
        tile_b[threadIdx.y][threadIdx.x] =
            (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;

        __syncthreads();

        for (int inner = 0; inner < TILE_SIZE; ++inner) {
            acc += tile_a[threadIdx.y][inner] * tile_b[inner][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        out[row * n + col] = acc;
    }
}

void launch_matmul(
    const float* a,
    const float* b,
    float* out,
    int m,
    int n,
    int k,
    cudaStream_t stream
) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
    matmul_kernel<<<grid, block, 0, stream>>>(a, b, out, m, n, k);
}

}  // namespace cuda_kernels
