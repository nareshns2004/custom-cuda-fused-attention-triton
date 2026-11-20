#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <limits>

namespace cuda_kernels {
void launch_add(const float* a, const float* b, float* out, int n, cudaStream_t stream);
void launch_matmul(
    const float* a, const float* b, float* out, int m, int n, int k, cudaStream_t stream
);
void launch_layernorm(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int hidden_size,
    float eps,
    cudaStream_t stream
);
}  // namespace cuda_kernels

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_INPUT(x) \
    do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x); } while (0)
#define CHECK_SAME_DEVICE(x, y) \
    TORCH_CHECK((x).device() == (y).device(), #x " and " #y " must be on the same device")

// Kernels index with int; reject sizes that would overflow it.
static int checked_int(int64_t v, const char* what) {
    TORCH_CHECK(v <= std::numeric_limits<int>::max(), what, " is too large for int indexing");
    return static_cast<int>(v);
}

static void check_launch(const char* msg) {
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

torch::Tensor cuda_add(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_SAME_DEVICE(a, b);
    TORCH_CHECK(a.sizes() == b.sizes(), "add inputs must have the same shape");

    // Launch on the inputs' device and that device's current stream, not device 0's.
    const c10::cuda::CUDAGuard device_guard(a.device());
    auto out = torch::empty_like(a);
    const int n = checked_int(a.numel(), "numel");
    if (n == 0) {
        return out;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cuda_kernels::launch_add(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        n,
        stream
    );
    check_launch("cuda_add launch failed");
    return out;
}

torch::Tensor cuda_matmul(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "matmul inputs must be 2D");
    TORCH_CHECK(a.size(1) == b.size(0), "inner dimensions must match");
    CHECK_SAME_DEVICE(a, b);

    const c10::cuda::CUDAGuard device_guard(a.device());
    const int m = checked_int(a.size(0), "m");
    const int k = checked_int(a.size(1), "k");
    const int n = checked_int(b.size(1), "n");
    checked_int(a.numel(), "a.numel()");
    checked_int(b.numel(), "b.numel()");
    checked_int(static_cast<int64_t>(m) * n, "output numel");
    auto out = torch::empty({m, n}, a.options());
    if (m == 0 || n == 0) {
        return out;
    }
    if (k == 0) {
        return out.zero_();
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cuda_kernels::launch_matmul(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        m,
        n,
        k,
        stream
    );
    check_launch("cuda_matmul launch failed");
    return out;
}

torch::Tensor cuda_layernorm(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps
) {
    CHECK_INPUT(input);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    TORCH_CHECK(input.dim() == 2, "layernorm input must be 2D [batch, hidden]");
    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, "gamma and beta must be 1D");
    TORCH_CHECK(
        input.size(1) == gamma.size(0) && input.size(1) == beta.size(0),
        "hidden size mismatch"
    );
    CHECK_SAME_DEVICE(input, gamma);
    CHECK_SAME_DEVICE(input, beta);

    const c10::cuda::CUDAGuard device_guard(input.device());
    const int batch_size = checked_int(input.size(0), "batch size");
    const int hidden_size = checked_int(input.size(1), "hidden size");
    checked_int(input.numel(), "input numel");
    auto out = torch::empty_like(input);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cuda_kernels::launch_layernorm(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        hidden_size,
        static_cast<float>(eps),
        stream
    );
    check_launch("cuda_layernorm launch failed");
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &cuda_add, "Elementwise add (CUDA)");
    m.def("matmul", &cuda_matmul, "Matrix multiply (CUDA)");
    m.def("layernorm", &cuda_layernorm, "LayerNorm (CUDA)");
}
