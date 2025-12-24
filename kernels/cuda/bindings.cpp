#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

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
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

torch::Tensor cuda_add(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    TORCH_CHECK(a.sizes() == b.sizes(), "add inputs must have the same shape");

    auto out = torch::empty_like(a);
    int n = static_cast<int>(a.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cuda_kernels::launch_add(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        n,
        stream
    );
    check_cuda_error(cudaGetLastError(), "cuda_add launch failed");
    return out;
}

torch::Tensor cuda_matmul(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "matmul inputs must be 2D");
    TORCH_CHECK(a.size(1) == b.size(0), "inner dimensions must match");

    int m = static_cast<int>(a.size(0));
    int k = static_cast<int>(a.size(1));
    int n = static_cast<int>(b.size(1));
    auto out = torch::empty({m, n}, a.options());
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
    check_cuda_error(cudaGetLastError(), "cuda_matmul launch failed");
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

    int batch_size = static_cast<int>(input.size(0));
    int hidden_size = static_cast<int>(input.size(1));
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
    check_cuda_error(cudaGetLastError(), "cuda_layernorm launch failed");
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &cuda_add, "Elementwise add (CUDA)");
    m.def("matmul", &cuda_matmul, "Matrix multiply (CUDA)");
    m.def("layernorm", &cuda_layernorm, "LayerNorm (CUDA)");
}
