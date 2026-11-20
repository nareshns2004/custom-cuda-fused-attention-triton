# CUDA_ARCHITECTURES "native" needs CMake 3.24 (audit C5).
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(cuda_kernels LANGUAGES CXX CUDA)

# Current PyTorch headers require C++20 (audit C5).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the GPU in this machine unless told otherwise, e.g. -DCMAKE_CUDA_ARCHITECTURES=90
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Torch REQUIRED)

# torch::Tensor <-> Python conversion lives in libtorch_python, which TORCH_LIBRARIES omits.
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)

pybind11_add_module(cuda_kernels MODULE
    add_kernel.cu
    matmul_kernel.cu
    layernorm_kernel.cu
    bindings.cpp
)

target_link_libraries(cuda_kernels PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
target_include_directories(cuda_kernels PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_definitions(cuda_kernels PRIVATE TORCH_EXTENSION_NAME=cuda_kernels)
