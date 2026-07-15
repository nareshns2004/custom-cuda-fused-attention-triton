cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Torch REQUIRED)

set(CUDA_KERNEL_SOURCES
    add_kernel.cu
    matmul_kernel.cu
    layernorm_kernel.cu
    bindings.cpp
)

pybind11_add_module(cuda_kernels MODULE ${CUDA_KERNEL_SOURCES})

target_link_libraries(cuda_kernels PRIVATE ${TORCH_LIBRARIES})
target_include_directories(cuda_kernels PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_definitions(cuda_kernels PRIVATE TORCH_EXTENSION_NAME=cuda_kernels)

set_property(TARGET cuda_kernels PROPERTY CUDA_ARCHITECTURES native)
