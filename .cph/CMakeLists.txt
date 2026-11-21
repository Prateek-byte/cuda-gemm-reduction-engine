cmake_minimum_required(VERSION 3.18)
project(CudaGemmReductionEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

include_directories(include)

add_executable(gpu_engine
    src/main.cu
    src/gemm_kernels.cu
    src/reduction_kernels.cu
    src/activation_kernels.cu
)

target_compile_options(gpu_engine PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
)