# CUDA Matrix Multiplication, Reduction & Fused LLM Kernel Engine

A modular CUDA C++ compute library implementing high-throughput GPU kernels for **Matrix Multiplication (GEMM)**, **Warp-Shuffle Parallel Reduction**, and **Fused Transformer Operators (Bias+GELU & Row-Wise Softmax)**, complete with host-side asynchronous stream management and Nsight Compute profiling support.

## Architecture & Kernel Optimizations

| Module | Kernel / Feature | Key GPU Optimization Techniques |
| :--- | :--- | :--- |
| `gemm_kernels.cu` | `tiledGemmKernel` | `16x16` on-chip `__shared__` memory tiling with `__syncthreads()` to minimize global DRAM bandwidth bottlenecks |
| `gemm_kernels.cu` | `coarsenedTiledGemmKernel` | **1D Thread Coarsening (`COARSE_FACTOR=4`)** + **+1 Shared Memory Padding** to eliminate 32-way bank conflicts and increase arithmetic intensity |
| `reduction_kernels.cu` | `warpShuffleReduceKernel` | Register-level intra-warp reductions using `__shfl_down_sync` + grid-stride unrolling |
| `activation_kernels.cu` | `fusedBiasGeluKernel` | Kernel fusion combining bias addition and fast `tanhf` GELU activation in a single global memory pass |
| `activation_kernels.cu` | `rowSoftmaxKernel` | Numerically stable 3-pass online max-subtraction Softmax using warp-shuffle reductions |

## Build & Run

### Using Make
    make
    ./gpu_engine

### Using CMake
    mkdir build && cd build
    cmake .. && make -j$(nproc)
    ./gpu_engine

### Profiling with NVIDIA Nsight Compute
    ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./gpu_engine

## Sample Benchmark Output (NVIDIA T4 / Ampere)
    ========================================================
    GPU Device          : Tesla T4
    Compute Capability  : 7.5
    SM Count            : 40
    Shared Mem / Block  : 48 KB
    Global Memory       : 15109 MB
    ========================================================

    === GEMM Kernel Benchmark (1024x1024x1024) ===
    1. Naive Global GEMM     : 4.812 ms (446.276 GFLOP/s)
    2. Shared-Memory Tiled   : 1.490 ms (1441.264 GFLOP/s)
    3. Coarsened + Bank-Free : 0.742 ms (2894.183 GFLOP/s)
    Sample Output C[0][0]    : 512.000 (Expected: 512.000)

    === Warp-Shuffle Parallel Reduction (16777216 elements) ===
    Kernel Latency      : 0.241 ms
    Effective Bandwidth : 278.460 GB/s
    Reduced Sum         : 16777216.000 (Expected: 16777216)

    === Fused LLM Activation & Softmax Pipeline (512x1024) ===
    Fused Bias+GELU + Softmax Latency : 0.068 ms