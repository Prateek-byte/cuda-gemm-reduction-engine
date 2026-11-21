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
```bash
make
./gpu_engine