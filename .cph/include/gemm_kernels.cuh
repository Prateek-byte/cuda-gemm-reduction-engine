#pragma once
#include <cuda_runtime.h>

constexpr int TILE_DIM = 16;
constexpr int COARSE_FACTOR = 4;

// 1. Naive Global-Memory GEMM (Baseline)
__global__ void naiveGemmKernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K);

// 2. Shared-Memory Tiled GEMM
__global__ void tiledGemmKernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K);

// 3. Bank-Conflict-Free Padded Shared-Memory + 1D Thread Coarsening GEMM
__global__ void coarsenedTiledGemmKernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int M, int N, int K);

void launchGemmBenchmark(int M, int N, int K, cudaStream_t stream = 0);