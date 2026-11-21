#pragma once
#include <cuda_runtime.h>

// 1. Shared-Memory Tree Reduction (Sequential Addressing)
__global__ void sharedTreeReduceKernel(const float* __restrict__ input,
                                       float* __restrict__ blockSums,
                                       int n);

// 2. Warp-Shuffle (__shfl_down_sync) + Vectorized float4 Load Reduction
__global__ void warpShuffleReduceKernel(const float* __restrict__ input,
                                        float* __restrict__ blockSums,
                                        int n);

float runParallelReduction(const float* d_input, int n, cudaStream_t stream = 0);