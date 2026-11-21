#pragma once
#include <cuda_runtime.h>

// Fused Bias + GELU Activation Kernel (Transformer / LLM Workload)
__global__ void fusedBiasGeluKernel(const float* __restrict__ input,
                                    const float* __restrict__ bias,
                                    float* __restrict__ output,
                                    int rows, int cols);

// Row-wise Numerically Stable Softmax Kernel (Attention Mechanism)
__global__ void rowSoftmaxKernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int rows, int cols);

void launchFusedGelu(const float* d_in, const float* d_bias, float* d_out,
                     int rows, int cols, cudaStream_t stream = 0);

void launchRowSoftmax(const float* d_in, float* d_out,
                      int rows, int cols, cudaStream_t stream = 0);