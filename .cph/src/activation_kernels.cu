#include "activation_kernels.cuh"
#include "cuda_utils.cuh"
#include <cmath>

// Fused Bias + Exact/Tanh-Approx GELU used in LLM MLP layers
__global__ void fusedBiasGeluKernel(const float* __restrict__ input,
                                    const float* __restrict__ bias,
                                    float* __restrict__ output,
                                    int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        float x = input[row * cols + col] + bias[col];
        // Fast tanh GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        constexpr float kAlpha = 0.7978845608f; // sqrt(2 / pi)
        constexpr float kBeta  = 0.044715f;
        float x3 = x * x * x;
        float inner = kAlpha * (x + kBeta * x3);
        output[row * cols + col] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

// Warp-parallel online/row-wise Softmax for Transformer Attention scores
__global__ void rowSoftmaxKernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* rowIn = input + row * cols;
    float* rowOut = output + row * cols;

    // 1. Find row max for numerical stability
    float threadMax = -INFINITY;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        threadMax = fmaxf(threadMax, rowIn[c]);
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        threadMax = fmaxf(threadMax, __shfl_down_sync(0xffffffff, threadMax, offset));
    }

    __shared__ float sMax[32];
    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    if (lane == 0) sMax[warpId] = threadMax;
    __syncthreads();

    float rowMax = (threadIdx.x < (blockDim.x >> 5)) ? sMax[lane] : -INFINITY;
    if (warpId == 0) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            rowMax = fmaxf(rowMax, __shfl_down_sync(0xffffffff, rowMax, offset));
        }
        if (lane == 0) sMax[0] = rowMax;
    }
    __syncthreads();
    rowMax = sMax[0];

    // 2. Compute exp(x - max) and row sum
    float threadSum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float e = expf(rowIn[c] - rowMax);
        rowOut[c] = e;
        threadSum += e;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        threadSum += __shfl_down_sync(0xffffffff, threadSum, offset);
    }

    __shared__ float sSum[32];
    if (lane == 0) sSum[warpId] = threadSum;
    __syncthreads();

    float rowSum = (threadIdx.x < (blockDim.x >> 5)) ? sSum[lane] : 0.0f;
    if (warpId == 0) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            rowSum += __shfl_down_sync(0xffffffff, rowSum, offset);
        }
        if (lane == 0) sSum[0] = rowSum;
    }
    __syncthreads();
    rowSum = sSum[0];

    // 3. Normalize
    float invSum = 1.0f / rowSum;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        rowOut[c] *= invSum;
    }
}

void launchFusedGelu(const float* d_in, const float* d_bias, float* d_out,
                     int rows, int cols, cudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    fusedBiasGeluKernel<<<grid, block, 0, stream>>>(d_in, d_bias, d_out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void launchRowSoftmax(const float* d_in, float* d_out,
                      int rows, int cols, cudaStream_t stream) {
    rowSoftmaxKernel<<<rows, 256, 0, stream>>>(d_in, d_out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}