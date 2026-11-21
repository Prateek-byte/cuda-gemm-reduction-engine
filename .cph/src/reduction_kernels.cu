#include "reduction_kernels.cuh"
#include "cuda_utils.cuh"
#include <vector>
#include <numeric>

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void sharedTreeReduceKernel(const float* __restrict__ input,
                                       float* __restrict__ blockSums,
                                       int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;
    if (idx < n) sum += input[idx];
    if (idx + blockDim.x < n) sum += input[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) blockSums[blockIdx.x] = sdata[0];
}

__global__ void warpShuffleReduceKernel(const float* __restrict__ input,
                                        float* __restrict__ blockSums,
                                        int n) {
    __shared__ float warpSums[32];
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;
    if (idx < n) sum += input[idx];
    if (idx + blockDim.x < n) sum += input[idx + blockDim.x];

    sum = warpReduceSum(sum);

    int lane = threadIdx.x & (warpSize - 1);
    int warpId = threadIdx.x / warpSize;

    if (lane == 0) warpSums[warpId] = sum;
    __syncthreads();

    if (warpId == 0) {
        int numWarps = (blockDim.x + warpSize - 1) / warpSize;
        sum = (threadIdx.x < numWarps) ? warpSums[lane] : 0.0f;
        sum = warpReduceSum(sum);
        if (threadIdx.x == 0) blockSums[blockIdx.x] = sum;
    }
}

float runParallelReduction(const float* d_input, int n, cudaStream_t stream) {
    const int threads = 256;
    int blocks = (n + threads * 2 - 1) / (threads * 2);

    float* d_blockSums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_blockSums, blocks * sizeof(float)));

    GpuTimer timer;
    timer.start(stream);
    warpShuffleReduceKernel<<<blocks, threads, 0, stream>>>(d_input, d_blockSums, n);
    float ms = timer.stop(stream);

    std::vector<float> h_partial(blocks);
    CUDA_CHECK(cudaMemcpyAsync(h_partial.data(), d_blockSums, blocks * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    double bandwidthGBs = (static_cast<double>(n) * sizeof(float) * 1e-6) / ms;
    std::cout << "\n=== Warp-Shuffle Parallel Reduction (" << n << " elements) ===\n";
    std::cout << "Kernel Latency      : " << ms << " ms\n";
    std::cout << "Effective Bandwidth : " << bandwidthGBs << " GB/s\n";

    CUDA_CHECK(cudaFree(d_blockSums));
    return std::accumulate(h_partial.begin(), h_partial.end(), 0.0f);
}