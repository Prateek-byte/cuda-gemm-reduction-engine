#include "cuda_utils.cuh"
#include "gemm_kernels.cuh"
#include "reduction_kernels.cuh"
#include "activation_kernels.cuh"
#include <iostream>
#include <vector>

void printDeviceInfo() {
    int deviceId = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    std::cout << "========================================================\n";
    std::cout << "GPU Device          : " << prop.name << "\n";
    std::cout << "Compute Capability  : " << prop.major << "." << prop.minor << "\n";
    std::cout << "SM Count            : " << prop.multiProcessorCount << "\n";
    std::cout << "Shared Mem / Block  : " << prop.sharedMemPerBlock / 1024 << " KB\n";
    std::cout << "Global Memory       : " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
    std::cout << "========================================================\n";
}

int main() {
    printDeviceInfo();

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 1. Run Multi-Stage GEMM Benchmark (Naive vs Tiled vs Coarsened Bank-Conflict-Free)
    launchGemmBenchmark(1024, 1024, 1024, stream);

    // 2. Run Warp-Shuffle Parallel Reduction Benchmark (16M elements)
    const int N = 1 << 24;
    std::vector<float> h_vec(N, 1.0f);
    float* d_vec = nullptr;
    CUDA_CHECK(cudaMalloc(&d_vec, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(d_vec, h_vec.data(), N * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    float totalSum = runParallelReduction(d_vec, N, stream);
    std::cout << "Reduced Sum         : " << totalSum << " (Expected: " << N << ")\n";
    CUDA_CHECK(cudaFree(d_vec));

    // 3. Run Fused Transformer Kernels (Bias+GELU & Row-Wise Softmax)
    const int seqLen = 512, hiddenDim = 1024;
    size_t matBytes = seqLen * hiddenDim * sizeof(float);
    float *d_in, *d_bias, *d_gelu, *d_softmax;
    CUDA_CHECK(cudaMalloc(&d_in, matBytes));
    CUDA_CHECK(cudaMalloc(&d_bias, hiddenDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gelu, matBytes));
    CUDA_CHECK(cudaMalloc(&d_softmax, matBytes));

    CUDA_CHECK(cudaMemsetAsync(d_in, 0, matBytes, stream));
    CUDA_CHECK(cudaMemsetAsync(d_bias, 0, hiddenDim * sizeof(float), stream));

    GpuTimer timer;
    timer.start(stream);
    launchFusedGelu(d_in, d_bias, d_gelu, seqLen, hiddenDim, stream);
    launchRowSoftmax(d_gelu, d_softmax, seqLen, hiddenDim, stream);
    float llmOpsMs = timer.stop(stream);

    std::cout << "\n=== Fused LLM Activation & Softmax Pipeline (" << seqLen << "x" << hiddenDim << ") ===\n";
    std::cout << "Fused Bias+GELU + Softmax Latency : " << llmOpsMs << " ms\n";

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_gelu));
    CUDA_CHECK(cudaFree(d_softmax));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return 0;
}