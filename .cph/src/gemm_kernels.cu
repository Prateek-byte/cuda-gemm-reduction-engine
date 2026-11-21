#include "gemm_kernels.cuh"
#include "cuda_utils.cuh"
#include <vector>
#include <cmath>
#include <iomanip>

__global__ void naiveGemmKernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void tiledGemmKernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K) {
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    float acc = 0.0f;

    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_DIM + threadIdx.x;
        int bRow = t * TILE_DIM + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// +1 padding on sA/sB avoids 32-way shared-memory bank conflicts
__global__ void coarsenedTiledGemmKernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int M, int N, int K) {
    __shared__ float sA[TILE_DIM][TILE_DIM + 1];
    __shared__ float sB[TILE_DIM][TILE_DIM * COARSE_FACTOR + 1];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int baseCol = blockIdx.x * (TILE_DIM * COARSE_FACTOR) + threadIdx.x;

    float acc[COARSE_FACTOR] = {0.0f};
    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_DIM + threadIdx.x;
        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;

        int bRow = t * TILE_DIM + threadIdx.y;
        #pragma unroll
        for (int c = 0; c < COARSE_FACTOR; ++c) {
            int col = baseCol + c * TILE_DIM;
            sB[threadIdx.y][threadIdx.x + c * TILE_DIM] =
                (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            float aVal = sA[threadIdx.y][k];
            #pragma unroll
            for (int c = 0; c < COARSE_FACTOR; ++c) {
                acc[c] += aVal * sB[k][threadIdx.x + c * TILE_DIM];
            }
        }
        __syncthreads();
    }

    if (row < M) {
        #pragma unroll
        for (int c = 0; c < COARSE_FACTOR; ++c) {
            int col = baseCol + c * TILE_DIM;
            if (col < N) {
                C[row * N + col] = acc[c];
            }
        }
    }
}

void launchGemmBenchmark(int M, int N, int K, cudaStream_t stream) {
    size_t bytesA = static_cast<size_t>(M) * K * sizeof(float);
    size_t bytesB = static_cast<size_t>(K) * N * sizeof(float);
    size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);

    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 0.5f);
    std::vector<float> h_C(M * N, 0.0f);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));

    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B.data(), bytesB, cudaMemcpyHostToDevice, stream));

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    dim3 coarseGrid((N + TILE_DIM * COARSE_FACTOR - 1) / (TILE_DIM * COARSE_FACTOR),
                    (M + TILE_DIM - 1) / TILE_DIM);

    GpuTimer timer;
    double gflopsFactor = 2.0 * M * N * K * 1e-6;

    // Warmup
    tiledGemmKernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    timer.start(stream);
    naiveGemmKernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
    float naiveMs = timer.stop(stream);

    timer.start(stream);
    tiledGemmKernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
    float tiledMs = timer.stop(stream);

    timer.start(stream);
    coarsenedTiledGemmKernel<<<coarseGrid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
    float coarseMs = timer.stop(stream);

    CUDA_CHECK(cudaMemcpyAsync(h_C.data(), d_C, bytesC, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "\n=== GEMM Kernel Benchmark (" << M << "x" << N << "x" << K << ") ===\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "1. Naive Global GEMM     : " << naiveMs  << " ms (" << (gflopsFactor / naiveMs)  << " GFLOP/s)\n";
    std::cout << "2. Shared-Memory Tiled   : " << tiledMs  << " ms (" << (gflopsFactor / tiledMs)  << " GFLOP/s)\n";
    std::cout << "3. Coarsened + Bank-Free : " << coarseMs << " ms (" << (gflopsFactor / coarseMs) << " GFLOP/s)\n";
    std::cout << "Sample Output C[0][0]    : " << h_C[0]   << " (Expected: " << K * 0.5f << ")\n";

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}