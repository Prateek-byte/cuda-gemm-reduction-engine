#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "[CUDA Error] " << cudaGetErrorString(err)            \
                      << " in " << __FILE__ << " at line " << __LINE__ << "\n";\
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

struct GpuTimer {
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&startEvent));
        CUDA_CHECK(cudaEventCreate(&stopEvent));
    }

    ~GpuTimer() {
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }

    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(startEvent, stream));
    }

    float stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stopEvent, stream));
        CUDA_CHECK(cudaEventSynchronize(stopEvent));
        float elapsedMs = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));
        return elapsedMs;
    }
};

void printDeviceInfo();