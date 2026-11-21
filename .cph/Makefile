NVCC = nvcc
CXXFLAGS = -O3 -std=c++17 --use_fast_math -lineinfo -Iinclude
SRCS = src/main.cu src/gemm_kernels.cu src/reduction_kernels.cu src/activation_kernels.cu
TARGET = gpu_engine

all:
	$(NVCC)$(CXXFLAGS) $(SRCS) -o$(TARGET)

clean:
	rm -f $(TARGET)