#!/usr/bin/env bash
set -e

echo "[1/3] Compiling CUDA Kernels..."
make clean && make

echo "[2/3] Running Execution & Throughput Benchmarks..."
./gpu_engine

echo "[3/3] Optional Nsight Compute Profiling Command:"
echo "ncu --set full --target-processes all -o profile_report ./gpu_engine"