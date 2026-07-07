// Counter-free timing: median latency over CUDA events. No ncu required (unavailable on the
// target box, ERR_NVGPUCTRPERM). See docs/hardware-and-measurement.md.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <vector>

#include "check.cuh"

namespace moonshot::bench {

// Time `launch` (a callable that enqueues work on the current stream) and return the median
// elapsed milliseconds over `runs` iterations, after `warmup` untimed iterations.
inline double median_ms(const std::function<void()>& launch, int warmup = 10, int runs = 50) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> samples(runs);
  for (int i = 0; i < runs; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&samples[i], start, stop));
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  std::sort(samples.begin(), samples.end());
  return samples[runs / 2];
}

// 2*M*N*K / t  in GFLOP/s, given median milliseconds.
inline double gemm_gflops(int64_t M, int64_t N, int64_t K, double ms) {
  return (2.0 * M * N * K) / (ms * 1e6);
}

}  // namespace moonshot::bench
