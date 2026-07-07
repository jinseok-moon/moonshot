// CUDA error-checking macro. Preprocessor-based (no std::source_location) for broad
// toolchain compatibility.
#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr)                                                          \
  do {                                                                           \
    cudaError_t _err = (expr);                                                   \
    if (_err != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__,      \
                   __LINE__, cudaGetErrorString(_err));                          \
      std::abort();                                                              \
    }                                                                            \
  } while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())
