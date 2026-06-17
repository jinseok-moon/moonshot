#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <utils.h>

template <cudaDataType_t Atype, cudaDataType_t Btype,
          cudaDataType_t Ctype = Atype,
          cublasComputeType_t ComputeType = CUBLAS_COMPUTE_32F>
void launch_cublas(int M, int N, int K, float alpha, void* A, void* B,
                   float beta, void* C, cublasHandle_t handle) {
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, Btype, N,
               A, Atype, K, &beta, C, Ctype, N, ComputeType,
               CUBLAS_GEMM_DEFAULT);
}
