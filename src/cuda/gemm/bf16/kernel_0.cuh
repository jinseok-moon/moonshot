#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <utils.h>

__global__ void gemm_0_dram_coalescing_bf16(int M, int N, int K, float alpha,
                                            __nv_bfloat16* A, __nv_bfloat16* B,
                                            float beta, float* C) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int row = tid / N;
  int col = tid % N;

  if (row >= M || col >= N) return;

  float sum = 0.0f;
  for (int k = 0; k < K; k++) {
    sum += __bfloat162float(A[row * K + k]) * __bfloat162float(B[k * N + col]);
  }
  C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

void launch_0_bf16(int M, int N, int K, float alpha, __nv_bfloat16* A,
                   __nv_bfloat16* B, float beta, float* C) {
  int BLOCKSIZE = 256;
  dim3 block(BLOCKSIZE);
  dim3 grid(ceil_div(M * N, BLOCKSIZE));
  gemm_0_dram_coalescing_bf16<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
