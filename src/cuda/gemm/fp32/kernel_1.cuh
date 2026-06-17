#pragma once
#include <cuda_runtime.h>

template <int BLOCKSIZE>
__global__ void gemm_1_sram_caching(int M, int N, int K, float alpha,
                                        float* A, float* B, float beta,
                                        float* C) {
  int bkRow = blockIdx.y;
  int bkCol = blockIdx.x;

  A += K * BLOCKSIZE * bkRow;
  B += BLOCKSIZE * bkCol;
  C += N * BLOCKSIZE * bkRow + BLOCKSIZE * bkCol;

  __shared__ float sA[BLOCKSIZE * BLOCKSIZE];
  __shared__ float sB[BLOCKSIZE * BLOCKSIZE];

  int tRow = threadIdx.x / BLOCKSIZE;
  int tCol = threadIdx.x % BLOCKSIZE;

  float sum = 0.0f;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    sA[threadIdx.x] = A[tRow * K + tCol];
    sB[threadIdx.x] = B[tRow * N + tCol];
    __syncthreads();

    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    for (int tIter = 0; tIter < BLOCKSIZE; tIter++) {
      sum += sA[tRow * BLOCKSIZE + tIter] * sB[tIter * BLOCKSIZE + tCol];
    }
    __syncthreads();
  }

  C[tRow * N + tCol] = alpha * sum + beta * C[tRow * N + tCol];
}

void launch_1_sram_caching(int M, int N, int K, float alpha, float* A,
                                  float* B, float beta, float* C) {
  const int BLOCKSIZE = 32;
  dim3 block(BLOCKSIZE * BLOCKSIZE);
  dim3 grid(ceil_div(N, BLOCKSIZE), ceil_div(M, BLOCKSIZE));
  gemm_1_sram_caching<BLOCKSIZE>
      <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
