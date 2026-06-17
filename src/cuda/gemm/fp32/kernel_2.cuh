#pragma once
#include <cuda_runtime.h>

#include <cassert>

template <int BM, int BN, int BK, int TM>
__global__ void gemm_2_sram_1d_tiling(int M, int N, int K, float alpha,
                                          float* A, float* B, float beta,
                                          float* C) {
  int bkRow = blockIdx.y;
  int bkCol = blockIdx.x;

  A += K * BM * bkRow;
  B += BN * bkCol;
  C += N * BM * bkRow + BN * bkCol;

  assert(BK == TM && "BK Should be same with TM");

  __shared__ float sA[BM * BK];
  __shared__ float sB[BK * BN];

  int tRow = threadIdx.x / BN;
  int tCol = threadIdx.x % BN;

  int innerRowA = threadIdx.x / BK;
  int innerColA = threadIdx.x % BK;

  int innerRowB = threadIdx.x / BN;
  int innerColB = threadIdx.x % BN;

  float sum[TM] = {
      0.0f,
  };

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    sA[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    sB[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    A += BK;
    B += BK * N;

    for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
      float _b = sB[dotIdx * BN + tCol];
      for (int resIdx = 0; resIdx < TM; resIdx++) {
        sum[resIdx] += sA[(tRow * TM + resIdx) * BK + dotIdx] * _b;
      }
    }
    __syncthreads();
  }

  for (int resIdx = 0; resIdx < TM; resIdx++) {
    C[(tRow * TM + resIdx) * N + tCol] =
        alpha * sum[resIdx] + beta * C[(tRow * TM + resIdx) * N + tCol];
  }
}

void launch_2_sram_1d_tiling(int M, int N, int K, float alpha, float* A,
                                    float* B, float beta, float* C) {
  const int BM = 64;
  const int BN = 64;
  const int BK = 8;
  const int TM = 8;
  dim3 block((BM * BN) / TM);
  dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
  gemm_2_sram_1d_tiling<BM, BN, BK, TM>
      <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
