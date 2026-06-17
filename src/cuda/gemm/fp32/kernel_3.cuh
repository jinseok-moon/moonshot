#pragma once
#include <cuda_runtime.h>

#include <cassert>

template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_3_sram_2d_tiling(int M, int N, int K, float alpha,
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

  int tRow = threadIdx.x / (BN / TN);
  int tCol = threadIdx.x % (BN / TN);

  int innerRowA = threadIdx.x / BK;
  int innerColA = threadIdx.x % BK;

  int innerRowB = threadIdx.x / BN;
  int innerColB = threadIdx.x % BN;

  float regM[TM] = {
      0.0f,
  };
  float regN[TN] = {
      0.0f,
  };

  float sum[TM * TN] = {
      0.0f,
  };

  int strideA = blockDim.x / BK;
  int strideB = blockDim.x / BN;

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    for (int offset = 0; offset < BM; offset += strideA) {
      sA[(innerRowA + offset) * BK + innerColA] =
          A[(innerRowA + offset) * K + innerColA];
    }
    for (int offset = 0; offset < BK; offset += strideB) {
      sB[(innerRowB + offset) * BN + innerColB] =
          B[(innerRowB + offset) * N + innerColB];
    }
    __syncthreads();

    A += BK;
    B += BK * N;

    for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
      for (int m = 0; m < TM; m++) {
        regM[m] = sA[(tRow * TM + m) * BK + dotIdx];
      }
      for (int n = 0; n < TN; n++) {
        regN[n] = sB[dotIdx * BN + tCol * TN + n];
      }

      for (int resM = 0; resM < TM; resM++) {
        for (int resN = 0; resN < TN; resN++) {
          sum[resM * TN + resN] += regM[resM] * regN[resN];
        }
      }
    }
    __syncthreads();
  }

  for (int resM = 0; resM < TM; resM++) {
    for (int resN = 0; resN < TN; resN++) {
      C[(tRow * TM + resM) * N + tCol * TN + resN] =
          alpha * sum[resM * TN + resN] +
          beta * C[(tRow * TM + resM) * N + tCol * TN + resN];
    }
  }
}

void launch_3_sram_2d_tiling(int M, int N, int K, float alpha, float* A,
                                    float* B, float beta, float* C) {
  const int BM = 64;
  const int BN = 64;
  const int BK = 8;
  const int TM = 8;
  const int TN = 8;
  dim3 block((BM * BN) / (TM * TN));
  dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
  gemm_3_sram_2d_tiling<BM, BN, BK, TM, TN>
      <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
