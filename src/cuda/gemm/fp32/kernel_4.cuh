#pragma once
#include <cuda_runtime.h>

#include <cassert>

template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_4_vectorized_sram_2d_tiling(int M, int N, int K,
                                               float alpha, float *A,
                                               float *B, float beta,
                                               float *C)
{
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

  int innerRowA = threadIdx.x / (BK / 4);
  int innerColA = threadIdx.x % (BK / 4);

  int innerRowB = threadIdx.x / (BN / 4);
  int innerColB = threadIdx.x % (BN / 4);

  float regM[TM] = {
      0.0f,
  };
  float regN[TN] = {
      0.0f,
  };

  float sum[TM * TN] = {
      0.0f,
  };

  int strideA = blockDim.x / (BK / 4);
  int strideB = blockDim.x / (BN / 4);

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK)
  {
    for (int offset = 0; offset < BM; offset += strideA)
    {
      float4 tmp = reinterpret_cast<const float4 *>(
          &A[(innerRowA + offset) * K + innerColA * 4])[0];
      // transpose A during the GMEM to SMEM transfer
      sA[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
      sA[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
      sA[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
      sA[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }
    for (int offset = 0; offset < BK; offset += strideB)
    {
      reinterpret_cast<float4 *>(&sB[(innerRowB + offset) * BN + innerColB * 4])
          [0] = reinterpret_cast<float4 *>(
              &B[(innerRowB + offset) * N + innerColB * 4])[0];
    }
    __syncthreads();

    A += BK;
    B += BK * N;

    for (int dotIdx = 0; dotIdx < BK; dotIdx++)
    {
      for (int m = 0; m < TM; m++)
      {
        regM[m] = sA[dotIdx * BM + tRow *TM + m];
      }
      for (int n = 0; n < TN; n++)
      {
        regN[n] = sB[dotIdx * BN + tCol * TN + n];
      }

      for (int resM = 0; resM < TM; resM++)
      {
        for (int resN = 0; resN < TN; resN++)
        {
          sum[resM * TN + resN] += regM[resM] * regN[resN];
        }
      }
    }
    __syncthreads();
  }

  for (int resM = 0; resM < TM; resM++)
  {
    for (int resN = 0; resN < TN; resN += 4)
    {
      int idx = (tRow * TM + resM) * N + tCol * TN + resN;
      float4 tmp = reinterpret_cast<float4 *>(&C[idx])[0];
      tmp.x = alpha * sum[resM * TN + resN + 0] + beta * tmp.x;
      tmp.y = alpha * sum[resM * TN + resN + 1] + beta * tmp.y;
      tmp.z = alpha * sum[resM * TN + resN + 2] + beta * tmp.z;
      tmp.w = alpha * sum[resM * TN + resN + 3] + beta * tmp.w;
      reinterpret_cast<float4 *>(&C[idx])[0] = tmp;
    }
  }
}

void launch_4_vectorized_sram_2d_tiling(int M, int N, int K, float alpha,
                                         float *A, float *B, float beta,
                                         float *C)
{
  const int BM = 64;
  const int BN = 128;
  const int BK = 16;
  const int TM = 16;
  const int TN = 4;

  dim3 block((BM * BN) / (TM * TN));
  dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
  gemm_4_vectorized_sram_2d_tiling<BM, BN, BK, TM, TN>
      <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
