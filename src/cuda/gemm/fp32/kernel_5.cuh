#pragma once

#include <cuda_runtime.h>
#include <utils.h>

#include <cassert>

template <int TILE_ROW, int TILE_COL, int STRIDE, bool TRANSPOSE>
__device__ void load_global_store_shared(int globalRowStride,
                                         const float* global, float* shared,
                                         int tRowInTile, int tColVec4InTile) {
  for (int rowOffset = 0; rowOffset + STRIDE <= TILE_ROW; rowOffset += STRIDE) {
    const int gIdx =
        (tRowInTile + rowOffset) * globalRowStride + tColVec4InTile * 4;
    const float4 tmp = reinterpret_cast<const float4*>(global + gIdx)[0];

    if constexpr (TRANSPOSE) {
      shared[(tColVec4InTile * 4 + 0) * TILE_ROW + tRowInTile + rowOffset] =
          tmp.x;
      shared[(tColVec4InTile * 4 + 1) * TILE_ROW + tRowInTile + rowOffset] =
          tmp.y;
      shared[(tColVec4InTile * 4 + 2) * TILE_ROW + tRowInTile + rowOffset] =
          tmp.z;
      shared[(tColVec4InTile * 4 + 3) * TILE_ROW + tRowInTile + rowOffset] =
          tmp.w;
    } else {
      const int sIdx = (tRowInTile + rowOffset) * TILE_COL + tColVec4InTile * 4;
      reinterpret_cast<float4*>(shared + sIdx)[0] = tmp;
    }
  }
}

template <int BM, int BN, int BK, int WM, int WN, int WMITER, int WNITER,
          int WSUBM, int WSUBN, int TM, int TN>
__device__ void mma(float* regA, float* regB, float* tResult, const float* As,
                    const float* Bs, const int warpRow, const int warpCol,
                    const int threadRowIdxInWarp,
                    const int threadColIdxInWarp) {
  for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
    for (int it = 0; it < WMITER; it++) {
      for (int i = 0; i < TM; i++) {
        regA[it * TM + i] = As[(dotIdx * BM) + warpRow * WM + it * WSUBM +
                               threadRowIdxInWarp * TM + i];
      }
    }
    for (int it = 0; it < WNITER; it++) {
      for (int i = 0; i < TN; i++) {
        regB[it * TN + i] = Bs[(dotIdx * BN) + warpCol * WN + it * WSUBN +
                               threadColIdxInWarp * TN + i];
      }
    }

    for (int it_wm = 0; it_wm < WMITER; it_wm++) {
      for (int it_wn = 0; it_wn < WNITER; it_wn++) {
        for (int it_tm = 0; it_tm < TM; it_tm++) {
          for (int it_tn = 0; it_tn < TN; it_tn++) {
            int row_idx = (it_wm * TM + it_tm);
            int col_idx = it_wn * TN + it_tn;
            int col_size = (WNITER * TN);
            tResult[row_idx * col_size + col_idx] +=
                regA[it_wm * TM + it_tm] * regB[it_wn * TN + it_tn];
          }
        }
      }
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN, int WM, int WN, int WMITER,
          int WNITER, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    gemm_warp(int M, int N, int K, float alpha, float* A, float* B, float beta,
              float* C) {
  int cRow = blockIdx.y;
  int cCol = blockIdx.x;

  const int warpIdx = threadIdx.x / WARPSIZE;
  const int warpRow = warpIdx / (BN / WN);
  const int warpCol = warpIdx % (BN / WN);

  constexpr int WSUBM = WM / WMITER;
  constexpr int WSUBN = WN / WNITER;

  const int threadIdxInWarp = threadIdx.x % WARPSIZE;
  const int threadRowIdxInWarp = threadIdxInWarp / (WSUBN / TN);
  const int threadColIdxInWarp = threadIdxInWarp % (WSUBN / TN);

  A += cRow * BM * K;
  B += cCol * BN;
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const int innerRowA = threadIdx.x / (BK / 4);
  const int innerColA = threadIdx.x % (BK / 4);
  constexpr int rowStrideA = NUM_THREADS * 4 / BK;

  const int innerRowB = threadIdx.x / (BN / 4);
  const int innerColB = threadIdx.x % (BN / 4);
  constexpr int rowStrideB = NUM_THREADS * 4 / BN;

  float tResult[WMITER * WNITER * TM * TN] = {
      0.0f,
  };
  float regA[WMITER * TM] = {
      0.0f,
  };
  float regB[WNITER * TN] = {
      0.0f,
  };

  for (int bkIter = 0; bkIter < K; bkIter += BK) {
    load_global_store_shared<BM, BK, rowStrideA, true>(K, A, As, innerRowA,
                                                       innerColA);
    load_global_store_shared<BK, BN, rowStrideB, false>(N, B, Bs, innerRowB,
                                                        innerColB);

    __syncthreads();

    mma<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
        regA, regB, tResult, As, Bs, warpRow, warpCol, threadRowIdxInWarp,
        threadColIdxInWarp);

    A += BK;
    B += BK * N;

    __syncthreads();
  }

  for (int it_wm = 0; it_wm < WMITER; it_wm++) {
    for (int it_wn = 0; it_wn < WNITER; it_wn++) {
      float* c_tmp = C + it_wm * WSUBM * N + it_wn * WSUBN;
      for (int it_tm = 0; it_tm < TM; it_tm++) {
        // loop unrolling for TN
        for (int it_tn = 0; it_tn < TN; it_tn += 4) {
          int reg_row_idx = (it_wm * TM + it_tm);
          int reg_col_idx = it_wn * TN + it_tn;
          int reg_col_size = (WNITER * TN);

          int idx = (threadRowIdxInWarp * TM + it_tm) * N +
                    threadColIdxInWarp * TN + it_tn;
          float4 tmp = *reinterpret_cast<float4*>(&c_tmp[idx]);
          tmp.x =
              alpha * tResult[reg_row_idx * reg_col_size + reg_col_idx + 0] +
              beta * tmp.x;
          tmp.y =
              alpha * tResult[reg_row_idx * reg_col_size + reg_col_idx + 1] +
              beta * tmp.y;
          tmp.z =
              alpha * tResult[reg_row_idx * reg_col_size + reg_col_idx + 2] +
              beta * tmp.z;
          tmp.w =
              alpha * tResult[reg_row_idx * reg_col_size + reg_col_idx + 3] +
              beta * tmp.w;

          *reinterpret_cast<float4*>(&c_tmp[idx]) = tmp;
        }
      }
    }
  }
}

void launch_gemm_5_warptiling(int M, int N, int K, float alpha, float* A,
                              float* B, float beta, float* C) {
  const int BM = 64;
  const int BN = 128;
  const int BK = 16;

  const int numWarps_M = 2;
  const int numWarps_N = 2;
  const int numWarps = numWarps_M * numWarps_N;
  const int numThreads = numWarps * WARPSIZE;

  const int TM = 4;
  const int TN = 4;

  const int WM = BM / numWarps_M;
  const int WN = BN / numWarps_N;

  const int WNITER = 2;
  const int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);

  // Current kernel does not have edge guards, so dimensions must be
  // tile-aligned.
  assert(M % BM == 0 && "M must be divisible by BM");
  assert(N % BN == 0 && "N must be divisible by BN");
  assert(K % BK == 0 && "K must be divisible by BK");
  // float4 vectorized load/store requires 4-float alignment on the N stride.
  assert(N % 4 == 0 && "N must be divisible by 4 for float4 access");
  assert((WM * WN) % (WARPSIZE * TM * TN * WNITER) == 0);

  dim3 block(numThreads);
  dim3 grid(ceil_div(N, BN), ceil_div(M, BM));

  gemm_warp<BM, BN, BK, TM, TN, WM, WN, WMITER, WNITER, numThreads>
      <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
