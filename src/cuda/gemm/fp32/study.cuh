#pragma once

#include <cuda_runtime.h>
#include <utils.h>

#include <cassert>

template <int BM, int BN, int BK, int WM, int WN, int WMITER, int WNITER,
          int WSUBM, int WSUBN, int TM, int TN>
__device__ void _mma(float* regA, float* regB, float* result, const float* sA,
                     const float* sB, const int warpRow, const int warpCol,
                     const int tRowIdxInWarp, const int tColIdxInWarp) {
  for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
    for (int it_wm = 0; it_wm < WMITER; it_wm++) {
      for (int it_tm = 0; it_tm < TM; it_tm++) {
        /* sA -> regA load */
        regA[it_wm * TM + it_tm] =
            sA[dotIdx * BM + warpRow * WM + it_wm * WSUBM + tRowIdxInWarp * TM +
               it_tm];
      }
    }

    for (int it_wn = 0; it_wn < WNITER; it_wn++) {
      for (int it_tn = 0; it_tn < TN; it_tn++) {
        regB[it_wn * TN + it_tn] =
            sB[dotIdx * BN + warpCol * WN + it_wn * WSUBN + tColIdxInWarp * TN +
               it_tn];
      }
    }
    /* sB -> regB load */

    for (int it_wm = 0; it_wm < WMITER; it_wm++) {
      for (int it_wn = 0; it_wn < WNITER; it_wn++) {
        for (int it_tm = 0; it_tm < TM; it_tm++) {
          for (int it_tn = 0; it_tn < TN; it_tn++) {
            int row_idx = it_wm * TM + it_tm;
            int col_idx = it_wn * TN + it_tn;
            int col_size = WNITER * TN;
            result[row_idx * col_size + col_idx] +=
                regA[row_idx] * regB[col_idx];
          }
        }
      }
    }
  }
}

template <int TILE_ROW, int TILE_COL, int STRIDE, bool TRANSPOSE>
__device__ void _load_global_store_shared(int globalRowStride,
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

template <int BM, int BN, int BK, int TM, int TN, int WM, int WN, int WMITER,
          int WNITER, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    gemm_warptiling(int M, int N, int K, float alpha, float* A, float* B,
                    float beta, float* C) {
  const int warpIdx = threadIdx.x / WARPSIZE;
  const int warpRow = warpIdx / (BN / WN);
  const int warpCol = warpIdx % (BN / WN);

  constexpr int WSUBM = WM / WMITER;
  constexpr int WSUBN = WN / WNITER;

  const int threadIdxInWarp = threadIdx.x % WARPSIZE;
  const int tRowIdxInWarp = threadIdxInWarp / (WSUBN / TN);
  const int tColIdxInWarp = threadIdxInWarp % (WSUBN / TN);

  const int innerRowA = threadIdx.x / (BK / 4);
  const int innerColA = threadIdx.x % (BK / 4);
  constexpr int rowStrideA = NUM_THREADS * 4 / BK;

  const int innerRowB = threadIdx.x / (BN / 4);
  const int innerColB = threadIdx.x % (BN / 4);
  constexpr int rowStrideB = NUM_THREADS * 4 / BN;

  __shared__ float sA[BM * BK];
  __shared__ float sB[BK * BN];

  float regA[WMITER * TM] = {
      0.0f,
  };

  float regB[WNITER * TN] = {
      0.0f,
  };

  float result[WMITER * WNITER * TM * TN] = {
      0.0f,
  };

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += (blockIdx.y * BM + warpRow * WM) * N + blockIdx.x * BN + warpCol * WN;

  for (int it_bk = 0; it_bk < K; it_bk += BK) {
    _load_global_store_shared<BM, BK, rowStrideA, true>(K, A, sA, innerRowA,
                                                        innerColA);
    _load_global_store_shared<BK, BN, rowStrideB, false>(N, B, sB, innerRowB,
                                                         innerColB);
    __syncthreads();

    A += BK;
    B += BK * N;

    _mma<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
        regA, regB, result, sA, sB, warpRow, warpCol, tRowIdxInWarp,
        tColIdxInWarp);
    __syncthreads();
  }

  for (int it_wm = 0; it_wm < WMITER; it_wm++) {
    for (int it_wn = 0; it_wn < WNITER; it_wn++) {
      // Calculate current position
      float* c_tmp = C + it_wm * WSUBM * N + it_wn * WSUBN;

      for (int it_tm = 0; it_tm < TM; it_tm++) {
        for (int it_tn = 0; it_tn < TN; it_tn += 4) {
          int reg_row_idx = it_wm * TM + it_tm;
          int reg_col_idx = it_wn * TN + it_tn;
          int reg_colsize = WNITER * TN;

          int gIdx =
              (tRowIdxInWarp * TM + it_tm) * N + tColIdxInWarp * TN + it_tn;
          float4 tmp = *reinterpret_cast<float4*>(&c_tmp[gIdx]);
          tmp.x = alpha * result[reg_row_idx * reg_colsize + reg_col_idx + 0] +
                  beta * tmp.x;
          tmp.y = alpha * result[reg_row_idx * reg_colsize + reg_col_idx + 1] +
                  beta * tmp.y;
          tmp.z = alpha * result[reg_row_idx * reg_colsize + reg_col_idx + 2] +
                  beta * tmp.z;
          tmp.w = alpha * result[reg_row_idx * reg_colsize + reg_col_idx + 3] +
                  beta * tmp.w;
          *reinterpret_cast<float4*>(&c_tmp[gIdx]) = tmp;
        }
      }
    }
  }
}

void launch_gemm_warptiling(int M, int N, int K, float alpha, float* A,
                            float* B, float beta, float* C) {
  const int BM = 64;
  const int BN = 128;
  const int BK = 16;

  assert(M % BM == 0 && "M must be divisible by BM");
  assert(N % BN == 0 && "N must be divisible by BN");
  assert(K % BK == 0 && "K must be divisible by BK");
  assert(N % 4 == 0 && "N must be divisible by 4 for float4 access");

  const int num_warps_M = 2;
  const int num_warps_N = 2;

  const int num_warps = num_warps_M * num_warps_N;
  const int num_threads = num_warps * WARPSIZE;

  const int TM = 4;
  const int TN = 4;

  const int WM = BM / num_warps_M;
  const int WN = BN / num_warps_N;

  const int WNITER = 2;
  const int WMITER = WM * WN / (WARPSIZE * TM * TN * WNITER);

  assert(((WM * WN) % (WARPSIZE * TM * TN * WNITER) == 0) &&
         "WNITER, WMITER must be set on this condition ((WM * WN) % (WARPSIZE "
         "* TM * TN * WNITER) == 0)");

  dim3 block(num_threads);
  dim3 grid(ceil_div(N, BN), ceil_div(M, BM));

  gemm_warptiling<BM, BN, BK, TM, TN, WM, WN, WMITER, WNITER, num_threads>
      <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}