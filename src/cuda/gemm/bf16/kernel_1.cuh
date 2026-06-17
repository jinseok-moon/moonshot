#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <utils.h>

using namespace nvcuda;

__global__ void gemm_1_wmma(int M, int N, int K, float alpha, __nv_bfloat16* A,
                            __nv_bfloat16* B, float beta, float* C) {
  int warpM = blockIdx.y;
  int warpN = blockIdx.x;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major>
      b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;

  wmma::fill_fragment(acc, 0.0f);

  for (int bk = 0; bk < K; bk += 16) {
    wmma::load_matrix_sync(a_frag, A + warpM * 16 * K + bk, K);
    wmma::load_matrix_sync(b_frag, B + bk * N + warpN * 16, N);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
  }

  float* C_tile = C + warpM * 16 * N + warpN * 16;
  wmma::store_matrix_sync(C_tile, acc, N, wmma::mem_row_major);
}

void launch_1_bf16(int M, int N, int K, float alpha, __nv_bfloat16* A,
                   __nv_bfloat16* B, float beta, float* C) {
  dim3 block(WARPSIZE);  // 1 warp per block
  dim3 grid(ceil_div(N, 16), ceil_div(M, 16));
  gemm_1_wmma<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
