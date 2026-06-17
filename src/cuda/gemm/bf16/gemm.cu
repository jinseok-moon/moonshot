#include <cuda_bf16.h>
#include <utils.h>

#include <cassert>
#include <random>

#include "../gemm_cublas.cuh"
#include "kernel_0.cuh"
#include "kernel_1.cuh"

static LatencyProfiler profiler;

constexpr float EPS = 1e-5;

bool copy_and_check_result(float* ref, float* dev_result, float* host_result,
                           int size, float threshold = 0.05,
                           bool print_error = false) {
  memset(host_result, 0, size * sizeof(float));
  cudaMemcpy(host_result, dev_result, size * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemset(dev_result, 0, size * sizeof(float));

  for (int i = 0; i < size; i++) {
    float diff = abs(host_result[i] - ref[i]);
    float ref_abs = abs(ref[i]);

    float relative_error = diff / (ref_abs + EPS);
    if (relative_error > threshold) {
      if (print_error) {
        std::cout << "result[" << i << "] = " << host_result[i] << " != ref["
                  << i << "] = " << ref[i]
                  << " (relative error: " << relative_error * 100 << "%)"
                  << std::endl;
      }
      return false;
    }
  }
  return true;
}

int main(int argc, char* argv[]) {
  int M = 1024;
  int N = 1024;
  int K = 1024;

  if (argc >= 2) M = std::atoi(argv[1]);
  if (argc >= 3) N = std::atoi(argv[2]);
  if (argc >= 4) K = std::atoi(argv[3]);

  assert(M % 16 == 0 && "M must be a multiple of 16");
  assert(N % 16 == 0 && "N must be a multiple of 16");
  assert(K % 16 == 0 && "K must be a multiple of 16");

  std::cout << "BF16 GEMM - Matrix dimensions: M=" << M << ", N=" << N
            << ", K=" << K << std::endl;

  // Host: generate in float, convert to bf16 for device
  float* host_A = (float*)malloc(M * K * sizeof(float));
  float* host_B = (float*)malloc(K * N * sizeof(float));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);

  for (int i = 0; i < M * K; i++) host_A[i] = dis(gen);
  for (int i = 0; i < K * N; i++) host_B[i] = dis(gen);

  // Convert to bf16 on host
  nv_bfloat16* host_A_bf16 = (nv_bfloat16*)malloc(M * K * sizeof(nv_bfloat16));
  nv_bfloat16* host_B_bf16 = (nv_bfloat16*)malloc(K * N * sizeof(nv_bfloat16));
  for (int i = 0; i < M * K; i++) host_A_bf16[i] = __float2bfloat16(host_A[i]);
  for (int i = 0; i < K * N; i++) host_B_bf16[i] = __float2bfloat16(host_B[i]);

  // Device: bf16 inputs, float output
  nv_bfloat16* dev_A = nullptr;
  nv_bfloat16* dev_B = nullptr;
  float* dev_C = nullptr;
  cudaMalloc((void**)&dev_A, M * K * sizeof(nv_bfloat16));
  cudaMalloc((void**)&dev_B, K * N * sizeof(nv_bfloat16));
  cudaMalloc((void**)&dev_C, M * N * sizeof(float));

  cudaMemcpy(dev_A, host_A_bf16, M * K * sizeof(nv_bfloat16),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, host_B_bf16, K * N * sizeof(nv_bfloat16),
             cudaMemcpyHostToDevice);

  // fp32 copies for cuBLAS fp32 reference
  float* dev_A_f32 = nullptr;
  float* dev_B_f32 = nullptr;
  cudaMalloc((void**)&dev_A_f32, M * K * sizeof(float));
  cudaMalloc((void**)&dev_B_f32, K * N * sizeof(float));
  cudaMemcpy(dev_A_f32, host_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B_f32, host_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

  float* ref_C = (float*)malloc(M * N * sizeof(float));
  float* host_C = (float*)malloc(M * N * sizeof(float));

  cublasHandle_t handle;
  cublasCreate(&handle);

  // cuBLAS fp32 reference
  profiler.benchmark_kernel("CUBLAS FP32 REF", [&]() {
    launch_cublas<CUDA_R_32F, CUDA_R_32F>(M, N, K, 1.0f, dev_A_f32, dev_B_f32,
                                          0.0f, dev_C, handle);
  });
  cudaMemcpy(ref_C, dev_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemset(dev_C, 0, M * N * sizeof(float));

  // cuBLAS bf16 input, fp32 output, fp32 compute
  profiler.benchmark_kernel(
      "CUBLAS BF16 GEMM",
      [&]() {
        launch_cublas<CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F>(
            M, N, K, 1.0f, dev_A, dev_B, 0.0f, dev_C, handle);
      },
      [&]() { return copy_and_check_result(ref_C, dev_C, host_C, M * N); });
  CUDA_CHECK(cudaDeviceSynchronize());

  // -- Kernels --
  profiler.benchmark_kernel(
      "GEMM 0 DRAM COALESCING",
      [&]() { launch_0_bf16(M, N, K, 1.0f, dev_A, dev_B, 0.0f, dev_C); },
      [&]() { return copy_and_check_result(ref_C, dev_C, host_C, M * N); });
  CUDA_CHECK(cudaDeviceSynchronize());

  // -- WMMA 16x16x16 --
  profiler.benchmark_kernel(
      "GEMM 1 WMMA 16x16x16",
      [&]() { launch_1_bf16(M, N, K, 1.0f, dev_A, dev_B, 0.0f, dev_C); },
      [&]() { return copy_and_check_result(ref_C, dev_C, host_C, M * N); });
  CUDA_CHECK(cudaDeviceSynchronize());

  // Cleanup
  cublasDestroy(handle);
  free(host_A);
  free(host_B);
  free(host_A_bf16);
  free(host_B_bf16);
  free(ref_C);
  free(host_C);
  cudaFree(dev_A);
  cudaFree(dev_B);
  cudaFree(dev_C);
  cudaFree(dev_A_f32);
  cudaFree(dev_B_f32);

  return 0;
}
