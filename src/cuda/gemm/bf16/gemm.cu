#include <cuda_bf16.h>
#include <utils.h>

#include <cassert>
#include <random>
#include <vector>

#include "../bench.cuh"
#include "../gemm_cublas.cuh"
#include "kernel_0.cuh"
#include "kernel_1.cuh"
#include "verify.hpp"

// Copy device result -> host, compare against fp32 reference, return whether it
// passes the bf16 tolerance. `out_err` receives the max relative error.
bool check(const float* ref, float* dev_C, float* host_C, int size,
           float threshold, float* out_err) {
  CUDA_CHECK(cudaMemcpy(host_C, dev_C, size * sizeof(float),
                        cudaMemcpyDeviceToHost));
  *out_err = moonshot::max_rel_error(ref, host_C, size);
  return *out_err <= threshold;
}

// Benchmark every bf16 kernel for one shape, append rows to the reporter.
void run_shape(const bench::Shape& s, bench::Timer& timer,
               bench::Reporter& reporter, cublasHandle_t handle) {
  const int M = s.M, N = s.N, K = s.K;
  // bf16 GEMM tolerance: result compared against an fp32 cuBLAS reference.
  constexpr float kTol = 2e-2f;

  std::cout << "=== shape " << s.label << " M=" << M << " N=" << N
            << " K=" << K << " ===\n";

  // Host data in fp32, plus a bf16 copy for the bf16 kernels.
  std::vector<float> host_A(static_cast<size_t>(M) * K);
  std::vector<float> host_B(static_cast<size_t>(K) * N);
  std::mt19937 gen(42);  // fixed seed: reproducible across runs/shapes
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  for (auto& x : host_A) x = dis(gen);
  for (auto& x : host_B) x = dis(gen);

  std::vector<nv_bfloat16> host_A_bf16(host_A.size());
  std::vector<nv_bfloat16> host_B_bf16(host_B.size());
  for (size_t i = 0; i < host_A.size(); ++i)
    host_A_bf16[i] = __float2bfloat16(host_A[i]);
  for (size_t i = 0; i < host_B.size(); ++i)
    host_B_bf16[i] = __float2bfloat16(host_B[i]);

  nv_bfloat16 *dev_A = nullptr, *dev_B = nullptr;
  float *dev_C = nullptr, *dev_A_f32 = nullptr, *dev_B_f32 = nullptr;
  CUDA_CHECK(cudaMalloc(&dev_A, host_A.size() * sizeof(nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&dev_B, host_B.size() * sizeof(nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&dev_C, static_cast<size_t>(M) * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_A_f32, host_A.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_B_f32, host_B.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dev_A, host_A_bf16.data(),
                        host_A.size() * sizeof(nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dev_B, host_B_bf16.data(),
                        host_B.size() * sizeof(nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dev_A_f32, host_A.data(), host_A.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dev_B_f32, host_B.data(), host_B.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  std::vector<float> ref_C(static_cast<size_t>(M) * N);
  std::vector<float> host_C(static_cast<size_t>(M) * N);
  const int size = M * N;

  // fp32 cuBLAS = ground-truth reference for correctness.
  launch_cublas<CUDA_R_32F, CUDA_R_32F>(M, N, K, 1.0f, dev_A_f32, dev_B_f32,
                                        0.0f, dev_C, handle);
  CUDA_CHECK(cudaMemcpy(ref_C.data(), dev_C, size * sizeof(float),
                        cudaMemcpyDeviceToHost));

  float err = 0.0f;
  bool ok = false;

  // cuBLAS bf16 = performance baseline (100%).
  auto cublas_bf16 = [&]() {
    launch_cublas<CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F>(M, N, K, 1.0f, dev_A,
                                                        dev_B, 0.0f, dev_C,
                                                        handle);
  };
  float base_ms = timer.median_ms(cublas_bf16);
  ok = check(ref_C.data(), dev_C, host_C.data(), size, kTol, &err);
  reporter.set_baseline(bench::gflops(M, N, K, base_ms));
  reporter.add("cuBLAS bf16 (baseline)", s, base_ms, err, ok);

  // kernel_0: naive bf16->fp32 cast GEMM.
  float ms0 = timer.median_ms(
      [&]() { launch_0_bf16(M, N, K, 1.0f, dev_A, dev_B, 0.0f, dev_C); });
  ok = check(ref_C.data(), dev_C, host_C.data(), size, kTol, &err);
  reporter.add("k0 naive cast", s, ms0, err, ok);

  // kernel_1: WMMA 16x16x16, 1 warp/block, no shared memory.
  float ms1 = timer.median_ms(
      [&]() { launch_1_bf16(M, N, K, 1.0f, dev_A, dev_B, 0.0f, dev_C); });
  ok = check(ref_C.data(), dev_C, host_C.data(), size, kTol, &err);
  reporter.add("k1 wmma 16x16x16", s, ms1, err, ok);

  cudaFree(dev_A);
  cudaFree(dev_B);
  cudaFree(dev_C);
  cudaFree(dev_A_f32);
  cudaFree(dev_B_f32);
}

int main(int argc, char* argv[]) {
  // Default: full sweep. Optional CLI override: `gemm_bf16 M N K`.
  std::vector<bench::Shape> shapes;
  if (argc >= 4) {
    int M = std::atoi(argv[1]), N = std::atoi(argv[2]), K = std::atoi(argv[3]);
    assert(M % 16 == 0 && N % 16 == 0 && K % 16 == 0 &&
           "M, N, K must be multiples of 16");
    shapes.push_back({M, N, K, "custom"});
  } else {
    shapes = {
        {1024, 1024, 1024, "square"},
        {2048, 2048, 2048, "square"},
        {4096, 4096, 4096, "square"},
        {4096, 4096, 1024, "skinny"},
    };
  }

  cublasHandle_t handle;
  cublasCreate(&handle);

  bench::Timer timer;
  bench::Reporter reporter;
  for (const auto& s : shapes) run_shape(s, timer, reporter, handle);

  reporter.print_table();
  reporter.write_csv("gemm_bf16_bench.csv");

  cublasDestroy(handle);
  return 0;
}
