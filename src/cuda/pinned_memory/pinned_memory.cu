#include <cuda_fp16.h>
#include <utils.h>

int main() {
  size_t size = 1024 * 1024 * 1024;

  half* host_ptr_pinned;
  half* host_ptr_pageable;
  half* dev_ptr;

  CUDA_CHECK(cudaMallocHost((void**)&host_ptr_pinned, size));
  host_ptr_pageable = (half*)malloc(size);
  CUDA_CHECK(cudaMalloc((void**)&dev_ptr, size));

  cudaEvent_t evt_start, evt_end;
  CUDA_CHECK(cudaEventCreate(&evt_start));
  CUDA_CHECK(cudaEventCreate(&evt_end));

  // warm up stage
  for (int i = 0; i < 3; i++) {
    CUDA_CHECK(
        cudaMemcpy(dev_ptr, host_ptr_pinned, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(dev_ptr, host_ptr_pageable, size, cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // measure stage
  CUDA_CHECK(cudaEventRecord(evt_start));
  for (int i = 0; i < 10; i++) {
    CUDA_CHECK(
        cudaMemcpy(dev_ptr, host_ptr_pinned, size, cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaEventRecord(evt_end));
  CUDA_CHECK(cudaEventSynchronize(evt_end));

  float dt_ms;
  CUDA_CHECK(cudaEventElapsedTime(&dt_ms, evt_start, evt_end));

  float avg_time = dt_ms / 10.0f;
  float bandwidth_gb_s = (size * 10) / (dt_ms * 1e-3) / (1024 * 1024 * 1024);

  std::cout << "Pinned memory" << std::endl;
  std::cout << "Total time: " << dt_ms << " ms" << std::endl;
  std::cout << "Average time per copy: " << avg_time << " ms" << std::endl;
  std::cout << "Data size: " << size / (1024 * 1024 * 1024) << " GB"
            << std::endl;
  std::cout << "Bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;

  CUDA_CHECK(cudaEventRecord(evt_start));
  for (int i = 0; i < 10; i++) {
    CUDA_CHECK(
        cudaMemcpy(dev_ptr, host_ptr_pageable, size, cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaEventRecord(evt_end));
  CUDA_CHECK(cudaEventSynchronize(evt_end));

  std::cout << std::endl;
  std::cout << "Pageable memory" << std::endl;
  CUDA_CHECK(cudaEventElapsedTime(&dt_ms, evt_start, evt_end));

  avg_time = dt_ms / 10.0f;
  bandwidth_gb_s = (size * 10) / (dt_ms * 1e-3) / (1024 * 1024 * 1024);

  std::cout << "Total time: " << dt_ms << " ms" << std::endl;
  std::cout << "Average time per copy: " << avg_time << " ms" << std::endl;
  std::cout << "Data size: " << size / (1024 * 1024 * 1024) << " GB"
            << std::endl;
  std::cout << "Bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;

  CUDA_CHECK(cudaEventDestroy(evt_start));
  CUDA_CHECK(cudaEventDestroy(evt_end));
  CUDA_CHECK(cudaFreeHost(host_ptr_pinned));
  free(host_ptr_pageable);
  CUDA_CHECK(cudaFree(dev_ptr));

  return 0;
}