#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/system/cuda/memory.h>
#include <utils.h>

static LatencyProfiler profiler;

using namespace std;

__global__ void kernel_0(int* d_vec_a, int* d_vec_b, int* output, int size) {
  int index = threadIdx.x * 32 + threadIdx.y;
  for (int i = index; i < size; i += 32 * 32) {
    output[i] = d_vec_a[i] + d_vec_b[i];
  }
}

__global__ void kernel_1(int* d_vec_a, int* d_vec_b, int* output, int size) {
  int index = threadIdx.y * 32 + threadIdx.x;
  for (int i = index; i < size; i += 32 * 32) {
    output[i] = d_vec_a[i] + d_vec_b[i];
  }
}

void launch_gpu_kernel_0(int *d_vec_a, int *d_vec_b, int *output, int size) {
  dim3 block(32, 32, 1);
  dim3 grid(1, 1);
  kernel_0<<<grid, block>>>(d_vec_a, d_vec_b, output, size);
}

void launch_gpu_kernel_1(int *d_vec_a, int *d_vec_b, int *output, int size) {
  dim3 block(32, 32, 1);
  dim3 grid(1, 1);
  kernel_1<<<grid, block>>>(d_vec_a, d_vec_b, output, size);
}

int main() {
  int size =1;
  thrust::device_vector<int> d_vec_a(32 * 32*size);
  thrust::device_vector<int> d_vec_b(32 * 32*size);
  thrust::host_vector<int> h_vec_a(32 * 32*size);
  thrust::host_vector<int> h_vec_b(32 * 32*size);
  thrust::device_vector<int> output_vec(32 * 32*size);
  
  thrust::copy(h_vec_a.begin(), h_vec_a.end(), d_vec_a.begin());
  thrust::copy(h_vec_b.begin(), h_vec_b.end(), d_vec_b.begin());

  profiler.benchmark_kernel("kernel_0", [&]() {
    launch_gpu_kernel_0(thrust::raw_pointer_cast(d_vec_a.data()),
                        thrust::raw_pointer_cast(d_vec_b.data()),
                        thrust::raw_pointer_cast(output_vec.data()),
                        32 * 32 * size);
  });

  profiler.benchmark_kernel("kernel_1", [&]() {
    launch_gpu_kernel_1(thrust::raw_pointer_cast(d_vec_a.data()),
                        thrust::raw_pointer_cast(d_vec_b.data()),
                        thrust::raw_pointer_cast(output_vec.data()),
                        32 * 32 * size);
  });
  return 0;
}