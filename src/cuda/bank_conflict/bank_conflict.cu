#include <cuda_runtime.h>
#include <iostream>
#include <vector>
using namespace std;

// Ideal case
__global__ void kernel_0(int* d_vec, int* output) {
  __shared__ int shared_mem[32 * 32];
  shared_mem[threadIdx.x] = d_vec[threadIdx.x];

  __syncthreads();

  output[threadIdx.x] = shared_mem[threadIdx.x];
}

// Bank conflicts!
__global__ void kernel_1(int* d_vec, int* output) {
  __shared__ int shared_mem[32 * 32];

  shared_mem[threadIdx.x * 32] = d_vec[threadIdx.x];

  __syncthreads();

  output[threadIdx.x] = shared_mem[threadIdx.x * 32];
}

// Does not occur bank conflicts
__global__ void kernel_2(int* d_vec, int* output) {
  __shared__ int shared_mem[32 * 32];

  shared_mem[threadIdx.x * 32 + threadIdx.x] = d_vec[threadIdx.x];

  __syncthreads();

  output[threadIdx.x] = shared_mem[threadIdx.x * 32 + threadIdx.x];
}

// Does not occur bank conflicts
// But, because of uncoalesced memory access, dram data loading is not optimized.
__global__ void kernel_3(int* d_vec, int* output) {
  __shared__ int shared_mem[32 * 32];
  int row = threadIdx.x;

  shared_mem[row * 32 + threadIdx.x] = d_vec[row * 32 + threadIdx.x];

  __syncthreads();

  output[threadIdx.x] = shared_mem[row * 32 + threadIdx.x];
}

int main() {

  vector<int> vec(32 * 32);
  for (int i = 0; i < 32 * 32; i++) {
    vec[i] = i;
  }
  vector<int> output_vec(32);

  int* d_vec;
  cudaMalloc(&d_vec, vec.size() * sizeof(int));
  cudaMemcpy(d_vec, vec.data(), vec.size() * sizeof(int),
             cudaMemcpyHostToDevice);

  int* output;
  cudaMalloc(&output, 32 * sizeof(int));

  kernel_0<<<1, 32>>>(d_vec, output);
  cudaDeviceSynchronize();
  cudaMemcpy(output_vec.data(), output, output_vec.size() * sizeof(int),
             cudaMemcpyDeviceToHost);
  cout << "kernel_0" << endl;
  for (int i = 0; i < output_vec.size(); i++) {
    cout << output_vec[i] << " ";
  }
  cout << endl;

  kernel_1<<<1, 32>>>(d_vec, output);
  cudaDeviceSynchronize();
  cudaMemcpy(output_vec.data(), output, output_vec.size() * sizeof(int),
             cudaMemcpyDeviceToHost);
  cout << "kernel_1" << endl;
  for (int i = 0; i < output_vec.size(); i++) {
    cout << output_vec[i] << " ";
  }
  cout << endl;

  kernel_2<<<1, 32>>>(d_vec, output);
  cudaDeviceSynchronize();
  cudaMemcpy(output_vec.data(), output, output_vec.size() * sizeof(int),
             cudaMemcpyDeviceToHost);
  cout << "kernel_2" << endl;
  for (int i = 0; i < output_vec.size(); i++) {
    cout << output_vec[i] << " ";
  }
  cout << endl;

  kernel_3<<<1, 32>>>(d_vec, output);
  cudaDeviceSynchronize();
  cudaMemcpy(output_vec.data(), output, output_vec.size() * sizeof(int),
             cudaMemcpyDeviceToHost);
  cout << "kernel_3" << endl;
  for (int i = 0; i < output_vec.size(); i++) {
    cout << output_vec[i] << " ";
  }
  cout << endl;

  cudaFree(d_vec);
  cudaFree(output);
  return 0;
}