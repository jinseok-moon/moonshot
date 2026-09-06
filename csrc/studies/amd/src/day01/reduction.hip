#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(call)                                                        \
  do {                                                                          \
    hipError_t status = (call);                                                \
    if (status != hipSuccess) {                                                \
      std::fprintf(stderr, "%s:%d HIP error: %s\n", __FILE__, __LINE__,       \
                   hipGetErrorString(status));                                \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (false)

template <int BlockSize>
__global__ void reduce_sum(const float* input, float* output, int n) {
  __shared__ float partial[BlockSize];
  int i = blockIdx.x * (BlockSize * 2) + threadIdx.x;
  float value = (i < n ? input[i] : 0.0f) +
                (i + BlockSize < n ? input[i + BlockSize] : 0.0f);
  partial[threadIdx.x] = value;
  __syncthreads();

  for (int stride = BlockSize / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) output[blockIdx.x] = partial[0];
}

int main() {
  constexpr int n = 1 << 20;
  constexpr int block_size = 256;
  constexpr int blocks = (n + block_size * 2 - 1) / (block_size * 2);
  std::vector<float> input(n, 1.0f), partial(blocks);
  float *device_input = nullptr, *device_partial = nullptr;

  HIP_CHECK(hipMalloc(&device_input, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&device_partial, blocks * sizeof(float)));
  HIP_CHECK(hipMemcpy(device_input, input.data(), n * sizeof(float), hipMemcpyHostToDevice));
  reduce_sum<block_size><<<blocks, block_size>>>(device_input, device_partial, n);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(partial.data(), device_partial, blocks * sizeof(float), hipMemcpyDeviceToHost));

  float total = 0.0f;
  for (float value : partial) total += value;
  std::printf("reduction: %.0f (expected %d)\n", total, n);
  HIP_CHECK(hipFree(device_input));
  HIP_CHECK(hipFree(device_partial));
  return total == n ? EXIT_SUCCESS : EXIT_FAILURE;
}
