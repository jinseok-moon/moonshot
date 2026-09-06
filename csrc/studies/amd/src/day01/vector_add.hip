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

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  constexpr int n = 1 << 20;
  const size_t bytes = n * sizeof(float);
  std::vector<float> host_a(n, 1.0f), host_b(n, 2.0f), host_c(n);
  float *a = nullptr, *b = nullptr, *c = nullptr;

  HIP_CHECK(hipMalloc(&a, bytes));
  HIP_CHECK(hipMalloc(&b, bytes));
  HIP_CHECK(hipMalloc(&c, bytes));
  HIP_CHECK(hipMemcpy(a, host_a.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(b, host_b.data(), bytes, hipMemcpyHostToDevice));

  vector_add<<<(n + 255) / 256, 256>>>(a, b, c, n);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(host_c.data(), c, bytes, hipMemcpyDeviceToHost));

  for (float value : host_c) {
    if (value != 3.0f) return EXIT_FAILURE;
  }
  std::puts("vector add: PASS");
  HIP_CHECK(hipFree(a));
  HIP_CHECK(hipFree(b));
  HIP_CHECK(hipFree(c));
}
