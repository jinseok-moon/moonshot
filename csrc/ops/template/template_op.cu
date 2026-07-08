// Template custom op — copy this directory to csrc/ops/<name>/ and replace the kernel.
//
// It shows the whole seam with a trivial elementwise op (out = x * alpha + beta):
//   1. a CUDA kernel,
//   2. a launcher returning a torch::Tensor,
//   3. TORCH_LIBRARY registration so it becomes torch.ops.moonshot.template_affine,
//      which moonshot/ops.py can dispatch to.
//
// Build via torch cpp_extension (pip install -e .); architecture is auto-detected or set
// with TORCH_CUDA_ARCH_LIST. This template is plain fp32 and runs on any arch. For a kernel
// that uses arch-specific features, guard them so one source compiles across the arch list:
//     #if __CUDA_ARCH__ >= 800      // bf16 tensor cores + cp.async (Ampere+)
//       ... fast path ...
//     #else
//       ... portable path (or leave the runtime capability gate to pick the fallback) ...
//     #endif
// and pass the matching min_capability in moonshot/ops.py.

#include <torch/extension.h>

#include <cuda_runtime.h>

#include "moonshot/check.cuh"

namespace {

__global__ void affine_kernel(const float* __restrict__ x, float* __restrict__ out,
                              float alpha, float beta, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = x[i] * alpha + beta;
}

torch::Tensor template_affine(torch::Tensor x, double alpha, double beta) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "template op expects float32");
  x = x.contiguous();
  auto out = torch::empty_like(x);

  const int64_t n = x.numel();
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  affine_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(),
                                     static_cast<float>(alpha), static_cast<float>(beta), n);
  CUDA_CHECK_LAST();
  return out;
}

}  // namespace

// Registers torch.ops.moonshot.template_affine
TORCH_LIBRARY_FRAGMENT(moonshot, m) {
  m.def("template_affine(Tensor x, float alpha, float beta) -> Tensor");
}
TORCH_LIBRARY_IMPL(moonshot, CUDA, m) {
  m.impl("template_affine", TORCH_FN(template_affine));
}
