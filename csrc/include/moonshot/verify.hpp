// Correctness check against a reference (bf16 tolerance: max rel err < 2e-2).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace moonshot::verify {

// Maximum relative error between `got` and `ref` over `n` elements. Denominator is
// max(|ref|, eps) so near-zero references don't blow up the ratio.
template <typename T>
double max_rel_error(const T* got, const T* ref, std::size_t n, double eps = 1e-6) {
  double worst = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    double g = static_cast<double>(got[i]);
    double r = static_cast<double>(ref[i]);
    double rel = std::abs(g - r) / std::max(std::abs(r), eps);
    worst = std::max(worst, rel);
  }
  return worst;
}

}  // namespace moonshot::verify
