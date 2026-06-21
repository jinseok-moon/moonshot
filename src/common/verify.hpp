#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>

namespace moonshot {

inline constexpr float kDefaultEps = 1e-5f;

// Largest element-wise relative error between result and ref. Host buffers only.
inline float max_rel_error(const float* ref, const float* result, int size,
                           float eps = kDefaultEps) {
  float worst = 0.0f;
  for (int i = 0; i < size; ++i) {
    const float diff = std::abs(result[i] - ref[i]);
    const float relative_error = diff / (std::abs(ref[i]) + eps);
    worst = std::max(worst, relative_error);
  }
  return worst;
}

// Host-side element-wise comparison. Pure: operates on host buffers only.
// Backends wrap this with a device->host copy before calling.
inline bool check_close(const float* ref, const float* result, int size,
                        float rel_threshold = 0.01f,
                        float eps = kDefaultEps,
                        bool print_error = false) {
  for (int i = 0; i < size; ++i) {
    const float diff = std::abs(result[i] - ref[i]);
    const float ref_abs = std::abs(ref[i]);
    const float relative_error = diff / (ref_abs + eps);
    if (relative_error > rel_threshold) {
      if (print_error) {
        std::cout << "result[" << i << "] = " << result[i] << " != ref[" << i
                  << "] = " << ref[i]
                  << " (relative error: " << relative_error * 100 << "%)"
                  << std::endl;
      }
      return false;
    }
  }
  return true;
}

}  // namespace moonshot
