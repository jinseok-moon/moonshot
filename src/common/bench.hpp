#pragma once

#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace hetero {

inline int ceil_div(int value, int divisor) {
  return (value + divisor - 1) / divisor;
}

#ifdef DEBUG
inline constexpr int kDefaultWarmupRuns = 0;
inline constexpr int kDefaultBenchmarkRuns = 1;
#else
inline constexpr int kDefaultWarmupRuns = 10;
inline constexpr int kDefaultBenchmarkRuns = 20;
#endif

// Timer strategy expected interface:
//   void start();
//   void stop();          // must synchronize so elapsed_ms() is meaningful
//   float elapsed_ms() const;
//
// Each backend provides a concrete Timer (cudaEvent / hipEvent / MTLCommandBuffer
// GPU timestamps) and exposes e.g. `using LatencyProfiler = hetero::Profiler<CudaEventTimer>;`.
template <typename Timer>
class Profiler {
 public:
  float time_function(std::function<void()> kernel_func) {
    timer_.start();
    kernel_func();
    timer_.stop();
    return timer_.elapsed_ms();
  }

  float benchmark_kernel(const std::string& name,
                         std::function<void()> kernel_func,
                         std::function<bool()> validate_func = {},
                         int warmup_runs = kDefaultWarmupRuns,
                         int benchmark_runs = kDefaultBenchmarkRuns) {
    for (int i = 0; i < warmup_runs; ++i) {
      kernel_func();
    }

    std::vector<float> times;
    times.reserve(benchmark_runs);
    for (int i = 0; i < benchmark_runs; ++i) {
      times.push_back(time_function(kernel_func));
    }
    const float avg_time =
        std::accumulate(times.begin(), times.end(), 0.0f) / benchmark_runs;

    constexpr const char* CYAN = "\033[36m";
    constexpr const char* BOLD = "\033[1m";
    constexpr const char* GREEN = "\033[32m";
    constexpr const char* RED = "\033[31m";
    constexpr const char* DIM = "\033[2m";
    constexpr const char* RESET = "\033[0m";

    const bool has_validation = static_cast<bool>(validate_func);
    const bool validation_ok = !has_validation || validate_func();

    std::cout << CYAN << "[BENCHMARK] " << RESET << BOLD
              << (validation_ok ? GREEN : RED) << std::right << std::setw(40)
              << name << RESET << " │ " << (validation_ok ? GREEN : RED)
              << std::fixed << std::setprecision(6) << avg_time << " ms"
              << RESET << DIM << " (w:" << warmup_runs
              << " r:" << benchmark_runs << ")" << RESET;

    if (has_validation) {
      std::cout << BOLD << (validation_ok ? GREEN : RED)
                << (validation_ok ? " [PASSED]" : " [FAILED]") << RESET;
    }
    std::cout << std::endl;
    return avg_time;
  }

 private:
  Timer timer_;
};

}  // namespace hetero
