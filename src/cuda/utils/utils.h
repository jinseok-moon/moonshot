#pragma once
#include <cuda_runtime.h>

#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <source_location>
#include <vector>

constexpr int WARPSIZE = 32;

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    std::source_location location = std::source_location::current();          \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
      std::cerr << "file: " << location.file_name() << '(' << location.line() \
                << ':' << location.column() << ") `"                          \
                << location.function_name()                                   \
                << "`: " << cudaGetErrorString(err) << std::endl;             \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

inline int ceil_div(int value, int divisor) {
  return (value + divisor - 1) / divisor;
}

class LatencyProfiler {
 public:
  LatencyProfiler() {
    cudaEventCreate(&evt_start);
    cudaEventCreate(&evt_end);
  }
  ~LatencyProfiler() {
    cudaEventDestroy(evt_start);
    cudaEventDestroy(evt_end);
  }

#ifdef DEBUG
#define WARMUP_RUNS 0
#define BENCHMARK_RUNS 1
#else
#define WARMUP_RUNS 10
#define BENCHMARK_RUNS 20
#endif
  // Function to perform warmup and benchmark runs.
  // If validate_func is provided, benchmark output includes pass/fail state.
  float benchmark_kernel(const std::string& name,
                         std::function<void()> kernel_func,
                         std::function<bool()> validate_func = {},
                         int warmup_runs = WARMUP_RUNS,
                         int benchmark_runs = BENCHMARK_RUNS) {
    // Warmup runs
    for (int i = 0; i < warmup_runs; ++i) {
      kernel_func();
    }

    // Benchmark runs
    std::vector<float> times;
    for (int i = 0; i < benchmark_runs; ++i) {
      float time = time_function(kernel_func);
      times.push_back(time);
    }

    // Calculate average time
    float avg_time =
        std::accumulate(times.begin(), times.end(), 0.0f) / benchmark_runs;

    // ANSI color codes
    const char* CYAN = "\033[36m";
    const char* BOLD = "\033[1m";
    const char* GREEN = "\033[32m";
    const char* RED = "\033[31m";
    const char* DIM = "\033[2m";
    const char* RESET = "\033[0m";

    const bool has_validation = static_cast<bool>(validate_func);
    const bool validation_ok = !has_validation || validate_func();

    std::cout << CYAN << "[BENCHMARK] " << RESET << BOLD
              << (validation_ok ? GREEN : RED) << std::right << std::setw(40)
              << name << RESET << " │ " << (validation_ok ? GREEN : RED)
              << std::fixed << std::setprecision(6) << avg_time << " ms"
              << RESET << DIM << " (w:" << warmup_runs << " r:" << benchmark_runs
              << ")" << RESET;

    if (has_validation) {
      std::cout << BOLD << (validation_ok ? GREEN : RED)
                << (validation_ok ? " [PASSED]" : " [FAILED]") << RESET;
    }
    std::cout << std::endl;

    return avg_time;
  }

  // Timing wrapper function using std::function
  float time_function(std::function<void()> kernel_func) {
    cudaEventRecord(evt_start, 0);
    // Execute the function and capture the result (if any)
    kernel_func();
    cudaEventRecord(evt_end, 0);
    cudaEventSynchronize(evt_end);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, evt_start, evt_end);
    return elapsed_time;
  }

 private:
  cudaEvent_t evt_start, evt_end;
};
