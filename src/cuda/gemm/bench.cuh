#pragma once
// GEMM benchmark harness (Phase 1, Week 1).
// - cudaEvent timing, warmup -> median over M runs (robust to outliers vs mean)
// - GFLOPs (2*M*N*K/t) and effective DRAM bandwidth (min-traffic model)
// - every kernel reported as % of a cuBLAS baseline
// - human-readable table + CSV export for plots
#include <cuda_runtime.h>
#include <utils.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace bench {

struct Shape {
  int M, N, K;
  std::string label;  // e.g. "square" / "skinny"
};

// Theoretical FLOP throughput: a GEMM is 2*M*N*K flops (mul + add).
inline double gflops(int M, int N, int K, float ms) {
  return (2.0 * M * N * K) / (static_cast<double>(ms) * 1e-3) / 1e9;
}

// Effective DRAM bandwidth assuming NO data reuse: read A + read B + write C.
// This is a lower bound on real traffic; useful as a sanity ceiling, not as a
// reuse-aware model. bytes_in/out default to bf16 inputs, fp32 output.
inline double bandwidth_gbs(int M, int N, int K, float ms, size_t bytes_in = 2,
                            size_t bytes_out = 4) {
  const double bytes = static_cast<double>(M) * K * bytes_in +
                       static_cast<double>(K) * N * bytes_in +
                       static_cast<double>(M) * N * bytes_out;
  return bytes / (static_cast<double>(ms) * 1e-3) / 1e9;
}

#ifdef DEBUG
inline constexpr int kWarmup = 0;
inline constexpr int kRuns = 1;
#else
inline constexpr int kWarmup = 10;
inline constexpr int kRuns = 50;
#endif

// cudaEvent timer: warmup, then return the MEDIAN of `runs` timings.
class Timer {
 public:
  Timer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }
  ~Timer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  float median_ms(const std::function<void()>& fn, int warmup = kWarmup,
                  int runs = kRuns) {
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times;
    times.reserve(runs);
    for (int i = 0; i < runs; ++i) {
      cudaEventRecord(start_, 0);
      fn();
      cudaEventRecord(stop_, 0);
      cudaEventSynchronize(stop_);
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, start_, stop_);
      times.push_back(ms);
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
  }

 private:
  cudaEvent_t start_, stop_;
};

struct Row {
  std::string kernel;
  std::string shape;
  int M, N, K;
  float ms;
  double gflops;
  double bandwidth;
  double pct_baseline;  // vs cuBLAS baseline (set by Reporter::add)
  float max_rel_err;
  bool passed;
};

class Reporter {
 public:
  // Per-shape baseline GFLOPs (the cuBLAS reference) used to compute %.
  void set_baseline(double baseline_gflops) { baseline_ = baseline_gflops; }

  void add(const std::string& kernel, const Shape& s, float ms,
           float max_rel_err, bool passed) {
    Row r;
    r.kernel = kernel;
    r.shape = s.label;
    r.M = s.M;
    r.N = s.N;
    r.K = s.K;
    r.ms = ms;
    r.gflops = gflops(s.M, s.N, s.K, ms);
    r.bandwidth = bandwidth_gbs(s.M, s.N, s.K, ms);
    r.pct_baseline = baseline_ > 0 ? r.gflops / baseline_ * 100.0 : 0.0;
    r.max_rel_err = max_rel_err;
    r.passed = passed;
    rows_.push_back(r);
  }

  void print_table() const {
    constexpr const char* BOLD = "\033[1m";
    constexpr const char* DIM = "\033[2m";
    constexpr const char* GREEN = "\033[32m";
    constexpr const char* RED = "\033[31m";
    constexpr const char* RESET = "\033[0m";

    std::cout << "\n"
              << BOLD << std::left << std::setw(26) << "kernel" << std::right
              << std::setw(18) << "shape" << std::setw(10) << "ms"
              << std::setw(11) << "GFLOP/s" << std::setw(10) << "GB/s"
              << std::setw(11) << "% cuBLAS" << std::setw(11) << "rel_err"
              << std::setw(8) << "ok" << RESET << "\n";
    std::cout << DIM << std::string(105, '-') << RESET << "\n";

    for (const auto& r : rows_) {
      char shp[32];
      std::snprintf(shp, sizeof(shp), "%dx%dx%d", r.M, r.N, r.K);
      std::cout << std::left << std::setw(26) << r.kernel << std::right
                << std::setw(18) << shp << std::setw(10) << std::fixed
                << std::setprecision(3) << r.ms << std::setw(11)
                << std::setprecision(1) << r.gflops << std::setw(10)
                << std::setprecision(1) << r.bandwidth << std::setw(10)
                << std::setprecision(1) << r.pct_baseline << "%"
                << std::setw(11) << std::scientific << std::setprecision(2)
                << r.max_rel_err << std::fixed << (r.passed ? GREEN : RED)
                << std::setw(8) << (r.passed ? "PASS" : "FAIL") << RESET
                << "\n";
    }
    std::cout << std::endl;
  }

  void write_csv(const std::string& path) const {
    std::ofstream f(path);
    if (!f) {
      std::cerr << "warning: could not open " << path << " for writing\n";
      return;
    }
    f << "kernel,shape,M,N,K,median_ms,gflops,bandwidth_gbs,pct_cublas,"
         "max_rel_err,passed\n";
    for (const auto& r : rows_) {
      f << r.kernel << ',' << r.shape << ',' << r.M << ',' << r.N << ',' << r.K
        << ',' << r.ms << ',' << r.gflops << ',' << r.bandwidth << ','
        << r.pct_baseline << ',' << r.max_rel_err << ',' << (r.passed ? 1 : 0)
        << '\n';
    }
    std::cout << "wrote " << rows_.size() << " rows to " << path << "\n";
  }

 private:
  std::vector<Row> rows_;
  double baseline_ = 0.0;
};

}  // namespace bench
