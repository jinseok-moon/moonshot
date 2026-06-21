# bench/ — tensor-core GEMM benchmark harness (Phase 1, Week 1)

Measurement discipline for the bf16 GEMM ladder, **without `ncu`** (blocked in this
container — `ERR_NVGPUCTRPERM`). Everything below uses cudaEvents, `ptxas -v`, SASS,
and `nsys`.

## What it measures
- **median** latency over `kRuns` runs after `kWarmup` warmups (robust to outliers)
- **GFLOP/s** = `2·M·N·K / t`
- **effective bandwidth** = min DRAM traffic (read A + read B + write C, no reuse) / t
- **% of cuBLAS bf16** — the baseline every kernel is scored against
- **max relative error** vs an fp32 cuBLAS reference (bf16 tolerance `2e-2`)

Harness code: [`src/cuda/gemm/bench.cuh`](../src/cuda/gemm/bench.cuh) (Timer + Reporter),
driver: [`src/cuda/gemm/bf16/gemm.cu`](../src/cuda/gemm/bf16/gemm.cu).

## Run
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target gemm_bf16 -j
./build/src/cuda/gemm/gemm_bf16            # full sweep -> table + gemm_bf16_bench.csv
./build/src/cuda/gemm/gemm_bf16 4096 4096 4096   # single shape
```
Shape sweep: `{1024,2048,4096}³` square + `4096×4096×1024` skinny.

## Static inspection (reg/smem/spill + SASS)
```bash
bench/inspect.sh                          # ptxas -v + HMMA SASS for bf16/gemm.cu
```

## nsys timeline
```bash
nsys profile -o /tmp/gemm --stats=true ./build/src/cuda/gemm/gemm_bf16 4096 4096 4096
```

## Baseline (RTX 3090, sm_86, CUDA 12.6) — k0/k1 starting point

| kernel | 4096³ GFLOP/s | % cuBLAS bf16 | regs (sm_86) |
|---|---|---|---|
| cuBLAS bf16 (baseline) | ~67600 | 100% | — |
| k1 wmma 16×16×16 (1 warp/block, no smem) | ~11000 | **16%** | 40 |
| k0 naive cast | ~1080 | 1.6% | 27 |

cuBLAS peaks ~71 TFLOP/s on the skinny shape. k1 is the climb's starting rung — it
leaves the tensor cores starved on global-memory latency (no shared-mem reuse, no
pipelining). Next: **k2 shared-memory tiling** to lift reuse, then k3 warp tiling and
k4 `cp.async` double-buffering where most of the gap closes.

> Numbers are medians; rebuild and re-run to refresh the CSV. Goal for Phase 1:
> k6/k7 ≥ 80% of cuBLAS bf16 @ 4096³.
