---
title: Hardware and measurement
status: living
audience: [human, agent]
updated: 2026-07-07
tags: [hardware, profiling, measurement, sm_86]
related: [architecture.md, roadmap.md, ai-collaboration.md]
---

# Hardware and measurement

## Target box

| | |
| --- | --- |
| GPU | RTX 3090 (Ampere, `sm_86`, 24 GB) |
| Tensor cores | bf16, int8 (`mma.sync.s8`) |
| Async copy | `cp.async` (`LDGSTS`) available |
| Not available | fp8, Hopper `wgmma`, TMA |
| Peak DRAM BW | ~936 GB/s (the bandwidth-bound target) |
| Toolchain | CUDA 13.0, torch `cpp_extension`, `nsys`, `ptxas -v`, `cuobjdump -sass` |

fp8 / Hopper-only experiments (FA3, fp8 GEMM) are out of scope on this box; if ever needed,
rent an H100 separately.

## The `ncu` constraint

`ncu` (Nsight Compute) does **not** work on the target container: `ERR_NVGPUCTRPERM`. GPU
hardware performance-counter access is gated by a host module parameter
(`NVreg_RestrictProfilingToAdminUsers`) that an unprivileged container cannot unlock. `nsys`
(tracing/CUPTI) works fine.

So the methodology is deliberately **counter-free** — the same constraint Simon Boehm's
SGEMM write-up works under. We quantify with:

- **CUDA events** — median latency over `kRuns` after `kWarmup` warmups, with
  `cudaDeviceSynchronize` placed exactly.
- **`ptxas -v`** — registers / shared memory / spills per kernel.
- **`cuobjdump -sass`** — confirm the instructions that matter actually emit (`HMMA` for
  tensor-core matmul, `LDGSTS` for `cp.async`).
- **`nsys`** — timeline, occupancy, kernel/memcpy overlap.
- **`cudaOccupancyMaxActiveBlocksPerMultiprocessor`** — theoretical occupancy, computed in-code.

If a hardware counter is genuinely required, cross-check on a box where profiling is
permitted; do not build workflows that assume `ncu`.

## Reporting discipline

- **Always relative.** Every kernel is reported as **% of a reference**, never absolute
  GFLOP/s alone. References, in order of use: its own torch fallback → cuBLAS/cuDNN/SDPA.
- **Derived metrics:** GFLOP/s = `2·M·N·K / t`; effective bandwidth = minimal DRAM traffic
  (read A + read B + write C, no reuse) / t.
- **Correctness gate:** compare against the fallback / an fp32 reference; bf16 tolerance
  `max rel err < 2e-2`.
- **Shape sweeps**, not single points: square `{1024, 2048, 4096}³` plus a skinny shape;
  for attention, seq-len sweeps with prefill vs decode separated.

## Definition of done

A kernel is finished only when **both** exist:

1. a number, as % of its reference, across a shape sweep; and
2. a why-fast explanation — grounded in GFLOP/s / bandwidth + SASS/occupancy + an `nsys`
   timeline — that a human can defend at a whiteboard without an AI in the room.

Recorded per kernel in `docs/kernels/<name>.md`. The number without the explanation is not a
result. See [ai-collaboration](ai-collaboration.md).
