---
title: Supported hardware and measurement
status: living
audience: [human, agent]
updated: 2026-07-08
tags: [hardware, portability, profiling, measurement]
related: [architecture.md, roadmap.md, ai-collaboration.md]
---

# Supported hardware and measurement

## Portability model

moonshot is **not tied to one GPU.** Support comes in two layers:

1. **The engine runs anywhere torch runs** — any CUDA device (including ROCm) or CPU. With
   no compiled kernels, every op is a torch fallback, so a fresh checkout runs a full
   forward pass on whatever hardware is present.
2. **Hand-written kernels are capability-gated.** Each kernel declares the compute
   capability it needs; at runtime [`moonshot/device.py`](../moonshot/device.py) falls back
   to torch on devices below it. A kernel built for Ampere simply doesn't engage on Turing —
   it doesn't break, it falls back.

So "support other GPUs" needs no per-GPU branches in the engine: correctness is portable via
fallbacks, and performance is opt-in per architecture.

## Primary dev box

Developed and profiled primarily on an **RTX 3090 (Ampere, `sm_86`, 24 GB)** — bf16/int8
tensor cores, `cp.async`, ~936 GB/s DRAM. No fp8, no Hopper `wgmma`/TMA. This is the
reference for baselines, not a hard requirement.

## Capability matrix (NVIDIA)

| Arch | CC | fp16 TC | int8 TC | bf16 TC | `cp.async` | fp8 TC | Kernel status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Volta | 7.0 | ✅ | — | — | — | — | fallbacks; fp16-TC kernels |
| Turing | 7.5 | ✅ | ✅ | — | — | — | + int8 kernels |
| Ampere | 8.0/8.6 | ✅ | ✅ | ✅ | ✅ | — | **primary target** (bf16 ladder, FlashAttn) |
| Ada | 8.9 | ✅ | ✅ | ✅ | ✅ | ✅ | Ampere kernels run; fp8 later |
| Hopper | 9.0 | ✅ | ✅ | ✅ | ✅ | ✅ | runs; `wgmma`/TMA path out of scope here |

Non-NVIDIA: the engine runs on ROCm/CPU via fallbacks. Custom CUDA kernels are
NVIDIA-specific; a HIP port is a possible later path, not a current goal.

## Build for multiple architectures

Kernels compile via torch `cpp_extension`. Architecture is auto-detected, or set explicitly
for a portable fatbin:

```bash
TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0" pip install -e .   # Turing … Hopper
```

Arch-specific features inside a kernel are guarded with `#if __CUDA_ARCH__ >= 800` so one
source compiles across the list; the runtime gate then picks kernel vs fallback.

## The `ncu` constraint (environment, not GPU)

On the current dev container `ncu` (Nsight Compute) fails with `ERR_NVGPUCTRPERM`: hardware
performance-counter access is gated by a host module parameter
(`NVreg_RestrictProfilingToAdminUsers`) an unprivileged container cannot unlock. This is a
**container** limitation, not a GPU one — on a box where profiling is permitted, use `ncu`
freely. Either way the methodology below is counter-free and portable.

We quantify with:

- **CUDA events** — median latency over `kRuns` after `kWarmup` warmups, with
  `cudaDeviceSynchronize` placed exactly ([`csrc/include/moonshot/bench.cuh`](../csrc/include/moonshot/bench.cuh)).
- **`ptxas -v`** — registers / shared memory / spills per kernel.
- **`cuobjdump -sass`** — confirm the instructions that matter emit (`HMMA` bf16/fp16 mma,
  `IMMA` int8 mma, `LDGSTS` `cp.async`).
- **`nsys`** — timeline, occupancy, kernel/memcpy overlap.
- **`cudaOccupancyMaxActiveBlocksPerMultiprocessor`** — theoretical occupancy, in-code.

## Reporting discipline

- **Always relative.** Every kernel is reported as **% of a reference**, never absolute
  GFLOP/s alone. References, in order: its own torch fallback → cuBLAS/cuDNN/SDPA. Report the
  arch the number was measured on (a % is arch-specific).
- **Derived metrics:** GFLOP/s = `2·M·N·K / t`; effective bandwidth = minimal DRAM traffic
  (read A + read B + write C, no reuse) / t.
- **Correctness gate:** compare against the fallback / an fp32 reference; bf16 tolerance
  `max rel err < 2e-2`.
- **Shape sweeps**, not single points: square `{1024, 2048, 4096}³` plus a skinny shape; for
  attention, seq-len sweeps with prefill vs decode separated.

## Definition of done

A kernel is finished only when **both** exist:

1. a number, as % of its reference, across a shape sweep (with the arch stated); and
2. a why-fast explanation — GFLOP/s / bandwidth + SASS/occupancy + an `nsys` timeline — that
   a human can defend at a whiteboard without an AI in the room.

Recorded per kernel in `docs/kernels/<name>.md`. The number without the explanation is not a
result. See [ai-collaboration](ai-collaboration.md).
