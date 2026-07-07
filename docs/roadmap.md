---
title: Roadmap
status: living
audience: [human, agent]
updated: 2026-07-07
tags: [roadmap, milestones]
related: [overview.md, architecture.md, hardware-and-measurement.md]
---

# Roadmap

Engineering phases. Principle for every phase: **problem → benchmark → profile → write-up**.
Each step leaves (1) code in the repo, (2) a measured number as % of a reference, (3) a
short knowledge note. No studying without an artifact.

The engine ([architecture](architecture.md)) is not a phase — it is the **spine** every
kernel plugs into. It comes up early on torch fallbacks; later phases harden its decode path
one custom op at a time.

## Phase 1 — bf16 tensor-core GEMM ladder + profiling discipline

Re-climb the GEMM optimization ladder for tensor cores. The point of interest vs an fp32
ladder: the bottleneck moves from compute to **memory latency**, and we quantify it without
`ncu`.

- `kernel_0` naive cast → `kernel_1` WMMA → `kernel_2` shared-memory tiling → `kernel_3` warp
  tiling → `kernel_4` `cp.async` double-buffered pipelining.
- **Cap at `kernel_4`.** Goal is to reach ~60–70% of cuBLAS bf16 and *explain* why it is
  memory-latency bound via `nsys`/SASS. The last rungs (`ldmatrix` + raw `mma.sync` SASS
  hand-tuning) are demoted to stretch: high effort, NVIDIA-specific, low transfer.
- Artifact: the ladder as custom ops + a roofline write-up.

## Phase 2 — inference primitives

Memory-bound building blocks; target is near peak bandwidth (~936 GB/s on the 3090).

- warp reduce (`__shfl_down_sync`) → block reduce → vectorized loads.
- softmax: naive 3-pass → online/2-pass (the FlashAttention build-up).
- normalization: RMSNorm (+ fused residual), LayerNorm.

## Phase 3 — engine spine v0 (Python + CUDA C++ decode layer)

Stand up the engine on fallbacks, then start replacing.

- Python: weight load, tokenizer, KV cache, sampling, serving loop; static → continuous
  batching.
- CUDA C++: decode hot path, kernels dispatched through the op seam.
- Bring up a real Llama forward pass end-to-end, measure it, then swap in Phase 1/2 kernels.

## Phase 4 — FlashAttention (keystone)

- Derive online-softmax attention; implement naive attention as the correctness oracle.
- Fused `QKᵀ → online softmax → ·V` single kernel (bf16 in, fp32 accumulate), `cp.async`
  double-buffer, tensor-core matmuls, causal mask, head_dim ∈ {64, 128}.
- Decode path (seq_len_q = 1) + KV-cache + GQA/MQA.
- Done: correctness < 2e-2 vs SDPA; speed ≥ 60% of the SDPA flash backend.

## Phase 5 — quantized GEMM

Highest-leverage shared bet across the two-track goal.

- int8 tensor-core GEMM (`mma.sync.s8`) with per-tensor/per-channel scales.
- **W4A16** weight-only: 4-bit packing + group-wise scale/zero-point, fused
  unpack → dequant → bf16 GEMM. Prove the memory-traffic saving in bandwidth numbers.

## Phase 6 — Triton + PyTorch integration

- Reimplement primitives (softmax/RMSNorm) and a matmul in Triton; `@triton.autotune`.
- Formalize the custom-op / autograd / `torch.compile` integration the engine already uses.
- Compare hand-tuned CUDA vs autotuned Triton, same problem.

## Phase 7 — specialization spike

Steered late by which opportunities are live (detail is personal, kept out of this repo):

- **Serving track:** contribute to a production engine (e.g. vLLM) now that building one
  makes its internals legible.
- **Compiler track:** MLIR-adjacent exploration (Triton internals, a small compiler/IR
  toy), computer-architecture review, and reading how accelerator compilers lower ops.

## Cross-cutting

- **Open-source sprint** runs in parallel from Phase 4: small doc/bug/perf PRs first, then
  kernel contributions.
- **AI-agentic practice** ([ai-collaboration](ai-collaboration.md)) applies throughout:
  agents drive plumbing/harness/docs; the why-fast reasoning stays human-owned. A possible
  capstone artifact: an agent-driven kernel autotuner that sweeps configs and summarizes why
  a winner is fast — demonstrating kernel depth and agent orchestration at once.

## Milestones

| Horizon | Focus | Artifact |
| --- | --- | --- |
| Near | Phase 1–2 | tensor-core GEMM ladder + primitives, both as custom ops with baselines |
| Mid | Phase 3–4 | engine running a real model; FlashAttention ≥ 60% of SDPA + write-up |
| Later | Phase 5–6 | W4A16 quant kernel; Triton/PyTorch integration; first merged OSS PR |
| Horizon | Phase 7 | specialization spike + portfolio packaging |
