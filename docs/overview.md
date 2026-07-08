---
title: Overview
status: living
audience: [human, agent]
updated: 2026-07-07
tags: [overview, north-star]
related: [roadmap.md, architecture.md, ai-collaboration.md]
---

# Overview

## What moonshot is

A from-scratch LLM inference engine, built kernel-up and architecture-portable (developed
primarily on an RTX 3090 / Ampere `sm_86`). It is deliberately structured so the **engine
runs before any custom kernel exists** — a real Llama forward pass on pure-PyTorch fallback
ops, on any GPU torch supports — and each CUDA kernel you write replaces one fallback, scored
as a percentage of it. Kernels are capability-gated, so one built for Ampere falls back
gracefully on older cards rather than breaking. The kernels (bf16 tensor-core GEMM,
FlashAttention, weight-only quantized matmul) are the substance; the engine is the harness
that gives every kernel a real workload to live in.

## Why it's shaped this way

Three forces shape the design:

1. **Motivation follows a running system.** A GEMM that beats cuBLAS by X% in a microbench
   is abstract. The *same* GEMM making a real model emit tokens faster is concrete. The
   engine-as-spine keeps every kernel attached to something that runs.

2. **Depth is the moat, and it's AI-resistant.** In performance engineering, the hard part
   is not writing a correct kernel — it's the last 3× and the *why*: occupancy, roofline,
   memory-latency behaviour, bank conflicts, SASS. That reasoning is where AI assistance is
   weakest, so it's exactly what a human here owns and must defend at a whiteboard. See
   [ai-collaboration](ai-collaboration.md).

3. **Extensibility over completeness.** Kernels land one at a time through a fixed seam
   ([architecture](architecture.md)), never a rewrite. The engine tolerates a half-finished
   kernel set because unimplemented ops fall back to torch.

## The two-track goal

The work targets two adjacent careers with a large shared core:

- **GPU inference serving** — the CUDA kernels, FlashAttention, quantization, and
  serving-system fluency map directly here.
- **Accelerator / compiler engineering** — the same tiling and dataflow reasoning, plus a
  later compiler-flavoured track (Triton, MLIR-adjacent exploration), maps here.

The common core (kernels + quantization + measurement discipline) is built first; the
divergent specialization comes late, steered by which opportunities are live. The detailed
personal strategy lives outside this public repo.

## North star

> Ship an inference engine whose every speedup is explained down to the SASS and defensible
> without an AI in the room — while using AI agents to move fast on everything that *isn't*
> that irreducible core.

Next: [roadmap](roadmap.md).
