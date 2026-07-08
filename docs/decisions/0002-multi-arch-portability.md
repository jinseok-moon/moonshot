---
title: ADR 0002 — Multi-architecture portability
status: accepted
audience: [human, agent]
updated: 2026-07-08
tags: [adr, decision, portability, hardware]
related: [hardware-and-measurement.md, architecture.md]
---

# ADR 0002 — Multi-architecture portability

**Status:** accepted · **Date:** 2026-07-08

## Context

The project began framed as "an inference engine for the RTX 3090 / Ampere `sm_86`", with the
build hard-coded to that arch. The RTX 3090 remains the primary dev box, but the engine
should run and be developed against other GPUs (other NVIDIA arches; the engine itself on any
torch backend) without friction.

## Decision

Support other GPUs **without per-GPU branches in the engine**, via two mechanisms already
implied by the fallback design:

1. **Correctness is portable through fallbacks.** With no compiled kernels, every op in
   `moonshot.ops` is a torch fallback, so the engine runs a full forward pass on any backend
   torch supports (CUDA incl. ROCm, or CPU).
2. **Performance is opt-in per architecture via capability gating.** Each kernel declares the
   compute capability it needs; `moonshot/device.py` checks the running device and dispatch
   falls back to torch below it. A kernel built for Ampere does not break on Turing — it
   simply doesn't engage.

Supporting changes:

- Build no longer hard-codes `sm_86`. Torch `cpp_extension` auto-detects the current GPU;
  `TORCH_CUDA_ARCH_LIST` overrides for a portable fatbin (Turing…Hopper).
- Kernels guard arch-specific features with `#if __CUDA_ARCH__ >= 800` so one source compiles
  across the arch list.
- `scripts/inspect.sh` auto-detects arch (nvidia-smi), still overridable by argument.
- Docs reframed: RTX 3090 is the *primary dev box*, not the only target; a capability matrix
  (Volta→Hopper) documents what runs where; the `ncu` block is labelled a **container**
  constraint, not a GPU one.

## Consequences

- Benchmark numbers are arch-specific — a `% of reference` must state the arch it was
  measured on.
- Non-NVIDIA support is engine-only (fallbacks). Custom CUDA kernels stay NVIDIA-specific; a
  HIP/ROCm port of kernels is a possible later path, explicitly out of current scope.
- Pre-Ampere cards get correctness now and can get their own kernel rungs later (fp16-TC on
  Volta+, int8-TC on Turing+) without touching the engine — just new ops behind the gate.
