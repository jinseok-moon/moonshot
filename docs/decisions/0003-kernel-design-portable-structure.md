---
title: ADR 0003 — Kernel design: portable structure, specialized atoms, tuned by config
status: accepted
audience: [human, agent]
updated: 2026-07-08
tags: [adr, decision, kernels, portability, autotune]
related: [architecture.md, hardware-and-measurement.md, 0002-multi-arch-portability.md]
---

# ADR 0003 — Kernel design: portable structure, specialized atoms, tuned by config

**Status:** accepted · **Date:** 2026-07-08

## Context

CUDA kernels have architecture-specific optima. The naive way to chase them is a separate
hand-tuned kernel per arch (bespoke tile shapes, raw `ldmatrix`/`mma.sync` SASS, `wgmma`/TMA
on Hopper). That yields peak numbers but N brittle codepaths — untenable for a solo developer
maintaining GEMM + attention + quant across arches — and it is exactly the "last 20%" this
project already considers low-transfer (see the roadmap's `kernel_4` cap).

The question: rather than the absolute highest per-arch kernel, write kernels **generally**?

## Decision

"General vs highest" is the wrong axis. Split every kernel into two layers and treat them
differently:

- **Algorithm / dataflow layer — arch-independent, written once.** Tiling strategy, the
  shared-memory hierarchy plan, double-buffered pipelining, the online-softmax recurrence.
  This is ~80% of the performance and 100% of the transferable understanding (the
  whiteboard-defensible *why-fast*, and the reasoning that carries to the compiler/NPU track).

- **Instruction / atom layer — arch-specific, isolated behind a thin abstraction.** The MMA
  and copy primitives (`mma.sync` vs `wgmma`, `cp.async` vs TMA) and SM-resource-tuned tile
  shapes. Only this layer diverges by arch.

Concretely:

1. **Write the dataflow generally.** One structure, not a per-arch fork.
2. **Template the tile parameters** (BM, BN, BK, warp tiles, pipeline stages) and **autotune
   per arch.** The per-arch difference is *config*, not *code* — this captures most of the
   per-arch win with a single codepath, and is realized by the agent-driven autotuner
   artifact (see [ai-collaboration](ai-collaboration.md)).
3. **Abstract the MMA/copy atom** behind a small interface; guard arch-specific instructions
   with `#if __CUDA_ARCH__ >= 800`. Ampere-family (sm_80/86/89) is one `mma.sync` path. A
   Hopper `wgmma`/TMA atom is a **documented seam, out of current scope** — added only if
   Hopper becomes a real target (it is a genuine step-change at the atom level).
4. **General ≠ portable-but-slow.** The general structure still uses tensor cores, shared
   memory, and `cp.async`; it parameterizes them instead of hand-scheduling. A kernel that
   drops tensor cores to be "portable" is a toy, not a general kernel.
5. Runtime dispatch keeps using the capability gate from
   [ADR 0002](0002-multi-arch-portability.md): kernel when supported, else torch fallback.

**Principle:** *Portable structure, specialized atoms, tuned by config.*

## Consequences

- One codepath per op for the Ampere family; per-arch performance comes from autotuned tile
  configs, not code forks.
- Target ~80–90% of the reference on the primary arch with clean parameterized code, rather
  than a brittle hand-tuned 95%. Report the arch each number was measured on.
- The *why-fast* explanation lives in the transferable dataflow layer — good for both the
  serving and compiler career tracks; the "separate schedule/config from computation" habit
  is the compiler/NPU mindset.
- Hopper `wgmma`/TMA and pre-Ampere (fp16-TC Volta, int8-TC Turing) atoms are future rungs
  behind the same seam, not rewrites.
- Ladder discipline is unchanged: each optimization step is a new `kernel_N` file.
