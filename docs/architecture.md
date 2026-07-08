---
title: Architecture — the extension seam
status: living
audience: [human, agent]
updated: 2026-07-07
tags: [architecture, engine, custom-ops]
related: [overview.md, roadmap.md, hardware-and-measurement.md]
---

# Architecture — the extension seam

The whole engine is organized around one seam so that kernels are **plugins, not rewrites**.

```
Python  (orchestration)     weights · tokenizer · KV cache · sampling · batching · serving loop
   │
   ▼
moonshot.ops                rmsnorm · linear · attention · dequant_linear · ...
   │  dispatch (per op)
   ├──▶ torch.ops.moonshot.<name>   registered CUDA custom op   (used if the ext is built)
   └──▶ pure-torch fallback          reference impl (always available)
```

## The rule

`moonshot/ops.py` exposes one function per engine operation. Each function asks the registry
whether a compiled CUDA custom op named `<name>` is present **and** the current device meets
the compute capability that kernel needs ([`device.py`](../moonshot/device.py)):

- **present and supported** → call `torch.ops.moonshot.<name>` (the hand-written kernel).
- **absent, or device below the kernel's capability** → run the pure-torch fallback (a
  correct, unoptimized reference).

The capability gate is what makes the engine portable across GPUs: a kernel written for
Ampere (`cc ≥ 8.0`, bf16 tensor cores + `cp.async`) simply falls back on a Turing card
instead of breaking. See [hardware-and-measurement](hardware-and-measurement.md#portability-model).

Consequences:

- **The engine runs with zero custom kernels.** On a fresh checkout with nothing compiled,
  every op is a torch fallback and a real Llama forward pass works end-to-end. This is what
  makes bring-up cheap and keeps motivation attached to a running model.
- **Every kernel has a built-in baseline.** The fallback it replaces *is* its correctness
  oracle and its first speed reference (`% of fallback`), before comparing to cuBLAS/SDPA.
- **Swapping is transparent.** Finishing a kernel changes performance, not orchestration.
  The decode layer never knows whether it called a kernel or a fallback.

## Layers

| Layer | Lives in | Owns |
| --- | --- | --- |
| Orchestration | `moonshot/engine.py` (Python) | weight load, tokenizer, KV cache, sampling, batching, serving loop |
| Op seam | `moonshot/ops.py` (Python) | dispatch: custom op vs fallback, per operation |
| Kernels | `csrc/ops/<name>/` (CUDA C++) | the hot path: RMSNorm, GEMM, FlashAttention, dequant-GEMM |
| Shared C++ | `csrc/include/moonshot/` | check macros, CUDA-event bench timer, verify helpers |

The boring, latency-insensitive plumbing stays in Python so CUDA effort concentrates on the
decode hot path. This is the gpt-fast-style split: Python orchestrates, C++ does the matmuls
and attention.

## Adding a kernel

1. Copy `csrc/ops/template/` → `csrc/ops/<name>/`. Implement the kernel; register it with
   `TORCH_LIBRARY(moonshot, m) { m.def(...); m.impl(...); }` (see the template README).
2. Ensure `moonshot/ops.py` has a `<name>` dispatcher (custom-op-preferred, torch fallback),
   passing the `min_capability` the kernel needs (e.g. `(8, 0)` for bf16 tensor cores). Guard
   arch-specific instructions in the kernel with `#if __CUDA_ARCH__ >= 800`.
3. Rebuild (`pip install -e .` on the 3090), bench against the fallback, and record the
   number + why-fast note in `docs/kernels/<name>.md`.

Ladder discipline: an optimization step is a **new** `kernel_N` file, never an in-place edit
of a prior rung. The ladder of rungs is the story a reader (and an interviewer) follows.

## Kernel authoring — portable structure, specialized atoms, tuned by config

Kernels are not hand-tuned per arch. Every kernel splits into two layers, treated
differently (full rationale in [ADR 0003](decisions/0003-kernel-design-portable-structure.md)):

- **Dataflow layer — arch-independent, written once.** Tiling, the shared-memory plan,
  double-buffered pipelining, online-softmax recurrence. ~80% of the performance and 100% of
  the transferable *why-fast*. This is where the depth goes.
- **Atom layer — arch-specific, isolated.** The MMA/copy primitive and SM-tuned tile shapes.
  Only this diverges by arch.

Rules:

1. **Write the dataflow generally** — one structure, not a per-arch fork.
2. **Template tile parameters** (BM, BN, BK, warp tiles, stages) and **autotune per arch**.
   The per-arch difference is *config*, not *code*.
3. **Abstract the MMA/copy atom** behind a thin interface; guard arch-specific instructions
   with `#if __CUDA_ARCH__ >= 800`. Ampere-family (sm_80/86/89) is one `mma.sync` path; a
   Hopper `wgmma`/TMA atom is a documented seam, added only if Hopper becomes a target.
4. **General ≠ slow-portable.** Still use tensor cores, shared memory, `cp.async` — parameterize
   them, don't drop them. A kernel that abandons tensor cores to be portable is a toy.
5. Pass the kernel's `min_capability` in `moonshot/ops.py`; the runtime gate picks kernel vs
   fallback ([hardware-and-measurement](hardware-and-measurement.md#portability-model)).

**Principle:** *Portable structure, specialized atoms, tuned by config.*

## Progression toward a fused decode path

Ops start individually dispatched (easy to bench and swap). Once the important kernels
exist, a stretch goal is to fuse the per-layer sequence (RMSNorm → QKV GEMM → attention → O
GEMM → RMSNorm → MLP) into a single C++ decode path exposed as one custom op — the "fully
custom engine" milestone. The seam stays the same; only the granularity of the plugin
changes.

## Build

CUDA custom ops compile via torch `cpp_extension` (`pip install -e .`) on the target box
(`sm_86`). Until a kernel lands the package is pure Python and imports without a compiler.
CUDA does not build on macOS — see [hardware-and-measurement](hardware-and-measurement.md).
