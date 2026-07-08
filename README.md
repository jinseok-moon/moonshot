# moonshot

**A from-scratch LLM inference engine, built kernel-up — architecture-portable, in the open, as an AI-agentic project.**

`moonshot` is an inference engine designed around one idea: **the engine runs before any
custom kernel exists, and every kernel you write transparently replaces a fallback.**
Bring up a real Llama forward pass on pure-PyTorch ops, then swap in your own CUDA kernels
one at a time — each benched against the exact fallback it replaces. FlashAttention, a bf16
tensor-core GEMM ladder, weight-only quantized matmuls: they plug into a fixed seam, not a
rewrite.

Two things are true about this repo at once:

1. **The engineering moat is depth.** The value is being able to say *why* a kernel is fast
   — occupancy, roofline, memory-latency behaviour on `sm_86` — and defend it at a
   whiteboard. That reasoning is owned by a human, on purpose. See
   [`docs/ai-collaboration.md`](docs/ai-collaboration.md).
2. **The development is agent-orchestrated.** Plumbing, harnesses, docs, and
   large-codebase navigation are driven by coding agents (Claude Code + Codex) against a
   shared [`AGENTS.md`](AGENTS.md). Breadth shipped is the evidence of leverage; the agent
   never owns the performance reasoning.

## The extension seam

```
Python  (orchestration)     weights · tokenizer · KV cache · sampling · batching · serving loop
   │
   ▼
moonshot.ops                rmsnorm · linear · attention · dequant_linear · ...
   │  dispatch
   ├──▶ torch.ops.moonshot.*   registered CUDA custom op   (if built)
   └──▶ pure-torch fallback    (always works — engine runs with zero custom kernels)
```

Every op checks for a registered CUDA custom op and uses it, else falls back to a
reference PyTorch implementation. Adding a kernel = dropping a new op under `csrc/ops/` and
registering it; the engine picks it up with no orchestration changes. Full design:
[`docs/architecture.md`](docs/architecture.md).

## Measurement discipline

- Every kernel is reported as **% of the fallback / a production reference**, never in isolation.
- `ncu` is unavailable on the target box (`ERR_NVGPUCTRPERM`); profiling uses **CUDA
  events + `ptxas -v` + SASS + `nsys`** instead. Methodology:
  [`docs/hardware-and-measurement.md`](docs/hardware-and-measurement.md).
- A kernel is "done" only when its number *and* its why-fast explanation both exist.

## Repo layout

```
README.md                     you are here
AGENTS.md                     operating manual for coding agents (Claude + Codex)
CLAUDE.md                     Claude Code entrypoint → imports AGENTS.md
pyproject.toml                the `moonshot` python package + build entry
docs/                         AI + human readable knowledge base (open-knowledge format)
  llms.txt                    machine-readable index of the knowledge base
  overview.md · roadmap.md · architecture.md · hardware-and-measurement.md
  ai-collaboration.md · decisions/ · kernels/
moonshot/                     python orchestration package
  ops.py                      the extension seam: custom-op dispatch + torch fallbacks
  engine.py                   orchestration loop (skeleton)
csrc/                         CUDA C++ custom ops — kernels plug in here
  include/moonshot/           shared headers (check, bench timer, verify)
  ops/template/               copy-me op showing the TORCH_LIBRARY registration seam
scripts/                      inspect.sh (ptxas -v + SASS), tooling
```

## Status

Fresh start. The scaffold, the extension seam, and the knowledge base exist; kernels are
the roadmap ([`docs/roadmap.md`](docs/roadmap.md)). The engine is designed to run on torch
fallbacks first, then earn its speed one custom kernel at a time.

> Prior fundamentals work (fp32 + bf16 GEMM ladders, bench harness) lives in the git tag
> `archive/fundamentals-v0`; this repo is the clean rebuild around a pluggable engine.

## Hardware support

Portable by construction: the engine runs on any backend torch supports (CUDA incl. ROCm, or
CPU) via fallbacks, and hand-written kernels are **capability-gated** — each declares the
compute capability it needs and falls back on devices below it, so a kernel built for Ampere
doesn't break on Turing, it just doesn't engage. Developed and profiled primarily on an
**RTX 3090** (Ampere, `sm_86`); Turing → Hopper covered by the build. Support matrix and the
no-`ncu` measurement methodology: [`docs/hardware-and-measurement.md`](docs/hardware-and-measurement.md).

## Where to start reading

New here (human or agent)? → [`docs/overview.md`](docs/overview.md), then
[`docs/roadmap.md`](docs/roadmap.md). Agents working in the repo → [`AGENTS.md`](AGENTS.md).
