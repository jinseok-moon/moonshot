---
title: ADR 0001 — Clean agentic restart around a pluggable engine
status: accepted
audience: [human, agent]
updated: 2026-07-07
tags: [adr, decision, architecture]
related: [architecture.md, roadmap.md, ai-collaboration.md]
---

# ADR 0001 — Clean agentic restart around a pluggable engine

**Status:** accepted · **Date:** 2026-07-07

## Context

The repository began as a CUDA fundamentals sandbox: fp32 and bf16 GEMM ladders, a
bank-conflict/pinned-memory/thread-indexing set of micro-topics, and a bf16 benchmark
harness. That work reached its purpose — deep understanding of a single problem (GEMM) with
measurement discipline. The next goal is different: an extensible **inference engine** that
gives kernels a real workload, presented as an AI-agentic project.

Carrying the old tree forward would (a) mix a learning sandbox's history with a portfolio
project and (b) shape the layout around microbenchmarks rather than a pluggable engine.

## Decision

1. **Restart clean.** New single-root-commit history organized around the engine, with the
   knowledge base (`docs/`), agent manual (`AGENTS.md`), and extension seam first.
2. **Remove the prior kernel code from the working tree.** Kernels re-enter later as custom
   ops through the seam — not migrated. The old work is preserved, not deleted: git tag
   `archive/fundamentals-v0`, branch `archive/pre-agentic-restructure`, and an out-of-repo
   `git bundle`. `main` is force-pushed to the clean root; archive refs are pushed alongside
   so remote history remains recoverable.
3. **Engine-runs-on-fallbacks design.** `moonshot.ops` dispatches each op to a CUDA custom
   op if built, else a pure-torch fallback, so the engine runs with zero custom kernels and
   each kernel replaces one fallback. See [architecture](architecture.md).
4. **Separate public engineering from private strategy.** Personal career strategy (target
   roles, résumé framing) lives in git-ignored `private/`; public `docs/` stays
   engineering-only.
5. **Codify AI collaboration.** Agents drive plumbing/harness/docs; humans own the why-fast
   reasoning, enforced by a whiteboard litmus. See [ai-collaboration](ai-collaboration.md).

## Consequences

- Old build (`CMakeLists` + microbench executables) is gone from `main`; new build is torch
  `cpp_extension` via `pyproject.toml`. Standalone (no-torch) kernel microbenching can be
  re-added under a separate build if needed.
- Phase 1 re-implements the bf16 GEMM ladder as custom ops rather than continuing the old
  files — accepted cost, in exchange for a coherent pluggable structure and clean history.
- Recovery path if anything is needed back: `git checkout archive/fundamentals-v0 -- <path>`
  or restore from the bundle.
