---
title: AI collaboration — the delegate-vs-own doctrine
status: living
audience: [human, agent]
updated: 2026-07-07
tags: [ai-agents, practice, doctrine]
related: [overview.md, architecture.md, hardware-and-measurement.md]
---

# AI collaboration — the delegate-vs-own doctrine

This project is built with coding agents (Claude Code + Codex), and treats *how* AI is used
as an explicit engineering practice, not an afterthought. The governing idea:

> Use agents to move fast on everything **except** the irreducible performance core, which a
> human owns and can defend at a whiteboard without an AI in the room.

In performance engineering this is not a compromise — it's the natural division. A correct
kernel is easy for an agent to produce; the last 3× and the *why* (occupancy, roofline,
memory-latency, bank conflicts, SASS) are exactly where AI is least reliable. That is the
moat, so that is what a human owns.

## Division of labour

**Agents own (drive freely):**

- Python plumbing — weight loading, tokenizer glue, sampling, serving loop, KV-cache
  bookkeeping, the op-registry seam.
- Benchmark harnesses, CSV/plot generation, build config, CI.
- Docs — keeping this knowledge base in sync, writing ADRs, cross-linking.
- Large-codebase navigation and summarization (e.g. locating a hook inside vLLM).
- Test/verification scaffolds and *first-draft* kernels meant to be profiled and rewritten.

**Human owns (agent assists, never silently decides):**

- The performance reasoning — why a kernel is fast. An agent may draft the analysis; the
  human must be able to reconstruct and defend it unaided.
- Interpretation of profiles (`nsys`, SASS, `ptxas -v`) that drives the next design step.

## The whiteboard litmus

Anything presented as "deep work" must be defensible at a whiteboard **without** an AI. The
test for whether an optimization is truly yours: can you explain, cold, why it's faster and
what the next bottleneck is? If not, it isn't done — the number may be real but the
understanding hasn't been earned yet. Agents are for reaching that understanding faster, not
for substituting it.

## Definition of done (restated)

A kernel is finished only with (1) a measured number as % of its reference and (2) a
why-fast explanation the human can defend unaided. Never ship the number alone.

## How leverage is shown, not claimed

- **Leverage is implicit in breadth and velocity.** A solo developer shipping a GEMM ladder,
  FlashAttention, quantization, and an engine is itself the evidence. It doesn't need to be
  announced as a feature.
- **Show the process honestly.** A short "tools & workflow" note is a methodology, not a
  boast. "Built with AI!" as an achievement signals shallowness; the AI fluency is table
  stakes, the kernel depth is the differentiator.
- **Judgment is the senior signal.** Knowing what to delegate and what to own — and drawing
  that line at the whiteboard litmus — is the thing worth demonstrating.

## A capstone that shows both at once

An agent-driven **kernel autotuner**: agents sweep tile/pipeline configs, read the profiles,
propose the next candidate, and summarize *why* a winner is fast with SASS/occupancy
evidence. It requires kernel depth (you must know what to tune and how to read the result)
and agent orchestration (the search is automated) in one artifact — and the "reason about a
scheduling search space" mindset transfers toward the compiler track.
