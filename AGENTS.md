# AGENTS.md

Operating manual for coding agents working in `moonshot`. Shared by **Claude Code** and
**Codex** (Claude Code reads it via [`CLAUDE.md`](CLAUDE.md)). Read this before touching
the repo; it encodes not just *how* to build but *what an agent is and isn't allowed to
own here*.

## What this project is

A from-scratch LLM inference engine for the RTX 3090 (`sm_86`), built kernel-up. Python
orchestrates; CUDA C++ custom ops do the work. The design rule: **the engine runs on torch
fallbacks with zero custom kernels, and each kernel you add replaces one fallback.** See
[`docs/overview.md`](docs/overview.md) and [`docs/architecture.md`](docs/architecture.md).

## Read order for a fresh agent

1. [`docs/llms.txt`](docs/llms.txt) — machine index of the knowledge base.
2. [`docs/overview.md`](docs/overview.md) — what and why.
3. [`docs/roadmap.md`](docs/roadmap.md) — current phase and what's next.
4. [`docs/architecture.md`](docs/architecture.md) — the extension seam.
5. [`docs/hardware-and-measurement.md`](docs/hardware-and-measurement.md) — the box and how
   results are measured.

## The one rule: delegate vs own

This is the load-bearing section. Full doctrine in
[`docs/ai-collaboration.md`](docs/ai-collaboration.md).

**Agents own (drive freely):**
- Python plumbing — weight loading, tokenizer glue, sampling, serving loop, KV-cache
  bookkeeping, the op-registry seam.
- Benchmark harnesses, CSV/plot generation, build config, CI.
- Docs: keep the knowledge base in sync, write ADRs, cross-link.
- Large-codebase navigation and summarization (e.g. reading vLLM to locate a hook).
- Test and verification scaffolds; *first-draft* kernels meant to be profiled and rewritten.

**Agents must NOT silently own (human decides, agent assists):**
- The performance reasoning — *why* a kernel is fast (occupancy, roofline, memory-latency,
  bank conflicts, SASS). An agent may draft an analysis, but the human must be able to
  defend it at a whiteboard without the agent. If an optimization can't be explained, it is
  not done.
- Interpretation of profiles (`nsys`, SASS, `ptxas -v`) that drives the next design step.

**Definition of done for a kernel:** (1) a measured number as **% of the fallback / a
production reference**, and (2) a why-fast explanation a human can defend unaided. Missing
either → not done. Never present a kernel as finished on the number alone.

## How to add a kernel (the extension seam)

1. Copy `csrc/ops/template/` to `csrc/ops/<name>/`; implement the kernel and register it
   with `TORCH_LIBRARY` (see the template's README).
2. In `moonshot/ops.py`, the `<name>` dispatcher already prefers
   `torch.ops.moonshot.<name>` when the compiled extension is present; add the dispatcher
   if the op is new, keeping the torch fallback.
3. Bench the kernel against its fallback; record the number **and** the why-fast note in
   `docs/kernels/<name>.md`.

Never delete or mutate a prior ladder rung in place — each optimization step is a new
`kernel_N` file. The ladder is the story.

## Build & bench

Target is CUDA on `sm_86`. **CUDA does not build on macOS** — edit and reason statically on
a dev Mac; compile and run on the 3090.

```bash
pip install -e .                 # builds csrc custom ops via torch cpp_extension (on the 3090)
python -m moonshot.engine ...    # run the engine (torch fallbacks until kernels land)
scripts/inspect.sh csrc/ops/<name>/<file>.cu   # ptxas -v (reg/smem/spill) + SASS
```

`ncu` is blocked on the target box (`ERR_NVGPUCTRPERM`). Use CUDA events, `ptxas -v`,
`cuobjdump -sass`, and `nsys`. Do not add workflows that assume `ncu`.

## Conventions

- **Reporting:** every kernel result is `% of fallback / cuBLAS / SDPA`. Absolute GFLOP/s
  alone is not a result. Verify correctness against the fallback (`max rel err < 2e-2` for bf16).
- **C++/CUDA:** C++20; headers `.cuh`/`.hpp`; shared helpers in `csrc/include/moonshot`.
  Match surrounding style — comment *why*, not *what*.
- **Python:** kernels exposed as `torch.library` / `cpp_extension` custom ops; keep ops
  individually benchable before fusing a whole decode layer.
- **Docs are code.** Any decision or new baseline updates the knowledge base in the same
  change. Architecture/strategy shifts get an ADR under `docs/decisions/`. Update
  `docs/llms.txt` when adding a doc.

## Knowledge base format (open-knowledge)

Docs under `docs/` carry YAML frontmatter (`title`, `status`, `audience`, `updated`,
`tags`, `related`) and cross-link with relative paths. Write for both a human skimming and
an agent grepping.

## Private material

`private/` is git-ignored and holds personal career strategy. **Never** move its contents
into tracked files, quote it in commits/PRs, or push it. Public docs stay engineering-only.

## Commits

Branch off `main` for non-trivial work; keep the ladder narrative intact. End commit
messages with the co-author trailer used by this project's tooling.
