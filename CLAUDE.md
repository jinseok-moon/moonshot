# CLAUDE.md

This project's agent guide is **[`AGENTS.md`](AGENTS.md)** — shared by Claude Code and
Codex so both tools follow the same rules. Read it first.

@AGENTS.md

Quick reminders (full detail in AGENTS.md):

- **Delegate vs own:** you drive plumbing, harnesses, docs, and codebase navigation. You do
  **not** silently own the *why-fast* performance reasoning — that must be human-defensible
  at a whiteboard. A kernel is done only with a number **and** an explanation.
- **No `ncu`** on the target box — use CUDA events, `ptxas -v`, SASS, `nsys`.
- **CUDA does not build on macOS** — reason statically here, run on the 3090.
- **`private/` is off-limits** — never quote, move, or push its contents.
