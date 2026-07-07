---
title: Kernel notes index
status: living
audience: [human, agent]
updated: 2026-07-07
tags: [kernels, baselines]
related: [../roadmap.md, ../hardware-and-measurement.md]
---

# Kernel notes

One note per kernel (or ladder), recording its baseline and — non-negotiably — its
**why-fast** explanation. A kernel without a note here is not done
([definition of done](../hardware-and-measurement.md#definition-of-done)).

## Template for a kernel note

```markdown
---
title: <op> — <rung>
status: living
audience: [human, agent]
updated: YYYY-MM-DD
tags: [kernel, <op>]
related: []
---

# <op> — <rung>

- **Reference:** torch fallback / cuBLAS / SDPA
- **Result:** <% of reference> across <shapes>
- **Registers / smem / spills:** from `ptxas -v`
- **Key SASS:** HMMA / LDGSTS present? occupancy?
- **Why it's faster than the previous rung:** <the reasoning, whiteboard-defensible>
- **Next bottleneck:** <what the profile says to do next>
```

## Notes

_None yet — Phase 1 (bf16 GEMM ladder) lands the first ones._
