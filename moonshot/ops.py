"""The extension seam.

Every engine operation is defined here once. Each dispatches to a registered CUDA custom op
(``torch.ops.moonshot.<name>``) when the compiled extension is present, and otherwise runs a
pure-torch fallback. Consequences:

* the engine runs end-to-end with **zero** custom kernels (all fallbacks);
* each kernel's fallback is its correctness oracle and its first speed baseline
  (``% of fallback``);
* swapping a kernel in changes performance, not orchestration.

Adding a kernel: implement + register it under ``csrc/ops/<name>/`` (see
``csrc/ops/template/``), then make sure a dispatcher below prefers the custom op while
keeping the fallback. See ``docs/architecture.md``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Load the compiled custom-op extension if it was built (torch cpp_extension). Absent on a
# fresh checkout or on a box without CUDA — the engine then runs on fallbacks only.
try:  # pragma: no cover - depends on build state
    import moonshot._C  # noqa: F401  registers torch.ops.moonshot.*

    _HAS_EXT = True
except ImportError:  # pragma: no cover
    _HAS_EXT = False


def _custom(name: str):
    """Return the registered custom op ``name`` if available, else ``None``."""
    if _HAS_EXT and hasattr(torch.ops, "moonshot"):
        return getattr(torch.ops.moonshot, name, None)
    return None


def has_custom(name: str) -> bool:
    """Whether a compiled custom op is backing ``name`` (vs the torch fallback)."""
    return _custom(name) is not None


# --- ops --------------------------------------------------------------------------------
# Each op: prefer the custom kernel; otherwise a correct, unoptimized reference.


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    op = _custom("rmsnorm")
    if op is not None:
        return op(x, weight, eps)
    dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps)).to(dtype) * weight


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    op = _custom("linear")
    if op is not None:
        return op(x, weight, bias)
    return F.linear(x, weight, bias)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    op = _custom("attention")
    if op is not None:
        return op(q, k, v, causal)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def dequant_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None = None,
    group_size: int = 128,
) -> torch.Tensor:
    """Weight-only quantized matmul (e.g. W4A16). Fallback dequantizes then matmuls; the
    custom op fuses unpack → dequant → GEMM. Fallback intentionally assumes a plain
    already-dequantized weight until the packing format lands (Phase 5)."""
    op = _custom("dequant_linear")
    if op is not None:
        return op(x, qweight, scales, zeros, group_size)
    raise NotImplementedError(
        "dequant_linear fallback needs the packing format defined in Phase 5; "
        "use a dequantized `linear` until the custom op exists."
    )


__all__ = [
    "has_custom",
    "rmsnorm",
    "linear",
    "attention",
    "dequant_linear",
]
