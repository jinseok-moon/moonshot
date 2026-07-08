"""The extension seam.

Every engine operation is defined here once. Each dispatches to a registered CUDA custom op
(``torch.ops.moonshot.<name>``) when the compiled extension is present **and** the current
device meets the compute capability that kernel needs; otherwise it runs a pure-torch
fallback. Consequences:

* the engine runs end-to-end with **zero** custom kernels (all fallbacks), on any GPU torch
  supports — a kernel built for Ampere simply falls back on older cards;
* each kernel's fallback is its correctness oracle and its first speed baseline
  (``% of fallback``);
* swapping a kernel in changes performance, not orchestration.

Adding a kernel: implement + register it under ``csrc/ops/<name>/`` (see
``csrc/ops/template/``), then make sure a dispatcher below prefers the custom op while
keeping the fallback and passing the capability the kernel requires. See
``docs/architecture.md``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .device import meets

# Load the compiled custom-op extension if it was built (torch cpp_extension). Absent on a
# fresh checkout or on a box without CUDA — the engine then runs on fallbacks only.
try:  # pragma: no cover - depends on build state
    import moonshot._C  # noqa: F401  registers torch.ops.moonshot.*

    _HAS_EXT = True
except ImportError:  # pragma: no cover
    _HAS_EXT = False


def _custom(name: str, min_capability: tuple[int, int] = (7, 0)):
    """Return the registered custom op ``name`` if it is available *and* the current device
    meets ``min_capability``; otherwise ``None`` (caller uses the torch fallback)."""
    if not (_HAS_EXT and hasattr(torch.ops, "moonshot")):
        return None
    op = getattr(torch.ops.moonshot, name, None)
    if op is None:
        return None
    if not meets(min_capability):
        return None  # device too old for this kernel — fall back
    return op


def has_custom(name: str, min_capability: tuple[int, int] = (7, 0)) -> bool:
    """Whether a compiled custom op is backing ``name`` on this device (vs the fallback)."""
    return _custom(name, min_capability) is not None


# --- ops --------------------------------------------------------------------------------
# Each op: prefer the custom kernel (when supported on this device), else a correct,
# unoptimized reference. `min_capability` reflects what the *kernel* needs, not the op.


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    op = _custom("rmsnorm", (7, 0))  # memory-bound; runs broadly
    if op is not None:
        return op(x, weight, eps)
    dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps)).to(dtype) * weight


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    op = _custom("linear", (8, 0))  # first kernel is a bf16 tensor-core GEMM (Ampere+)
    if op is not None:
        return op(x, weight, bias)
    return F.linear(x, weight, bias)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    op = _custom("attention", (8, 0))  # FlashAttention uses cp.async + bf16 TC (Ampere+)
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
    op = _custom("dequant_linear", (8, 0))
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
