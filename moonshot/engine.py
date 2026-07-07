"""Engine orchestration (skeleton).

This is the spine described in ``docs/architecture.md``: Python owns weight loading,
tokenizer, KV cache, sampling, batching, and the serving loop, while the per-layer decode
math goes through :mod:`moonshot.ops` (custom kernel if built, else torch fallback).

Skeleton only — the pieces below are stubs with the intended shape, filled in over Phase 3.
The design invariant: a decode layer never knows whether an op was a kernel or a fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import ops


@dataclass
class ModelConfig:
    hidden_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    rms_eps: float = 1e-6


def decode_layer(
    x: torch.Tensor,
    layer_weights: dict[str, torch.Tensor],
    cfg: ModelConfig,
) -> torch.Tensor:
    """One transformer decode block, expressed purely through the op seam.

    Every call below is a custom kernel when compiled, a torch fallback otherwise — so this
    function is correct before any kernel exists and gets faster as kernels land.
    """
    # RMSNorm -> QKV proj -> attention -> O proj -> RMSNorm -> MLP (SwiGLU)
    h = ops.rmsnorm(x, layer_weights["attn_norm"], cfg.rms_eps)
    q = ops.linear(h, layer_weights["wq"])
    k = ops.linear(h, layer_weights["wk"])
    v = ops.linear(h, layer_weights["wv"])
    # (reshape to heads / apply RoPE / KV-cache append omitted in the skeleton)
    attn = ops.attention(q, k, v, causal=True)
    x = x + ops.linear(attn, layer_weights["wo"])

    h = ops.rmsnorm(x, layer_weights["mlp_norm"], cfg.rms_eps)
    gate = ops.linear(h, layer_weights["w_gate"])
    up = ops.linear(h, layer_weights["w_up"])
    x = x + ops.linear(torch.nn.functional.silu(gate) * up, layer_weights["w_down"])
    return x


class Engine:
    """Serving engine skeleton. Phase 3 fills in load/generate with KV cache + batching."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def load(self, checkpoint_path: str) -> "Engine":
        raise NotImplementedError("Phase 3: weight loading")

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        raise NotImplementedError("Phase 3: prefill + decode loop with KV cache")


if __name__ == "__main__":  # pragma: no cover
    print("moonshot engine skeleton — see docs/roadmap.md (Phase 3).")
