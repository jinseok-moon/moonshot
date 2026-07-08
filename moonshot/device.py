"""Device capability detection — keeps kernel dispatch portable across GPU architectures.

The engine runs on any backend torch supports (CUDA including ROCm, or CPU) via the
fallbacks in :mod:`moonshot.ops`. A hand-written CUDA kernel only runs when the current
device meets the compute capability it needs; otherwise dispatch falls back to torch. That
is what lets moonshot "support other GPUs" without per-GPU branches in the engine.

Compute-capability feature map (NVIDIA):

    fp16 tensor cores   Volta+     (7.0)
    int8 tensor cores   Turing+    (7.5)   mma.sync.s8
    bf16 tensor cores   Ampere+    (8.0)
    cp.async (LDGSTS)   Ampere+    (8.0)
    fp8 tensor cores    Ada/Hopper (8.9 / 9.0)
"""

from __future__ import annotations

import functools

import torch


@functools.lru_cache(maxsize=None)
def cuda_capability() -> tuple[int, int]:
    """(major, minor) of the current CUDA device, or (0, 0) if no CUDA device."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_capability()
    return (0, 0)


def meets(min_capability: tuple[int, int]) -> bool:
    """Whether the current device is at least ``min_capability``."""
    return cuda_capability() >= min_capability


def has_fp16_tensorcore() -> bool:
    return meets((7, 0))


def has_int8_tensorcore() -> bool:
    return meets((7, 5))


def has_bf16_tensorcore() -> bool:
    return meets((8, 0))


def has_cp_async() -> bool:
    return meets((8, 0))


def has_fp8_tensorcore() -> bool:
    return cuda_capability() in {(8, 9), (9, 0)} or cuda_capability()[0] >= 9


def summary() -> str:
    """Human-readable one-liner for logs."""
    if not torch.cuda.is_available():
        return "no CUDA device — engine runs on CPU / torch fallbacks"
    maj, minor = cuda_capability()
    return f"{torch.cuda.get_device_name()} (sm_{maj}{minor})"
