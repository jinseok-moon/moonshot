"""moonshot — a from-scratch LLM inference engine for Ampere, built kernel-up.

The engine runs on pure-torch fallbacks with zero custom kernels; each CUDA kernel added
under ``csrc/ops/`` replaces one fallback through the seam in :mod:`moonshot.ops`.
"""

__version__ = "0.0.0"

from . import ops  # noqa: F401

__all__ = ["ops", "__version__"]
