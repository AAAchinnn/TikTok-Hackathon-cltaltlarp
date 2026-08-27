"""Shape/dtype-aware attention dispatcher.

This module is the stable integration boundary for the Transformer.  The
higher-level model does not need to know which Triton configuration won; it
only calls ``attention(...)`` and the dispatcher chooses small, medium, large,
or the correctness-first PyTorch fallback.

The supplied benchmark has one fixed sequence length per run, so shape
specialization is especially useful: the winning thresholds can be tuned on
the actual benchmark shapes without changing the Transformer code.
"""

from __future__ import annotations

import torch

from kernels import fallback


# These are deliberately easy-to-tune starting points.  The benchmark primer
# says final thresholds should come from measured latency on the target GPU.
SMALL_MAX_N = 64
MEDIUM_MAX_N = 512


# Keeping imports lazy is useful for two reasons:
#   1. The benchmark can still run its CPU reference/fallback without Triton.
#   2. Triton compilation happens only when a custom CUDA path is actually used.
def _get_small():
    from kernels import small_attention
    return small_attention


def _get_medium():
    from kernels import medium_attention
    return medium_attention


def _get_large():
    from kernels import large_attention
    return large_attention


def select_path(q: torch.Tensor) -> str:
    """Return the implementation family for a Q tensor without launching it."""

    if q.ndim != 4:
        return "fallback"
    if not q.is_cuda:
        return "fallback"
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return "fallback"

    _, _, n_ctx, head_dim = q.shape

    # The shared Triton kernel currently caps the internal feature tile at 256.
    if head_dim > 256:
        return "fallback"

    if n_ctx <= SMALL_MAX_N:
        return "small"
    if n_ctx <= MEDIUM_MAX_N:
        return "medium"
    return "large"


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: torch.Tensor | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Dispatch attention while preserving the benchmark's masking semantics."""

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must have shape [B, H, N, D]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k and v must have identical shapes [B, H, N, D]")

    if valid_token_mask is not None:
        expected = (q.shape[0], q.shape[2])
        if valid_token_mask.shape != expected:
            raise ValueError(
                f"valid_token_mask must have shape {expected}, "
                f"got {tuple(valid_token_mask.shape)}"
            )
        if valid_token_mask.device != q.device:
            raise ValueError("valid_token_mask must be on the same device as q")

    path = select_path(q)

    if path == "small":
        return _get_small().forward(q, k, v, valid_token_mask, causal)
    if path == "medium":
        return _get_medium().forward(q, k, v, valid_token_mask, causal)
    if path == "large":
        return _get_large().forward(q, k, v, valid_token_mask, causal)

    # Unsupported shapes never fail merely because a custom kernel does not
    # cover them.  They use the well-tested PyTorch implementation instead.
    return fallback.forward(q, k, v, valid_token_mask, causal)
