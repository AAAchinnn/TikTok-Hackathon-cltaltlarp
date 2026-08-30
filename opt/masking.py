"""Mask analysis: everything that has to happen on the CPU, once, and be cached.

All three facts below need a GPU->CPU sync to read. That is fine exactly once
per distinct mask tensor -- the first read lands in `warmup_model`, and every
later forward is a dict lookup. It is not fine per forward, which is why none
of this may move inside the compiled region: `.item()` forces a graph break.

Caches are keyed on `(data_ptr, shape)` and hold a reference to the mask
tensor itself. Holding the reference is load-bearing: without it the
allocation could be freed and a different mask could be handed the same
pointer, turning a stale entry into a silent wrong answer.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from .precision import settings

__all__ = [
    "MaskFacts",
    "analyse",
    "additive_bias",
    "build_attn_mask",
]

_CACHE_LIMIT = 32


class MaskFacts:
    """What we know about a key-padding mask, read once."""

    __slots__ = ("tensor", "all_valid", "suffix_padded")

    def __init__(
        self, tensor: torch.Tensor, all_valid: bool, suffix_padded: bool
    ) -> None:
        self.tensor = tensor
        self.all_valid = all_valid
        self.suffix_padded = suffix_padded

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"MaskFacts(all_valid={self.all_valid}, "
            f"suffix_padded={self.suffix_padded})"
        )


_facts_cache: Dict[Tuple, MaskFacts] = {}
_bias_cache: Dict[Tuple, torch.Tensor] = {}


def analyse(mask: Optional[torch.Tensor]) -> Optional[MaskFacts]:
    """Read both mask properties in one sync, then cache them.

    `all_valid`: if every position is valid, masking is a no-op and the mask
    can be dropped entirely. That removes one masked_fill per layer, worth
    ~4% at the hub shape (2.01x with elision, 1.93x without).

    `suffix_padded`: whether each row is a run of True followed by a run of
    False. This is the assumption the causal path used to *make* -- under
    causal attention with suffix padding, a valid query at position i attends
    only keys j <= i < length, all of which are valid, so the key-padding mask
    is redundant and `is_causal=True` alone reproduces the baseline exactly.
    That is a statement about the data, not the shape, and if it were ever
    false the output would be silently wrong. Verifying costs one extra
    reduction on a sync we are already paying, so there is no reason to
    assume it.
    """
    if mask is None:
        return None

    key = (mask.data_ptr(), tuple(mask.shape))
    cached = _facts_cache.get(key)
    if cached is not None:
        return cached

    all_valid = bool(mask.all().item())
    if all_valid or not settings.verify_suffix_padding or mask.shape[-1] < 2:
        suffix_padded = True
    else:
        # A violation is a False immediately followed by a True.
        gap = (~mask[..., :-1]) & mask[..., 1:]
        suffix_padded = not bool(gap.any().item())

    facts = MaskFacts(mask, all_valid, suffix_padded)
    if len(_facts_cache) >= _CACHE_LIMIT:
        _facts_cache.clear()
    _facts_cache[key] = facts
    return facts


def effective_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """The mask with all-valid cases collapsed to None."""
    facts = analyse(mask)
    if facts is None:
        return None
    if settings.elide_mask and facts.all_valid:
        return None
    return mask


def additive_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """[B, 1, 1, N] bias: 0 where valid, -inf where padded.

    Float bias rather than a bool mask, because the fused SDPA backends refuse
    bool masks more often, and a refusal falls back to MATH -- which is the
    baseline's own algorithm, so the win disappears with no warning.

    -inf rather than a large negative constant, to match the baseline's
    masked_fill(-inf) bit for bit. `generate_random_case` guarantees at least
    one valid token per row, so no row is ever fully masked and no NaN appears.
    """
    key = (mask.data_ptr(), tuple(mask.shape), dtype)
    cached = _bias_cache.get(key)
    if cached is not None:
        return cached

    bias = torch.zeros(
        (mask.shape[0], 1, 1, mask.shape[1]), dtype=dtype, device=mask.device
    ).masked_fill(~mask[:, None, None, :], float("-inf"))

    if len(_bias_cache) >= _CACHE_LIMIT:
        _bias_cache.clear()
    _bias_cache[key] = bias
    return bias


def build_attn_mask(
    seq_len: int,
    mask: Optional[torch.Tensor],
    causal: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], bool]:
    """Return `(attn_mask, is_causal)`. SDPA rejects both at once.

    The causal branch takes the cheap path only when `analyse` has *confirmed*
    suffix padding; otherwise it builds the combined causal+padding bias, which
    is correct without any assumption about the data and about 7% slower.
    """
    if mask is None:
        return None, causal

    if causal:
        facts = analyse(mask)
        if facts is not None and facts.suffix_padded:
            return None, True
        blocked = torch.ones(
            (seq_len, seq_len), device=device, dtype=torch.bool
        ).triu(diagonal=1)
        causal_bias = torch.zeros(
            (seq_len, seq_len), dtype=dtype, device=device
        ).masked_fill(blocked, float("-inf"))
        return causal_bias[None, None] + additive_bias(mask, dtype), False

    return additive_bias(mask, dtype), False


def clear_caches() -> None:
    """Drop every cached fact. Called when the model moves device or dtype."""
    _facts_cache.clear()
    _bias_cache.clear()
