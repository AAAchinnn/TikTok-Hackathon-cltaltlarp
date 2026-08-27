"""Medium-sequence Triton attention path.

The medium configuration gives each program a wider tile and more warps,
which is intended to provide more useful parallel work than the small path.
"""

from __future__ import annotations

import torch

from ._attention_triton import launch_attention


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: torch.Tensor | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Run the medium-shape Triton attention configuration."""

    # A 32 x 64 tile is a reasonable starting point for middle-sized sequences.
    return launch_attention(
        q,
        k,
        v,
        valid_token_mask,
        causal,
        block_m=32,
        block_n=64,
        num_warps=4,
        num_stages=3,
    )
