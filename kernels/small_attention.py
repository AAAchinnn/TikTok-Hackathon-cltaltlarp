"""Small-sequence Triton attention path.

This path uses small tiles because short sequences often become launch-bound:
the GPU has little total work, so keeping each program compact can help.
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
    """Run the small-shape Triton attention configuration."""

    # 16 query rows x 32 key rows is deliberately conservative.
    return launch_attention(
        q,
        k,
        v,
        valid_token_mask,
        causal,
        block_m=16,
        block_n=32,
        num_warps=2,
        num_stages=2,
    )
