"""Large-sequence Triton attention path.

This is the most aggressive tile of the three families.  It still uses the
same online-softmax algorithm, so the N x N score matrix is never written to
GPU global memory.
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
    """Run the large-shape Triton attention configuration."""

    # 64 x 128 gives the GPU more arithmetic per program for long sequences.
    return launch_attention(
        q,
        k,
        v,
        valid_token_mask,
        causal,
        block_m=64,
        block_n=128,
        num_warps=8,
        num_stages=3,
    )
