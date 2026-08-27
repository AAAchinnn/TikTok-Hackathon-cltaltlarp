"""PyTorch fallback for shapes that custom Triton paths do not support."""

from __future__ import annotations

import math

import torch


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: torch.Tensor | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Compute the same attention operation with PyTorch SDPA.

    The fallback intentionally accepts the same mask/causal arguments as the
    Triton kernels.  This makes it a true drop-in safety net for the dispatcher.
    """

    attn_mask = None
    if valid_token_mask is not None:
        # True means "this key may be attended to" for a boolean attention mask.
        key_mask = valid_token_mask[:, None, None, :].to(torch.bool)

        if causal:
            n_ctx = q.shape[-2]
            causal_mask = torch.ones(
                (n_ctx, n_ctx), device=q.device, dtype=torch.bool
            ).tril()
            attn_mask = key_mask & causal_mask[None, None, :, :]
        else:
            attn_mask = key_mask
    elif causal:
        # Let SDPA create its native causal implementation when there is no
        # padding mask to combine with it.
        return torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=1.0 / math.sqrt(q.shape[-1]),
        )

    out = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=1.0 / math.sqrt(q.shape[-1]),
    )

    # The benchmark explicitly zeros padded query positions after attention.
    if valid_token_mask is not None:
        out = out.masked_fill(~valid_token_mask[:, None, :, None], 0)

    return out
