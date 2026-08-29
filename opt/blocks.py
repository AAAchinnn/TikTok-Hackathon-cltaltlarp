"""Block candidates and the calling convention they share.

A candidate is a plain function over tensors, registered into one of the
dispatcher's slots. The dispatcher picks which one runs for a given shape; the
encoder in general.py calls whatever it was handed. Adding a candidate is one
function plus one decorator line, and nothing else in the project changes.

Why plain functions rather than modules: `general.py` runs the layer stack
inside `torch.compile`, and Inductor inlines a called function into the same
graph. Selection happens outside the compiled region -- by the time `_run`
starts, the callables are already chosen, so Dynamo guards on their identity
and the fusion opportunities are unchanged. A candidate that is chosen for a
shape costs nothing extra for being reached through the dispatcher.

The convention, for the two sub-block slots:

    fn(normed, w, ctx) -> Tensor

`normed` is the sub-block's input *after* its LayerNorm, `w` is the layer's
packed weight tuple, and the return value is the contribution to add to the
residual stream, already cast back to `ctx.stream`. LayerNorm stays outside
because it must run in the residual dtype regardless of what a candidate does
internally, and keeping it out means no candidate can get that wrong.

The `full_block` slot is the escape hatch for anything that needs to fuse
across the whole block, and takes the layer module too:

    fn(x, layer, w, ctx) -> Tensor

It returns the complete block output. When a plan supplies a full_block, the
attn/ffn slots are ignored.

Weight tuple layout, shared by every candidate:
    0 qkv_w  1 qkv_b  2 out_w  3 out_b  4 in_w  5 in_b  6 fo_w  7 fo_b
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F

from .dispatcher import register

__all__ = ["BlockContext", "attn_general", "ffn_general"]


class BlockContext(NamedTuple):
    """Everything a candidate needs that is not a weight or an activation.

    A NamedTuple, not a dataclass: it is a tuple subclass, so Dynamo traces it
    as a structure of constants and tensors without a graph break. Every field
    is either a Python scalar the CPU already knows or a tensor built once
    outside the loop.
    """

    batch: int
    seq_len: int
    d_model: int
    num_heads: int
    head_dim: int
    attn_mask: Optional[torch.Tensor]
    is_causal: bool
    stream: torch.dtype
    qkv_dt: torch.dtype
    attn_dt: torch.dtype
    out_dt: torch.dtype
    in_dt: torch.dtype
    fo_dt: torch.dtype
    fuse_qkv: bool


@register("attn_block", "general")
def attn_general(
    normed: torch.Tensor,
    w: Tuple[torch.Tensor, ...],
    ctx: BlockContext,
) -> torch.Tensor:
    """SDPA attention with a packed qkv projection. Correct on any shape.

    This is the permanent fallback for the attention slot: it never
    materializes the [B, H, N, N] score matrix, so it is the only candidate
    that can run the largest official shapes at all, and it makes no
    assumption about head_dim, sequence length or the mask.
    """
    qkv_w, qkv_b, out_w, out_b = w[0], w[1], w[2], w[3]
    hidden = normed.to(ctx.qkv_dt)

    if ctx.fuse_qkv:
        qkv = F.linear(hidden, qkv_w, qkv_b)
        qkv = qkv.view(ctx.batch, ctx.seq_len, 3, ctx.num_heads, ctx.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    else:
        # Slicing the packed weight is free and reproduces three Linears.
        parts = []
        for start in (0, ctx.d_model, 2 * ctx.d_model):
            piece = F.linear(
                hidden,
                qkv_w[start : start + ctx.d_model],
                qkv_b[start : start + ctx.d_model],
            )
            parts.append(
                piece.view(ctx.batch, ctx.seq_len, ctx.num_heads, ctx.head_dim)
                .transpose(1, 2)
            )
        q, k, v = parts

    if ctx.attn_dt != ctx.qkv_dt:
        q, k, v = q.to(ctx.attn_dt), k.to(ctx.attn_dt), v.to(ctx.attn_dt)

    context = F.scaled_dot_product_attention(
        q, k, v, attn_mask=ctx.attn_mask, is_causal=ctx.is_causal
    )
    context = context.transpose(1, 2).reshape(ctx.batch, ctx.seq_len, ctx.d_model)
    return F.linear(context.to(ctx.out_dt), out_w, out_b).to(ctx.stream)


@register("ffn_block", "general")
def ffn_general(
    normed: torch.Tensor,
    w: Tuple[torch.Tensor, ...],
    ctx: BlockContext,
) -> torch.Tensor:
    """Up-projection, exact-erf GELU, down-projection."""
    in_w, in_b, fo_w, fo_b = w[4], w[5], w[6], w[7]
    up = F.linear(normed.to(ctx.in_dt), in_w, in_b)
    # GELU in fp32: the one elementwise op with enough curvature for
    # low-precision rounding to show up in the output. Inductor fuses the
    # up-cast, the erf and the down-cast into a single kernel, so the
    # precision is free.
    gelu = F.gelu(up.float(), approximate="none").to(ctx.fo_dt)
    return F.linear(gelu, fo_w, fo_b).to(ctx.stream)
