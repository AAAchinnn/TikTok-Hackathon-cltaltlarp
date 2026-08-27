"""Shared Triton implementation used by the small/medium/large paths.

The three public kernel-family files choose different tile sizes and launch
settings, while this file contains the common mathematical implementation.

Attention computed here is:

    softmax(Q K^T / sqrt(D)) V

The kernel also understands the two masking rules used by the supplied
benchmark:

1. A valid-token mask removes padded tokens from the KEY side of attention.
2. Causal attention removes keys to the right of the current query.

Rows that correspond to invalid QUERY tokens are written as zero.  This is
important because the benchmark expects padded output tokens to be zero.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    valid_mask_ptr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    mask_stride_b,
    mask_stride_n,
    n_ctx,
    head_dim,
    scale,
    causal: tl.constexpr,
    has_valid_mask: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program computes BLOCK_M query rows for one batch/head pair."""

    # Program IDs tell this instance which batch, head, and query tile it owns.
    pid_m = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols_d = tl.arange(0, BLOCK_D)

    row_valid = rows < n_ctx
    dim_valid = cols_d < head_dim
    q_value_mask = row_valid[:, None] & dim_valid[None, :]

    # Load Q once.  We convert the computation copy to FP32 so that the dot
    # product and the softmax reduction are numerically stable.
    q = tl.load(
        q_ptr
        + batch * stride_qb
        + head * stride_qh
        + rows[:, None] * stride_qm
        + cols_d[None, :] * stride_qd,
        mask=q_value_mask,
        other=0.0,
    ).to(tl.float32)

    # Online softmax state.
    #
    # m_i = largest score seen so far for each query row.
    # l_i = sum(exp(score - m_i)) for all keys seen so far.
    # acc = weighted sum of V values after the same rescaling.
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # We stream K/V tiles instead of allocating an N x N score matrix.
    for start_n in range(0, n_ctx, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)
        key_in_range = cols < n_ctx

        k = tl.load(
            k_ptr
            + batch * stride_kb
            + head * stride_kh
            + cols[:, None] * stride_kn
            + cols_d[None, :] * stride_kd,
            mask=key_in_range[:, None] & dim_valid[None, :],
            other=0.0,
        ).to(tl.float32)

        v = tl.load(
            v_ptr
            + batch * stride_vb
            + head * stride_vh
            + cols[:, None] * stride_vn
            + cols_d[None, :] * stride_vd,
            mask=key_in_range[:, None] & dim_valid[None, :],
            other=0.0,
        ).to(tl.float32)

        # QK^T creates a BLOCK_M x BLOCK_N tile of attention scores.
        scores = tl.dot(q, tl.trans(k)) * scale

        # Keys outside the real sequence are invalid.
        scores = tl.where(key_in_range[None, :], scores, float("-inf"))

        # Remove padded keys from attention.
        if has_valid_mask:
            valid_keys = tl.load(
                valid_mask_ptr
                + batch * mask_stride_b
                + cols * mask_stride_n,
                mask=key_in_range,
                other=0,
            )
            scores = tl.where(valid_keys[None, :], scores, float("-inf"))

        # Causal attention may only look at the current token and tokens before it.
        if causal:
            scores = tl.where(cols[None, :] <= rows[:, None], scores, float("-inf"))

        # Stable online softmax update.
        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(m_i, block_max)

        old_scale = tl.exp(m_i - new_max)
        probs = tl.exp(scores - new_max[:, None])
        block_sum = tl.sum(probs, axis=1)

        l_i = l_i * old_scale + block_sum
        acc = acc * old_scale[:, None] + tl.dot(probs, v)
        m_i = new_max

    # For every real query row, acc is the unnormalized weighted V sum and l_i
    # is the matching softmax denominator.
    out = acc / l_i[:, None]

    # Padded query rows must be zero in the benchmark's final output.
    if has_valid_mask:
        valid_queries = tl.load(
            valid_mask_ptr + batch * mask_stride_b + rows * mask_stride_n,
            mask=row_valid,
            other=0,
        ).to(tl.int1)
        out = tl.where(valid_queries[:, None], out, 0.0)

    tl.store(
        o_ptr
        + batch * stride_ob
        + head * stride_oh
        + rows[:, None] * stride_om
        + cols_d[None, :] * stride_od,
        out,
        mask=q_value_mask,
    )


def launch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: torch.Tensor | None,
    causal: bool,
    *,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    """Validate inputs, allocate output, and launch the shared Triton kernel."""

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must have shape [B, H, N, D]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k and v must have identical shapes")
    if not q.is_cuda:
        raise ValueError("Triton attention requires CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("supported dtypes are float16, bfloat16 and float32")

    batch, heads, n_ctx, head_dim = q.shape

    # The kernel uses a power-of-two D tile and masks the unused lanes.  Keeping
    # this capped prevents accidentally creating an enormous register tile.
    block_d = triton.next_power_of_2(head_dim)
    if block_d > 256:
        raise ValueError("Triton attention currently supports head_dim <= 256")

    if valid_token_mask is None:
        # A zero-size/dummy pointer is not safe to dereference in a compiled
        # kernel, so use a one-element dummy mask when the mask is absent.  The
        # has_valid_mask constexpr tells Triton not to read it.
        dummy_mask = torch.empty((1,), device=q.device, dtype=torch.bool)
        mask = dummy_mask
        has_valid_mask = False
    else:
        if valid_token_mask.shape != (batch, n_ctx):
            raise ValueError(
                "valid_token_mask must have shape [B, N] when supplied"
            )
        if valid_token_mask.device != q.device:
            raise ValueError("valid_token_mask must be on the same device as q")
        mask = valid_token_mask.to(dtype=torch.bool)
        has_valid_mask = True

    out = torch.empty_like(q)
    grid = (triton.cdiv(n_ctx, block_m), heads, batch)

    attention_kernel[grid](
        q,
        k,
        v,
        out,
        mask,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        mask.stride(0),
        mask.stride(1),
        n_ctx,
        head_dim,
        1.0 / math.sqrt(head_dim),
        causal=causal,
        has_valid_mask=has_valid_mask,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out
