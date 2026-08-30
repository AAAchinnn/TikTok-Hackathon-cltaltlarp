"""Shape-specialized Triton Transformer for the supplied torch benchmark.

The benchmark is intentionally a small, fixed-family Transformer rather than a
general training implementation.  This module keeps the benchmark's parameter
names and forward signature, and uses a shape dispatcher to choose between
accuracy-first native PyTorch and Triton kernels:

* short sequences preserve the benchmark's native LayerNorm/GELU and FP32
  softmax boundaries;
* the Triton path provides online-softmax causal attention (FlashAttention-
  style), with no score or probability tensor materialized;
* auxiliary Triton kernels for row-wise LayerNorm, residual+LayerNorm, exact
  GELU and residual add remain available for targeted experiments.

The attention kernel is deliberately written for the benchmark's head sizes
(8, 32, 64, 128 and 256) and sequence lengths (32, 128, 1024 and very long
sequences).  PyTorch remains responsible for the large dense GEMMs, where its
cuBLAS/cuBLASLt kernels are generally better than a hand-written generic
Triton GEMM.

For a 100,000-token input, exact causal attention is still quadratic in work;
the streaming path only removes the otherwise impossible O(N^2) *memory*
allocation.  It is the exact path required by the benchmark's formula.
"""

from __future__ import annotations

import importlib.util
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from dispatcher import (
        NATIVE_EXACT,
        NATIVE_SDPA,
        TRITON_FLASH,
        TRITON_STREAM,
        select_path,
    )
except ImportError:  # pragma: no cover - keeps this file standalone
    NATIVE_EXACT = "native_exact"
    NATIVE_SDPA = "native_sdpa"
    TRITON_FLASH = "triton_flash"
    TRITON_STREAM = "triton_stream"

    def select_path(**_: Any) -> str:
        return NATIVE_EXACT


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


_TRITON_AVAILABLE: Optional[bool] = None
_KERNELS: Optional[Dict[str, Any]] = None
_FAILED_KERNELS: set[str] = set()


def _triton_is_available() -> bool:
    """Check package availability without importing Triton at module import."""
    global _TRITON_AVAILABLE
    if _TRITON_AVAILABLE is None:
        _TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None
    return _TRITON_AVAILABLE


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _normalise_mask(
    valid_token_mask: Optional[torch.Tensor],
    batch: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a contiguous bool [B, S] mask for all internal kernels."""
    if valid_token_mask is None:
        return torch.ones((batch, seq_len), device=device, dtype=torch.bool)
    if valid_token_mask.shape != (batch, seq_len):
        raise ValueError(
            "valid_token_mask must have shape "
            f"({batch}, {seq_len}), got {tuple(valid_token_mask.shape)}"
        )
    return valid_token_mask.to(device=device, dtype=torch.bool).contiguous()


def _reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
    causal: bool,
) -> torch.Tensor:
    """The benchmark attention, retained as a safe fallback."""
    scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(q.shape[-1]))

    if causal:
        seq_len = q.shape[-2]
        causal_mask = torch.ones(
            (seq_len, seq_len), device=q.device, dtype=torch.bool
        ).triu(diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))

    if valid_token_mask is not None:
        scores = scores.masked_fill(
            ~valid_token_mask[:, None, None, :], float("-inf")
        )
    probs = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
    output = torch.matmul(probs, v)
    if valid_token_mask is not None:
        output = output.masked_fill(
            ~valid_token_mask[:, None, :, None],
            0,
        )
    return output


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


def _get_kernels() -> Optional[Dict[str, Any]]:
    """Build Triton kernels lazily so CPU-only benchmark runs still import."""
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    if not _triton_is_available():
        return None

    import triton
    import triton.language as tl

    @triton.jit
    def layer_norm_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        mask_ptr,
        y_ptr,
        stride_x_row,
        stride_x_col,
        stride_mask,
        stride_y_row,
        stride_y_col,
        n_cols,
        eps,
        BLOCK_N: tl.constexpr,
    ):
        """LayerNorm over one flattened token row."""
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        col_mask = cols < n_cols

        x = tl.load(
            x_ptr + pid * stride_x_row + cols * stride_x_col,
            mask=col_mask,
            other=0.0,
        )
        row_valid = tl.load(mask_ptr + pid * stride_mask, other=0).to(tl.int1)
        x = tl.where(row_valid, x, 0.0)
        x_f32 = x.to(tl.float32)

        mean = tl.sum(x_f32, axis=0) / n_cols
        centered = x_f32 - mean
        variance = tl.sum(centered * centered, axis=0) / n_cols
        inv_std = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=1.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        y = (centered * inv_std) * weight + bias
        y = tl.where(row_valid, y, 0.0)

        # x.dtype is the model dtype (fp16, bf16 or fp32); this preserves the
        # benchmark's dtype boundary after LayerNorm.
        tl.store(
            y_ptr + pid * stride_y_row + cols * stride_y_col,
            y.to(x.dtype),
            mask=col_mask,
        )

    @triton.jit
    def residual_layer_norm_kernel(
        x_ptr,
        add_ptr,
        weight_ptr,
        bias_ptr,
        mask_ptr,
        residual_out_ptr,
        norm_out_ptr,
        stride_x_row,
        stride_x_col,
        stride_add_row,
        stride_add_col,
        stride_mask,
        stride_res_row,
        stride_res_col,
        stride_norm_row,
        stride_norm_col,
        n_cols,
        eps,
        BLOCK_N: tl.constexpr,
    ):
        """Store x+add and normalize that *stored* dtype in one pass.

        The explicit cast before the reduction is important: the reference
        performs the residual add in the model dtype, then LayerNorm reads the
        rounded residual tensor.
        """
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        col_mask = cols < n_cols

        x = tl.load(
            x_ptr + pid * stride_x_row + cols * stride_x_col,
            mask=col_mask,
            other=0.0,
        )
        add = tl.load(
            add_ptr + pid * stride_add_row + cols * stride_add_col,
            mask=col_mask,
            other=0.0,
        )
        row_valid = tl.load(mask_ptr + pid * stride_mask, other=0).to(tl.int1)

        residual = (x + add).to(x.dtype)
        residual = tl.where(row_valid, residual, 0.0)
        tl.store(
            residual_out_ptr + pid * stride_res_row + cols * stride_res_col,
            residual,
            mask=col_mask,
        )

        residual_f32 = residual.to(tl.float32)
        mean = tl.sum(residual_f32, axis=0) / n_cols
        centered = residual_f32 - mean
        variance = tl.sum(centered * centered, axis=0) / n_cols
        inv_std = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=1.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        norm = (centered * inv_std) * weight + bias
        norm = tl.where(row_valid, norm, 0.0)
        tl.store(
            norm_out_ptr + pid * stride_norm_row + cols * stride_norm_col,
            norm.to(x.dtype),
            mask=col_mask,
        )

    @triton.jit
    def add_residual_kernel(
        x_ptr,
        add_ptr,
        mask_ptr,
        stride_x_row,
        stride_x_col,
        stride_add_row,
        stride_add_col,
        stride_mask,
        n_cols,
        BLOCK_N: tl.constexpr,
    ):
        """In-place model-dtype residual add with padding zeroing."""
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        col_mask = cols < n_cols

        x = tl.load(
            x_ptr + pid * stride_x_row + cols * stride_x_col,
            mask=col_mask,
            other=0.0,
        )
        add = tl.load(
            add_ptr + pid * stride_add_row + cols * stride_add_col,
            mask=col_mask,
            other=0.0,
        )
        row_valid = tl.load(mask_ptr + pid * stride_mask, other=0).to(tl.int1)
        y = (x + add).to(x.dtype)
        y = tl.where(row_valid, y, 0.0)
        tl.store(
            x_ptr + pid * stride_x_row + cols * stride_x_col,
            y,
            mask=col_mask,
        )

    @triton.jit
    def gelu_inplace_kernel(
        x_ptr,
        n_elements,
        BLOCK: tl.constexpr,
    ):
        """Exact (non-approximate) GELU, matching F.gelu(approximate='none')."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        y = 0.5 * x_f32 * (1.0 + tl.erf(x_f32 * 0.7071067811865475))
        tl.store(x_ptr + offsets, y.to(x.dtype), mask=mask)

    @triton.jit
    def flash_attention_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        mask_ptr,
        o_ptr,
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
        stride_mb,
        stride_mn,
        stride_ob,
        stride_oh,
        stride_om,
        stride_od,
        n_ctx,
        head_dim,
        scale,
        causal: tl.constexpr,
        SINGLE_TILE: tl.constexpr,
        LOOP_STAGES: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused QK^T, causal/padding mask, online softmax, and PV.

        The running (m, l, acc) state is FP32.  No [B,H,S,S] temporary is
        allocated, which is the key difference from the benchmark reference.
        """
        pid_m = tl.program_id(0)
        head = tl.program_id(1)
        batch = tl.program_id(2)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        dims = tl.arange(0, BLOCK_D)
        row_valid = rows < n_ctx
        dim_valid = dims < head_dim

        q = tl.load(
            q_ptr
            + batch * stride_qb
            + head * stride_qh
            + rows[:, None] * stride_qm
            + dims[None, :] * stride_qd,
            mask=row_valid[:, None] & dim_valid[None, :],
            other=0.0,
        )

        query_valid = tl.load(
            mask_ptr + batch * stride_mb + rows * stride_mn,
            mask=row_valid,
            other=0,
        ).to(tl.int1)
        query_valid = row_valid & query_valid

        running_max = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
        accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        # tl.range keeps the key scan a compact runtime loop.  This matters
        # for the 100k-token case: a Python range would ask Triton to unroll
        # hundreds of iterations into the generated program.  For causal
        # attention, stop after the largest query row in this tile; that cuts
        # the scan from N^2 to roughly N^2/2 work without changing results.
        if causal:
            key_end = tl.minimum(n_ctx, tl.max(rows, axis=0) + 1)
        else:
            key_end = n_ctx
        if SINGLE_TILE:
            # One tile can be normalized before the probability cast, which
            # matches the reference's FP32 softmax -> model-dtype P -> PV
            # boundary closely.
            for start_n in tl.range(
                0,
                key_end,
                BLOCK_N,
                num_stages=LOOP_STAGES,
            ):
                cols = start_n + tl.arange(0, BLOCK_N)
                col_valid = cols < n_ctx
                k = tl.load(
                    k_ptr
                    + batch * stride_kb
                    + head * stride_kh
                    + cols[:, None] * stride_kn
                    + dims[None, :] * stride_kd,
                    mask=col_valid[:, None] & dim_valid[None, :],
                    other=0.0,
                )
                v = tl.load(
                    v_ptr
                    + batch * stride_vb
                    + head * stride_vh
                    + cols[:, None] * stride_vn
                    + dims[None, :] * stride_vd,
                    mask=col_valid[:, None] & dim_valid[None, :],
                    other=0.0,
                )
                if BLOCK_D <= 16:
                    scores = tl.sum(
                        q[:, None, :] * k[None, :, :],
                        axis=2,
                    ).to(tl.float32) * scale
                else:
                    scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
                scores = scores.to(q.dtype).to(tl.float32)
                key_valid = tl.load(
                    mask_ptr + batch * stride_mb + cols * stride_mn,
                    mask=col_valid,
                    other=0,
                ).to(tl.int1)
                allowed = col_valid[None, :] & key_valid[None, :]
                if causal:
                    allowed = allowed & (cols[None, :] <= rows[:, None])
                allowed = allowed & query_valid[:, None]
                scores = tl.where(allowed, scores, float("-inf"))
                tile_max = tl.max(scores, axis=1)
                safe_max = tl.where(tile_max == float("-inf"), 0.0, tile_max)
                probabilities = tl.where(
                    allowed,
                    tl.exp(scores - safe_max[:, None]),
                    0.0,
                )
                tile_sum = tl.sum(probabilities, axis=1)
                probabilities = tl.where(
                    tile_sum[:, None] > 0.0,
                    probabilities / tile_sum[:, None],
                    0.0,
                )
                running_sum = tl.where(tile_sum > 0.0, 1.0, 0.0)
                accumulator = tl.dot(
                    probabilities.to(v.dtype),
                    v,
                    out_dtype=tl.float32,
                )
                running_max = tile_max
        else:
            # For multiple key tiles, an online implementation must not cast
            # unnormalized exponentials to FP16 before the final denominator
            # is known.  That boundary was the main source of avoidable error.
            # Pass 1 computes the FP32 global max and denominator.
            for start_n in tl.range(
                0,
                key_end,
                BLOCK_N,
                num_stages=LOOP_STAGES,
            ):
                cols = start_n + tl.arange(0, BLOCK_N)
                col_valid = cols < n_ctx
                k = tl.load(
                    k_ptr
                    + batch * stride_kb
                    + head * stride_kh
                    + cols[:, None] * stride_kn
                    + dims[None, :] * stride_kd,
                    mask=col_valid[:, None] & dim_valid[None, :],
                    other=0.0,
                )
                if BLOCK_D <= 16:
                    scores = tl.sum(
                        q[:, None, :] * k[None, :, :],
                        axis=2,
                    ).to(tl.float32) * scale
                else:
                    scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
                scores = scores.to(q.dtype).to(tl.float32)
                key_valid = tl.load(
                    mask_ptr + batch * stride_mb + cols * stride_mn,
                    mask=col_valid,
                    other=0,
                ).to(tl.int1)
                allowed = col_valid[None, :] & key_valid[None, :]
                if causal:
                    allowed = allowed & (cols[None, :] <= rows[:, None])
                allowed = allowed & query_valid[:, None]
                scores = tl.where(allowed, scores, float("-inf"))
                tile_max = tl.max(scores, axis=1)
                new_max = tl.maximum(running_max, tile_max)
                safe_max = tl.where(new_max == float("-inf"), 0.0, new_max)
                old_scale = tl.where(
                    running_max == float("-inf"),
                    0.0,
                    tl.exp(running_max - safe_max),
                )
                probabilities = tl.where(
                    allowed,
                    tl.exp(scores - safe_max[:, None]),
                    0.0,
                )
                running_sum = (
                    old_scale * running_sum + tl.sum(probabilities, axis=1)
                )
                running_max = new_max

            final_max = tl.where(
                running_max == float("-inf"),
                0.0,
                running_max,
            )
            final_sum = tl.where(running_sum > 0.0, running_sum, 1.0)
            accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

            # Pass 2 recomputes scores and casts only the *normalized* P to
            # the model dtype, matching the reference operation order.
            for start_n in tl.range(
                0,
                key_end,
                BLOCK_N,
                num_stages=LOOP_STAGES,
            ):
                cols = start_n + tl.arange(0, BLOCK_N)
                col_valid = cols < n_ctx
                k = tl.load(
                    k_ptr
                    + batch * stride_kb
                    + head * stride_kh
                    + cols[:, None] * stride_kn
                    + dims[None, :] * stride_kd,
                    mask=col_valid[:, None] & dim_valid[None, :],
                    other=0.0,
                )
                v = tl.load(
                    v_ptr
                    + batch * stride_vb
                    + head * stride_vh
                    + cols[:, None] * stride_vn
                    + dims[None, :] * stride_vd,
                    mask=col_valid[:, None] & dim_valid[None, :],
                    other=0.0,
                )
                if BLOCK_D <= 16:
                    scores = tl.sum(
                        q[:, None, :] * k[None, :, :],
                        axis=2,
                    ).to(tl.float32) * scale
                else:
                    scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
                scores = scores.to(q.dtype).to(tl.float32)
                key_valid = tl.load(
                    mask_ptr + batch * stride_mb + cols * stride_mn,
                    mask=col_valid,
                    other=0,
                ).to(tl.int1)
                allowed = col_valid[None, :] & key_valid[None, :]
                if causal:
                    allowed = allowed & (cols[None, :] <= rows[:, None])
                allowed = allowed & query_valid[:, None]
                scores = tl.where(allowed, scores, float("-inf"))
                probabilities = tl.where(
                    allowed,
                    tl.exp(scores - final_max[:, None])
                    / final_sum[:, None],
                    0.0,
                )
                accumulator += tl.dot(
                    probabilities.to(v.dtype),
                    v,
                    out_dtype=tl.float32,
                )

        if SINGLE_TILE:
            output = tl.where(
                running_sum[:, None] > 0.0,
                accumulator / running_sum[:, None],
                0.0,
            )
        else:
            output = accumulator
        output = tl.where(query_valid[:, None], output, 0.0)

        tl.store(
            o_ptr
            + batch * stride_ob
            + head * stride_oh
            + rows[:, None] * stride_om
            + dims[None, :] * stride_od,
            output.to(q.dtype),
            mask=row_valid[:, None] & dim_valid[None, :],
        )

    _KERNELS = {
        "triton": triton,
        "layer_norm": layer_norm_kernel,
        "residual_layer_norm": residual_layer_norm_kernel,
        "add_residual": add_residual_kernel,
        "gelu": gelu_inplace_kernel,
        "attention": flash_attention_kernel,
    }
    return _KERNELS


def _mark_failed(name: str) -> None:
    # A compilation/runtime failure should not make every following layer pay
    # the exception cost.  The PyTorch fallback remains numerically safe.
    _FAILED_KERNELS.add(name)


def _can_use_triton(x: torch.Tensor, name: str) -> bool:
    return (
        x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and _triton_is_available()
        and name not in _FAILED_KERNELS
    )


def _launch_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
) -> Optional[torch.Tensor]:
    if not _can_use_triton(x, "layer_norm"):
        return None
    kernels = _get_kernels()
    if kernels is None:
        return None
    triton = kernels["triton"]

    tokens = x.shape[0] * x.shape[1]
    cols = x.shape[2]
    x2 = x.reshape(tokens, cols)
    y = torch.empty_like(x2)
    mask1 = mask.reshape(tokens)
    block_n = _next_power_of_two(cols)
    try:
        kernels["layer_norm"][(tokens,)](
            x2,
            weight,
            bias,
            mask1,
            y,
            x2.stride(0),
            x2.stride(1),
            mask1.stride(0),
            y.stride(0),
            y.stride(1),
            cols,
            eps,
            BLOCK_N=block_n,
            num_warps=8 if cols >= 512 else (4 if cols >= 64 else 2),
            num_stages=2,
        )
        return y.reshape_as(x)
    except Exception:
        _mark_failed("layer_norm")
        return None


def _launch_residual_layer_norm(
    x: torch.Tensor,
    add: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
) -> Optional[torch.Tensor]:
    if not _can_use_triton(x, "residual_layer_norm"):
        return None
    kernels = _get_kernels()
    if kernels is None:
        return None

    tokens = x.shape[0] * x.shape[1]
    cols = x.shape[2]
    x2 = x.reshape(tokens, cols)
    add2 = add.reshape(tokens, cols)
    normed = torch.empty_like(x2)
    mask1 = mask.reshape(tokens)
    block_n = _next_power_of_two(cols)
    try:
        kernels["residual_layer_norm"][(tokens,)](
            x2,
            add2,
            weight,
            bias,
            mask1,
            x2,
            normed,
            x2.stride(0),
            x2.stride(1),
            add2.stride(0),
            add2.stride(1),
            mask1.stride(0),
            x2.stride(0),
            x2.stride(1),
            normed.stride(0),
            normed.stride(1),
            cols,
            eps,
            BLOCK_N=block_n,
            num_warps=8 if cols >= 512 else (4 if cols >= 64 else 2),
            num_stages=2,
        )
        return normed.reshape_as(x)
    except Exception:
        _mark_failed("residual_layer_norm")
        return None


def _launch_add_residual(
    x: torch.Tensor,
    add: torch.Tensor,
    mask: torch.Tensor,
) -> bool:
    if not _can_use_triton(x, "add_residual"):
        return False
    kernels = _get_kernels()
    if kernels is None:
        return False

    tokens = x.shape[0] * x.shape[1]
    cols = x.shape[2]
    x2 = x.reshape(tokens, cols)
    add2 = add.reshape(tokens, cols)
    mask1 = mask.reshape(tokens)
    block_n = _next_power_of_two(cols)
    try:
        kernels["add_residual"][(tokens,)](
            x2,
            add2,
            mask1,
            x2.stride(0),
            x2.stride(1),
            add2.stride(0),
            add2.stride(1),
            mask1.stride(0),
            cols,
            BLOCK_N=block_n,
            num_warps=8 if cols >= 512 else (4 if cols >= 64 else 2),
            num_stages=2,
        )
        return True
    except Exception:
        _mark_failed("add_residual")
        return False


def _launch_gelu(x: torch.Tensor) -> bool:
    if not _can_use_triton(x, "gelu"):
        return False
    kernels = _get_kernels()
    if kernels is None:
        return False
    triton = kernels["triton"]
    n_elements = x.numel()
    block = 1024 if x.shape[-1] >= 512 else 256
    try:
        kernels["gelu"][(triton.cdiv(n_elements, block),)](
            x,
            n_elements,
            BLOCK=block,
            num_warps=8 if block >= 1024 else 4,
            num_stages=2,
        )
        return True
    except Exception:
        _mark_failed("gelu")
        return False


def _attention_config(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
) -> Tuple[int, int, int, int]:
    """Hand-tuned launch family for the fourteen supplied shape regimes."""
    # Large head dimensions are register-heavy; smaller row tiles keep the
    # online accumulator resident without spilling.
    if head_dim >= 256:
        block_m = 16
        block_n = 64
        warps = 8
    elif head_dim >= 128:
        block_m = 32 if seq_len >= 1024 else 16
        block_n = 64
        warps = 8
    elif head_dim >= 64:
        block_m = 32 if seq_len <= 1024 else 16
        block_n = 128
        warps = 8 if seq_len >= 1024 else 4
    elif head_dim >= 32:
        block_m = 64 if batch >= 128 and seq_len <= 128 else 32
        block_n = 128
        warps = 8 if block_m >= 64 else 4
    else:
        block_m = 64 if batch >= 128 and seq_len <= 128 else 32
        block_n = 128
        warps = 4

    # A short sequence should use one or two tiles, reducing loop and launch
    # overhead.  These values remain powers of two for Triton's arange.
    if seq_len <= 32:
        block_m = 32
        block_n = 32
        warps = min(warps, 4)
    elif seq_len <= 128 and head_dim <= 64:
        block_n = 128

    # The 100k-token case must be memory-bounded.  Larger row tiles amortize
    # the long key scan while avoiding the score matrix entirely.
    if seq_len >= 8192:
        block_m = 32 if head_dim <= 64 else 16
        block_n = 128
        warps = 8 if head_dim >= 64 else 4

    stages = 3 if seq_len >= 1024 else 2
    return block_m, block_n, warps, stages


def _launch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    mask: torch.Tensor,
    causal: bool,
) -> bool:
    if not _can_use_triton(q, "attention"):
        return False
    head_dim = q.shape[-1]
    if head_dim > 256:
        return False
    kernels = _get_kernels()
    if kernels is None:
        return False
    triton = kernels["triton"]

    batch, heads, seq_len, _ = q.shape
    block_m, block_n, warps, stages = _attention_config(
        batch, heads, seq_len, head_dim
    )
    block_d = _next_power_of_two(head_dim)
    grid = (triton.cdiv(seq_len, block_m), heads, batch)
    try:
        kernels["attention"][grid](
            q,
            k,
            v,
            mask,
            output,
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
            mask.stride(0),
            mask.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            seq_len,
            head_dim,
            1.0 / math.sqrt(head_dim),
            causal=causal,
            # SINGLE_TILE is only valid when one program owns the complete
            # query and key ranges.  For seq_len=128, BLOCK_M is often 32,
            # so even though BLOCK_N is 128 that program still needs several
            # key tiles (especially for causal rows).
            SINGLE_TILE=seq_len <= block_n and seq_len <= block_m,
            LOOP_STAGES=stages,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=warps,
            num_stages=stages,
        )
        return True
    except Exception:
        _mark_failed("attention")
        return False


# ---------------------------------------------------------------------------
# Attention and elementwise dispatch
# ---------------------------------------------------------------------------


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
    causal: bool,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch, _, seq_len, head_dim = q.shape
    mask = _normalise_mask(valid_token_mask, batch, seq_len, q.device)

    # The kernel uses only strides for the output, so callers can provide a
    # [B,H,S,D] view backed by a contiguous [B,S,H,D] allocation.  This lets
    # the output projection consume the result without a transpose copy.
    if output is None:
        output = torch.empty_like(q)
    if _launch_attention(q, k, v, output, mask, causal):
        return output

    reference = _reference_attention(q, k, v, mask, causal)
    output.copy_(reference)
    return output


def _layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    triton_result = _launch_layer_norm(x, weight, bias, mask, eps)
    if triton_result is not None:
        return triton_result
    # Use the same biased variance convention as nn.LayerNorm.
    y = F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
    return y.masked_fill(~mask[..., None], 0)


def _residual_layer_norm(
    x: torch.Tensor,
    add: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    triton_result = _launch_residual_layer_norm(
        x, add, weight, bias, mask, eps
    )
    if triton_result is not None:
        return triton_result

    # The fallback intentionally materializes the same intermediate ordering
    # as the reference block.
    x.add_(add)
    x.masked_fill_(~mask[..., None], 0)
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps).masked_fill(
        ~mask[..., None], 0
    )


def _gelu_inplace(x: torch.Tensor) -> None:
    if _launch_gelu(x):
        return
    # The benchmark asks for exact GELU, not the tanh approximation.
    x.copy_(F.gelu(x, approximate="none"))


def _add_residual_inplace(
    x: torch.Tensor,
    add: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if _launch_add_residual(x, add, mask):
        return
    x.add_(add)
    x.masked_fill_(~mask[..., None], 0)


# ---------------------------------------------------------------------------
# Drop-in Transformer modules
# ---------------------------------------------------------------------------


class UserOptimizedSelfAttention(nn.Module):
    """Benchmark-compatible self-attention with fused Triton internals."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        # Preserve the exact parameter names used by the supplied benchmark.
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self._qkv_weight_cache: Optional[torch.Tensor] = None
        self._qkv_bias_cache: Optional[torch.Tensor] = None
        self._qkv_weight_versions: Optional[Tuple[int, ...]] = None

    def _get_qkv_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        versions = (
            self.q_proj.weight._version,
            self.k_proj.weight._version,
            self.v_proj.weight._version,
            self.q_proj.bias._version,
            self.k_proj.bias._version,
            self.v_proj.bias._version,
        )
        cache_valid = (
            self._qkv_weight_cache is not None
            and self._qkv_bias_cache is not None
            and self._qkv_weight_versions == versions
            and self._qkv_weight_cache.device == self.q_proj.weight.device
            and self._qkv_weight_cache.dtype == self.q_proj.weight.dtype
        )
        if not cache_valid:
            # Inference-only benchmark: keeping a single concatenated weight
            # avoids three independent GEMM launches.  The cache is not a
            # Parameter, so state_dict compatibility is unchanged.
            with torch.no_grad():
                self._qkv_weight_cache = torch.cat(
                    [
                        self.q_proj.weight,
                        self.k_proj.weight,
                        self.v_proj.weight,
                    ],
                    dim=0,
                ).contiguous()
                self._qkv_bias_cache = torch.cat(
                    [
                        self.q_proj.bias,
                        self.k_proj.bias,
                        self.v_proj.bias,
                    ],
                    dim=0,
                ).contiguous()
            self._qkv_weight_versions = versions
        return self._qkv_weight_cache, self._qkv_bias_cache

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reference-compatible helper used by the failure microscope.

        The production forward keeps Q/K/V as strided views to avoid three
        copies.  The diagnostic intentionally calls this helper directly, so
        retain the benchmark's public internal layout contract here.
        """
        batch, seq_len, _ = x.shape
        return (
            x.view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def _native_forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
        path: str,
    ) -> torch.Tensor:
        """Native path used where T4 Triton tiles lose to PyTorch kernels."""
        batch, seq_len, _ = x.shape

        if path == NATIVE_EXACT:
            # Keep the benchmark's three projection and head-split boundaries
            # for the strict short-sequence path.
            q = self._split_heads(self.q_proj(x))
            k = self._split_heads(self.k_proj(x))
            v = self._split_heads(self.v_proj(x))
            context = _reference_attention(
                q, k, v, valid_token_mask, causal
            )
        else:
            # One GEMM for QKV is beneficial for the longer native-SDPA path.
            # SDPA accepts these strided views and can select its CUDA backend.
            qkv_weight, qkv_bias = self._get_qkv_cache()
            qkv = F.linear(
                x.reshape(batch * seq_len, self.d_model),
                qkv_weight,
                qkv_bias,
            ).view(batch, seq_len, 3, self.num_heads, self.head_dim)
            q = qkv[:, :, 0].permute(0, 2, 1, 3)
            k = qkv[:, :, 1].permute(0, 2, 1, 3)
            v = qkv[:, :, 2].permute(0, 2, 1, 3)

            if path == NATIVE_SDPA and valid_token_mask is None:
                context = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=causal,
                )
            else:
                # Arbitrary padding masks use the exact benchmark formula.
                context = _reference_attention(
                    q, k, v, valid_token_mask, causal
                )

        context = context.transpose(1, 2).contiguous()
        context = context.view(batch, seq_len, self.d_model)
        output = self.out_proj(context)
        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output

    def forward_path(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
        path: str,
    ) -> torch.Tensor:
        if path in (NATIVE_EXACT, NATIVE_SDPA):
            return self._native_forward(x, valid_token_mask, causal, path)
        return self._triton_forward(x, valid_token_mask, causal)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Run the dispatcher-selected attention path.

        The Transformer calls ``forward_path`` directly so it can reuse the
        path selected once for the whole model call.  Direct callers (such as
        the benchmark's one-layer microscope) arrive here instead, so they
        must make the same selection rather than silently defaulting to the
        Triton implementation.
        """
        batch, seq_len, _ = x.shape
        mask = _normalise_mask(valid_token_mask, batch, seq_len, x.device)
        all_valid = mask is None or bool(torch.all(mask).item())
        path = select_path(
            batch_size=batch,
            seq_len=seq_len,
            d_model=self.d_model,
            num_heads=self.num_heads,
            causal=causal,
            device=x.device,
            dtype=x.dtype,
            all_valid=all_valid,
        )

        if path in (TRITON_FLASH, TRITON_STREAM):
            if mask is None:
                mask = _normalise_mask(None, batch, seq_len, x.device)
        elif all_valid:
            # The exact native path should have the same mask-free boundary as
            # the model forward.  An all-true mask has identical mathematics,
            # but omitting it also keeps the diagnostic boundary identical.
            mask = None

        return self.forward_path(x, mask, causal, path)

    def _triton_forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        mask = _normalise_mask(valid_token_mask, batch, seq_len, x.device)

        # Keep the same three projection and head-split boundaries as the
        # reference on the precision-sensitive Triton path.  The fused QKV
        # cache remains available to the native SDPA path, but a single large
        # GEMM can choose a different accumulation/tiling order on T4 and
        # needlessly perturb Q/K/V before attention begins.
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        # Output projection wants [B,S,D].  Allocate that layout and expose a
        # [B,H,S,D] view to the attention kernel; no attention-sized transpose
        # copy is needed after the kernel returns.
        context_bshd = torch.empty(
            (batch, seq_len, self.num_heads, self.head_dim),
            device=x.device,
            dtype=x.dtype,
        )
        context_bhsd = context_bshd.permute(0, 2, 1, 3)
        _attention(q, k, v, mask, causal, output=context_bhsd)
        context = context_bshd.view(batch * seq_len, self.d_model)

        output = self.out_proj(context).view(batch, seq_len, self.d_model)
        # The projection bias makes padded rows non-zero even though their
        # context is zero.  Zero them in-place; output is a fresh GEMM result.
        output.masked_fill_(~mask[..., None], 0)
        return output


class UserOptimizedTransformerBlock(nn.Module):
    """Pre-LN Transformer block with fused residual boundaries."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = UserOptimizedSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
        path: Optional[str] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        mask = (
            None
            if valid_token_mask is None
            else _normalise_mask(valid_token_mask, batch, seq_len, x.device)
        )

        if path is None:
            # Direct block calls from the benchmark microscope do not carry
            # the model-level path selected in UserOptimizedTransformer.
            # Select it here so those calls do not silently default to the
            # numerically different Triton attention path.
            all_valid = mask is None or bool(torch.all(mask).item())
            path = select_path(
                batch_size=batch,
                seq_len=seq_len,
                d_model=x.shape[-1],
                num_heads=self.attention.num_heads,
                ffn_dim=self.ffn_out.in_features,
                causal=causal,
                device=x.device,
                dtype=x.dtype,
                all_valid=all_valid,
            )
            if path in (TRITON_FLASH, TRITON_STREAM) and mask is None:
                mask = _normalise_mask(None, batch, seq_len, x.device)
            elif all_valid:
                mask = None

        # Native LayerNorm/GELU and out-of-place residuals are intentionally
        # used on every model path.  The T4 measurements showed that the
        # custom reductions were slower and their small rounding differences
        # accumulated across four blocks under the strict gate.
        normed = F.layer_norm(
            x,
            (x.shape[-1],),
            self.norm1.weight,
            self.norm1.bias,
            self.norm1.eps,
        )
        attn = self.attention.forward_path(normed, mask, causal, path)
        post_attention = x + attn

        normed = F.layer_norm(
            post_attention,
            (post_attention.shape[-1],),
            self.norm2.weight,
            self.norm2.bias,
            self.norm2.eps,
        )
        hidden = F.gelu(
            self.ffn_in(normed),
            approximate="none",
        )
        output = post_attention + self.ffn_out(hidden)
        if mask is not None:
            output = output.masked_fill(~mask[..., None], 0)
        return output


class UserOptimizedTransformer(nn.Module):
    """Drop-in replacement for the benchmark's UserOptimizedTransformer."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                UserOptimizedTransformerBlock(
                    config.d_model,
                    config.num_heads,
                    config.ffn_dim,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected x with shape [B,S,D], got {tuple(x.shape)}")

        if valid_token_mask is None:
            mask = None
            all_valid = True
        else:
            mask = _normalise_mask(
                valid_token_mask,
                x.shape[0],
                x.shape[1],
                x.device,
            )
            # The supplied performance cases pass an all-true mask.  Detect it
            # once per model call so native SDPA can keep its fast mask-free
            # backend; padded cases retain the exact masked path.
            all_valid = bool(torch.all(mask).item())

        path = select_path(
            batch_size=x.shape[0],
            seq_len=x.shape[1],
            d_model=x.shape[2],
            num_heads=self.config.num_heads,
            ffn_dim=self.config.ffn_dim,
            num_layers=self.config.num_layers,
            causal=self.config.causal,
            device=x.device,
            dtype=x.dtype,
            all_valid=all_valid,
        )
        self._last_path = path

        if path in (TRITON_FLASH, TRITON_STREAM):
            # The Triton residual helper is not used by the default block
            # implementation anymore, but keep its inputs contiguous for the
            # attention kernel and any forced custom path.
            if not x.is_contiguous():
                x = x.contiguous()
            if mask is None:
                # Triton uses the pointer for both key and query validity;
                # allocate this once rather than once per layer.
                mask = _normalise_mask(
                    None,
                    x.shape[0],
                    x.shape[1],
                    x.device,
                )

        # Native paths can leave the caller's input untouched.  Triton
        # attention writes only to fresh output storage, so no public-boundary
        # clone is required here either.
        for layer in self.layers:
            if path in (TRITON_FLASH, TRITON_STREAM):
                layer_mask = mask
            else:
                layer_mask = None if all_valid else mask
            x = layer(x, layer_mask, self.config.causal, path)

        x = F.layer_norm(
            x,
            (x.shape[-1],),
            self.final_norm.weight,
            self.final_norm.bias,
            self.final_norm.eps,
        )
        if not all_valid:
            x = x.masked_fill(~mask[..., None], 0)
        return x


__all__ = [
    "UserOptimizedSelfAttention",
    "UserOptimizedTransformerBlock",
    "UserOptimizedTransformer",
]
