"""Drop-in optimized Transformer with PyTorch GEMMs at the precision-sensitive boundaries.

For manageable benchmark shapes, this version deliberately uses PyTorch for both
large matrix multiplies in attention:

    Q @ K^T  -> PyTorch matmul
    softmax  -> Triton FP32 row-wise kernel
    P @ V    -> PyTorch matmul

This keeps the benchmark's Q/K/V dtype and its explicit FP32 softmax boundary,
while removing the two custom Triton GEMM reduction orders that were producing
the first numerical divergence in our diagnostics.

For extremely long sequences where an NxN score/probability matrix is impossible
to materialize (for example N=100000), the implementation falls back to a
streaming Triton attention kernel.
"""

from __future__ import annotations

import importlib.util
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Triton availability
# ---------------------------------------------------------------------------


def _triton_is_available() -> bool:
    """Return True when the Triton Python package is installed."""
    return importlib.util.find_spec("triton") is not None


# ---------------------------------------------------------------------------
# Exact benchmark attention fallback
# ---------------------------------------------------------------------------


def _reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
    causal: bool,
) -> torch.Tensor:
    """Exactly mirror the benchmark's original attention implementation."""
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

    # The benchmark explicitly performs softmax in FP32 and then converts
    # probabilities back to the model dtype before P @ V.
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


# We only materialize P for reasonably sized attention matrices.  This keeps
# the precision experiment safe while preserving the streaming path for very
# long inputs such as N=100000.
_MAX_MATERIALIZED_PROBS = 100_000_000


def _get_triton_kernels():
    """Create and return the Triton kernels lazily."""
    if not _triton_is_available():
        return None, None

    import triton
    import triton.language as tl

    @triton.jit
    def softmax_probs_kernel(
        scores_ptr,
        mask_ptr,
        p_ptr,
        stride_sb,
        stride_sh,
        stride_sm,
        stride_sn,
        stride_mb,
        stride_mn,
        stride_pb,
        stride_ph,
        stride_pm,
        stride_pn,
        n_ctx,
        causal: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Apply the benchmark's FP32 softmax to one score row at a time.

        The score matrix itself is produced by PyTorch's matmul. This kernel only
        applies causal/padding masks and performs the numerically sensitive softmax
        reduction in FP32, then stores probabilities in the model dtype.
        """
        pid_n = tl.program_id(0)
        head = tl.program_id(1)
        batch = tl.program_id(2)

        row = pid_n
        cols = tl.arange(0, BLOCK_N)
        col_valid = cols < n_ctx
        row_valid = row < n_ctx

        scores = tl.load(
            scores_ptr
            + batch * stride_sb
            + head * stride_sh
            + row * stride_sm
            + cols * stride_sn,
            mask=col_valid,
            other=float("-inf"),
        ).to(tl.float32)

        key_valid = tl.load(
            mask_ptr + batch * stride_mb + cols * stride_mn,
            mask=col_valid,
            other=0,
        ).to(tl.int1)

        allowed = col_valid & key_valid
        if causal:
            allowed = allowed & (cols <= row)

        scores = tl.where(allowed, scores, float("-inf"))

        row_max = tl.max(scores, axis=0)
        exp_scores = tl.where(allowed, tl.exp(scores - row_max), 0.0)
        row_sum = tl.sum(exp_scores, axis=0)

        probs = tl.where(
            allowed & (row_sum > 0),
            exp_scores / row_sum,
            0.0,
        )

        tl.store(
            p_ptr
            + batch * stride_pb
            + head * stride_ph
            + row * stride_pm
            + cols * stride_pn,
            probs.to(tl.float16),
            mask=col_valid & row_valid,
        )

    @triton.jit
    def streaming_attention_kernel(
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
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Streaming online-softmax attention for very large N.

        This is retained only for inputs where materializing P would be too
        memory-intensive.  The running softmax state stays in FP32.
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

        m = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        l = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        for start_n in range(0, n_ctx, BLOCK_N):
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

            scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale

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
            new_m = tl.maximum(m, tile_max)

            old_scale = tl.exp(m - new_m)
            p = tl.where(
                allowed,
                tl.exp(scores - new_m[:, None]),
                0.0,
            )

            l = l * old_scale + tl.sum(p, axis=1)

            # Triton requires both dot operands to have the same dtype.
            # Keep the online-softmax state in FP32, but cast this probability
            # tile back to the model dtype before multiplying by V.
            p_native = p.to(v.dtype)
            pv = tl.dot(
                p_native,
                v,
                out_dtype=tl.float32,
                input_precision="ieee",
            )
            acc = acc * old_scale[:, None] + pv
            m = new_m

        output = tl.where(
            l[:, None] > 0,
            acc / l[:, None],
            0.0,
        )
        output = tl.where(query_valid[:, None], output, 0.0)

        tl.store(
            o_ptr
            + batch * stride_ob
            + head * stride_oh
            + rows[:, None] * stride_om
            + dims[None, :] * stride_od,
            output.to(tl.float16),
            mask=row_valid[:, None] & dim_valid[None, :],
        )

    return softmax_probs_kernel, streaming_attention_kernel


_PROB_KERNEL = None
_STREAM_KERNEL = None


def _get_kernels():
    global _PROB_KERNEL, _STREAM_KERNEL
    if _PROB_KERNEL is None or _STREAM_KERNEL is None:
        _PROB_KERNEL, _STREAM_KERNEL = _get_triton_kernels()
    return _PROB_KERNEL, _STREAM_KERNEL


def _launch_probability_kernel(
    scores: torch.Tensor,
    valid_token_mask: torch.Tensor,
    causal: bool,
    block_n: int,
) -> torch.Tensor:
    """Launch Triton softmax on a score matrix created by PyTorch matmul."""
    import triton

    softmax_kernel, _ = _get_kernels()
    if softmax_kernel is None:
        raise RuntimeError("Triton is not available")

    batch, heads, seq_len, _ = scores.shape
    p = torch.empty_like(scores)
    grid = (seq_len, heads, batch)

    softmax_kernel[grid](
        scores,
        valid_token_mask,
        p,
        scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
        valid_token_mask.stride(0), valid_token_mask.stride(1),
        p.stride(0), p.stride(1), p.stride(2), p.stride(3),
        seq_len,
        causal=causal,
        BLOCK_N=triton.next_power_of_2(seq_len),
        num_warps=4 if seq_len <= 256 else 8,
        num_stages=2,
    )
    return p


def _launch_streaming_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: torch.Tensor,
    causal: bool,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    """Launch the large-N streaming attention path."""
    import triton

    _, stream_kernel = _get_kernels()
    if stream_kernel is None:
        raise RuntimeError("Triton is not available")

    batch, heads, seq_len, head_dim = q.shape
    output = torch.empty_like(q)
    block_d = triton.next_power_of_2(head_dim)
    grid = (triton.cdiv(seq_len, block_m), heads, batch)

    stream_kernel[grid](
        q,
        k,
        v,
        valid_token_mask,
        output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        valid_token_mask.stride(0), valid_token_mask.stride(1),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        seq_len,
        head_dim,
        1.0 / math.sqrt(head_dim),
        causal=causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
    causal: bool,
) -> torch.Tensor:
    """Shape-aware attention that leaves GEMM to PyTorch when practical.

    The optimized work here is the Triton masked/FP32 softmax kernel.  QK^T
    and P@V are deliberately kept as PyTorch matmuls for numerical fidelity
    and because the benchmark does not give us evidence that a custom GEMM is
    faster.  The separate streaming path remains for extremely large N.
    """
    if valid_token_mask is None:
        valid_token_mask = torch.ones(
            q.shape[0], q.shape[2], device=q.device, dtype=torch.bool
        )
    else:
        valid_token_mask = valid_token_mask.to(
            device=q.device, dtype=torch.bool
        ).contiguous()

    if (
        not q.is_cuda
        or q.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or not _triton_is_available()
    ):
        return _reference_attention(q, k, v, valid_token_mask, causal)

    batch, heads, seq_len, head_dim = q.shape
    if head_dim > 256:
        return _reference_attention(q, k, v, valid_token_mask, causal)

    score_elements = batch * heads * seq_len * seq_len

    # For manageable matrices, let Triton compute the attention probabilities
    # and let PyTorch perform exactly the P @ V GEMM used by the benchmark.
    if score_elements <= _MAX_MATERIALIZED_PROBS:
        block_m = 32 if seq_len >= 128 else 16
        block_n = 64 if seq_len >= 64 else 32
        # Use the same PyTorch GEMM implementation as the benchmark for Q @ K^T.
        # This removes the custom Triton reduction order from the first matmul.
        scores = torch.matmul(
            q, k.transpose(-2, -1)
        ) * (1.0 / math.sqrt(head_dim))

        # Triton handles the row-wise FP32 softmax + masking, but does not perform
        # either matrix multiply for the manageable shapes.
        probs = _launch_probability_kernel(
            scores, valid_token_mask, causal, block_n
        )

        # Use PyTorch's matmul for P @ V as well, matching the benchmark.
        output = torch.matmul(probs, v)
        output = output.masked_fill(
            ~valid_token_mask[:, None, :, None],
            0,
        )
        return output

    # Very large attention matrices are streamed so that we never allocate
    # the impossible N x N probability matrix.
    if seq_len >= 8192:
        block_m, block_n, warps, stages = 64, 128, 8, 3
    elif seq_len >= 1024:
        block_m, block_n, warps, stages = 32, 64, 4, 3
    else:
        block_m, block_n, warps, stages = 32, 64, 4, 3

    return _launch_streaming_attention(
        q, k, v, valid_token_mask, causal,
        block_m, block_n, warps, stages,
    )


# ---------------------------------------------------------------------------
# Drop-in Transformer modules
# ---------------------------------------------------------------------------


class UserOptimizedSelfAttention(nn.Module):
    """Same parameters as the benchmark, with a cached fused QKV projection.

    The benchmark stores Q, K and V as three separate Linear layers.  We keep
    those exact parameter names so the benchmark can still copy its state_dict
    into this module.  During inference, we concatenate their weights/biases
    once and perform one larger PyTorch GEMM instead of three separate GEMMs.

    This is a performance optimization only; Q/K/V remain in the model dtype
    (FP16 for the current benchmark run).
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        # These cached tensors are populated lazily after the benchmark copies
        # the weights into this model.  They are intentionally NOT Parameters:
        # the benchmark expects the original q_proj/k_proj/v_proj parameter names.
        self._qkv_weight_cache = None
        self._qkv_bias_cache = None
        self._qkv_weight_versions = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return (
            x.view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Fuse the three independent projections into one PyTorch Linear/GEMM:
        #
        #   X @ Wq  -> Q
        #   X @ Wk  -> K
        #   X @ Wv  -> V
        #
        # becomes
        #
        #   X @ [Wq; Wk; Wv] -> [Q | K | V]
        #
        # PyTorch remains responsible for the matrix multiplication, so we are
        # not replacing a mature GEMM implementation with a hand-written one.
        # The concatenated tensors are cached because weights are fixed during
        # inference; the cache is refreshed automatically if a parameter changes.
        versions = (
            self.q_proj.weight._version,
            self.k_proj.weight._version,
            self.v_proj.weight._version,
            self.q_proj.bias._version,
            self.k_proj.bias._version,
            self.v_proj.bias._version,
        )
        if self._qkv_weight_cache is None or versions != self._qkv_weight_versions:
            self._qkv_weight_cache = torch.cat(
                [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight],
                dim=0,
            )
            self._qkv_bias_cache = torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias],
                dim=0,
            )
            self._qkv_weight_versions = versions

        qkv = F.linear(
            x,
            self._qkv_weight_cache,
            self._qkv_bias_cache,
        )
        q, k, v = qkv.chunk(3, dim=-1)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        context = _attention(q, k, v, valid_token_mask, causal)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.d_model)
        )

        output = self.out_proj(context)
        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class UserOptimizedTransformerBlock(nn.Module):
    """Same pre-LN Transformer block ordering as the benchmark."""

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
    ) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), valid_token_mask, causal)
        x = x + self.ffn_out(
            F.gelu(self.ffn_in(self.norm2(x)), approximate="none")
        )
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


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
        for layer in self.layers:
            x = layer(x, valid_token_mask, self.config.causal)
        x = self.final_norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x
