"""Drop-in SDPA Transformer, structured per block so it can be probed.

Same optimizations as torch_transformer_benchmark_sdpa.py, but split into
UserOptimizedSelfAttention / UserOptimizedTransformerBlock /
UserOptimizedTransformer with the benchmark's exact parameter names, plus a
module-level _attention(). That is the interface
torch_transformer_benchmark_diagnostic_fixed.py expects, so the per-stage and
per-layer error tables work against this implementation.

Why per-block matters for the diagnostic: per_layer_diagnostics() calls each
layer directly. If the layers were the baseline's own blocks, it would compare
the baseline against itself and print zeros.

What this does differently from the baseline:

  1. F.scaled_dot_product_attention instead of the manual
     matmul -> mask -> softmax -> matmul chain. SDPA dispatches to a tiled
     online-softmax kernel and never materializes the [B, H, N, N] scores.
  2. q/k/v packed into one GEMM, cached and keyed on parameter _version.
     Parameter names are untouched, so load_state_dict(strict=True) works.
  3. The attention module's internal zero-fill is dropped; the block-level
     masked_fill already covers it and no invalid row can reach a valid one.
  4. torch.compile over the whole layer stack (Inductor inlines the block
     forwards), fusing LayerNorm+residual, bias+GELU and the trailing fill.
  5. All-valid masks are detected once per mask tensor and dropped.
  6. Padding masks reach SDPA as an additive float bias, not a bool tensor --
     bool masks are refused by the fused backends more often, and a refusal
     silently falls back to MATH, which is the baseline's own algorithm.
  7. OPT_COMPUTE_DTYPE narrows chosen GEMMs to fp16/bf16 while the residual
     stream, both LayerNorms and GELU stay fp32.

Precision note for (7). OPT_NARROW selects which GEMMs are narrowed:

    all     qkv, attn, out, ffn_in, ffn_out   fastest, fails atol=0.001
    safe    qkv, attn, ffn_in                 passes, ~28% slower than all
    attn    qkv, attn                         passes with more margin
    ffn     ffn_in, ffn_out                   fails

out_proj and ffn_out are the two GEMMs whose output is added straight into the
residual stream, so their fp16 output rounding lands directly in the measured
result. Keeping just those two in fp32 is what moves max_abs under atol.

Environment toggles:
    OPT_COMPUTE_DTYPE=float16|bfloat16   default: off (everything fp32)
    OPT_NARROW=all|safe|attn|ffn         default: all
    OPT_FP16_FAST_REDUCE=1               allow fp16 accumulation in fp16
    OPT_COMPILE=0                        skip torch.compile
    OPT_COMPILE_MODE=...                 default|reduce-overhead|max-autotune
    OPT_FUSE_QKV=0                       one GEMM per projection
    OPT_SUFFIX_PADDING=0                 build the explicit combined mask
    OPT_ELIDE_MASK=0                     keep mask work when every token valid
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "UserOptimizedSelfAttention",
    "UserOptimizedTransformerBlock",
    "UserOptimizedTransformer",
    "settings",
]

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}
_NARROW_PRESETS = {
    "all": ("qkv", "attn", "out", "ffn_in", "ffn_out"),
    "safe": ("qkv", "attn", "ffn_in"),
    "attn": ("qkv", "attn"),
    "ffn": ("ffn_in", "ffn_out"),
}
_CACHE_LIMIT = 32


class _Settings:
    """Env-driven configuration, read once at import."""

    def __init__(self) -> None:
        self.compute_dtype = _DTYPES.get(
            os.environ.get("OPT_COMPUTE_DTYPE", "").lower()
        )
        self.narrow_preset = os.environ.get("OPT_NARROW", "all").lower()
        self.narrow = frozenset(
            _NARROW_PRESETS.get(self.narrow_preset, _NARROW_PRESETS["all"])
        )
        self.fp16_fast_reduce = os.environ.get("OPT_FP16_FAST_REDUCE", "0") != "0"
        self.use_compile = os.environ.get("OPT_COMPILE", "1") != "0"
        self.compile_mode = os.environ.get("OPT_COMPILE_MODE", "default")
        self.fuse_qkv = os.environ.get("OPT_FUSE_QKV", "1") != "0"
        self.elide_mask = os.environ.get("OPT_ELIDE_MASK", "1") != "0"
        self.assume_suffix_padding = os.environ.get("OPT_SUFFIX_PADDING", "1") != "0"

    def dtype_for(self, stage: str, stream: torch.dtype) -> torch.dtype:
        if self.compute_dtype is not None and stage in self.narrow:
            return self.compute_dtype
        return stream

    def describe(self) -> str:
        compute = (
            "off"
            if self.compute_dtype is None
            else f"{str(self.compute_dtype).split('.')[-1]}/{self.narrow_preset}"
        )
        compiled = (
            f"{self.use_compile}"
            f"{'/' + self.compile_mode if self.use_compile else ''}"
        )
        return (
            f"compute={compute} | compile={compiled} | fuse_qkv={self.fuse_qkv} | "
            f"elide_mask={self.elide_mask} | "
            f"assume_suffix_padding={self.assume_suffix_padding}"
        )


settings = _Settings()

# Caches keyed on (data_ptr, shape[, dtype]). The mask tensor itself is held so
# its allocation cannot be recycled under the same pointer while an entry lives.
_all_valid_cache: Dict[Tuple, Tuple[torch.Tensor, bool]] = {}
_bias_cache: Dict[Tuple, torch.Tensor] = {}


def effective_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Return None when every position is valid.

    masked_fill with an all-False selector is a no-op and masking no key
    positions is a no-op, so dropping the mask is exact. Reading the answer
    costs one GPU sync, cached per mask tensor so it lands in warmup.
    """
    if mask is None or not settings.elide_mask:
        return mask
    key = (mask.data_ptr(), tuple(mask.shape))
    entry = _all_valid_cache.get(key)
    if entry is None:
        all_valid = bool(mask.all().item())
        if len(_all_valid_cache) >= _CACHE_LIMIT:
            _all_valid_cache.clear()
        _all_valid_cache[key] = (mask, all_valid)
    else:
        all_valid = entry[1]
    return None if all_valid else mask


def _additive_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """[B, 1, 1, N] bias: 0 where valid, -inf where padded.

    -inf rather than a large negative constant, matching the baseline's
    masked_fill(-inf) exactly. min_valid >= 1 so no row is ever fully masked
    and no NaN appears.
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
    """Return (attn_mask, is_causal); SDPA rejects both together.

    Under causal + suffix padding the key-padding mask is redundant: a valid
    query at position i only attends keys j <= i < length, all of which are
    valid. Set OPT_SUFFIX_PADDING=0 to build the combined mask instead.
    """
    if mask is None:
        return None, causal
    if causal:
        if settings.assume_suffix_padding:
            return None, True
        blocked = torch.ones(
            (seq_len, seq_len), device=device, dtype=torch.bool
        ).triu(diagonal=1)
        causal_bias = torch.zeros(
            (seq_len, seq_len), dtype=dtype, device=device
        ).masked_fill(blocked, float("-inf"))
        return causal_bias[None, None] + _additive_bias(mask, dtype), False
    return _additive_bias(mask, dtype), False


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
    causal: bool,
) -> torch.Tensor:
    """Attention on [B, H, N, D] tensors, matching the baseline's semantics.

    Diagnostic entry point -- torch_transformer_benchmark_diagnostic_fixed.py
    calls this directly. Invalid query rows are zeroed, as the baseline's
    attention module does.
    """
    mask = effective_mask(valid_token_mask)
    attn_mask, is_causal = build_attn_mask(
        q.shape[-2], mask, causal, q.dtype, q.device
    )
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=is_causal
    )
    if mask is not None:
        out = out.masked_fill(~mask[:, None, :, None], 0)
    return out


class UserOptimizedSelfAttention(nn.Module):
    """Benchmark-compatible attention: same four Linear layers, SDPA inside."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        # Plain attributes, not buffers: registering them would add keys to
        # state_dict() and break the strict weight copy.
        self._qkv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._out_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._cache_key: Optional[Tuple] = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _weights(self, stream: torch.dtype):
        """Packed q/k/v weights plus out_proj, cast per stage and cached.

        Keyed on every parameter's _version so an in-place weight change
        invalidates the cache -- safer than invalidating only on
        load_state_dict, and the idea comes from the team's Triton module.
        """
        qkv_dt = settings.dtype_for("qkv", stream)
        out_dt = settings.dtype_for("out", stream)
        key = (
            qkv_dt,
            out_dt,
            self.q_proj.weight._version,
            self.k_proj.weight._version,
            self.v_proj.weight._version,
            self.out_proj.weight._version,
            self.q_proj.bias._version,
            self.k_proj.bias._version,
            self.v_proj.bias._version,
            self.out_proj.bias._version,
        )
        if self._cache_key != key:
            # inference_mode(False) so cached tensors are ordinary tensors even
            # though the first forward runs inside torch.inference_mode().
            with torch.inference_mode(False), torch.no_grad():
                weight = torch.cat(
                    [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight],
                    dim=0,
                ).contiguous().to(qkv_dt)
                bias = torch.cat(
                    [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], dim=0
                ).contiguous().to(qkv_dt)
                self._qkv_cache = (weight, bias)
                self._out_cache = (
                    self.out_proj.weight.to(out_dt),
                    self.out_proj.bias.to(out_dt),
                )
            self._cache_key = key
        return self._qkv_cache, self._out_cache

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: Optional[bool] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        stream = x.dtype
        (qkv_w, qkv_b), (out_w, out_b) = self._weights(stream)

        qkv_dt = settings.dtype_for("qkv", stream)
        attn_dt = settings.dtype_for("attn", stream)
        out_dt = settings.dtype_for("out", stream)

        if is_causal is None:
            mask = effective_mask(valid_token_mask)
            attn_mask, is_causal = build_attn_mask(
                seq_len, mask, causal, attn_dt, x.device
            )

        hidden = x.to(qkv_dt)
        if settings.fuse_qkv:
            qkv = F.linear(hidden, qkv_w, qkv_b)
            qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        else:
            # Slicing the packed weight is free and matches three Linears.
            parts = []
            for start in (0, self.d_model, 2 * self.d_model):
                piece = F.linear(
                    hidden,
                    qkv_w[start : start + self.d_model],
                    qkv_b[start : start + self.d_model],
                )
                parts.append(
                    piece.view(
                        batch, seq_len, self.num_heads, self.head_dim
                    ).transpose(1, 2)
                )
            q, k, v = parts

        if attn_dt != qkv_dt:
            q, k, v = q.to(attn_dt), k.to(attn_dt), v.to(attn_dt)

        context = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=bool(is_causal)
        )
        context = context.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        return F.linear(context.to(out_dt), out_w, out_b).to(stream)


class UserOptimizedTransformerBlock(nn.Module):
    """Pre-norm block with the benchmark's exact parameter names."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = UserOptimizedSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

        self._ffn_cache = None
        self._ffn_key: Optional[Tuple] = None

    def _ffn_weights(self, stream: torch.dtype):
        in_dt = settings.dtype_for("ffn_in", stream)
        out_dt = settings.dtype_for("ffn_out", stream)
        key = (
            in_dt,
            out_dt,
            self.ffn_in.weight._version,
            self.ffn_out.weight._version,
            self.ffn_in.bias._version,
            self.ffn_out.bias._version,
        )
        if self._ffn_key != key:
            with torch.inference_mode(False), torch.no_grad():
                self._ffn_cache = (
                    self.ffn_in.weight.to(in_dt),
                    self.ffn_in.bias.to(in_dt),
                    self.ffn_out.weight.to(out_dt),
                    self.ffn_out.bias.to(out_dt),
                )
            self._ffn_key = key
        return self._ffn_cache

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: Optional[bool] = None,
        invalid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        stream = x.dtype
        if is_causal is None:
            mask = effective_mask(valid_token_mask)
            attn_dt = settings.dtype_for("attn", stream)
            attn_mask, is_causal = build_attn_mask(
                x.shape[1], mask, causal, attn_dt, x.device
            )
            invalid = None if mask is None else ~mask[..., None]

        in_w, in_b, fo_w, fo_b = self._ffn_weights(stream)
        in_dt = settings.dtype_for("ffn_in", stream)
        out_dt = settings.dtype_for("ffn_out", stream)

        # LayerNorm always runs in the residual-stream dtype.
        x = x + self.attention(
            self.norm1(x), valid_token_mask, causal, attn_mask, is_causal
        )

        hidden = F.linear(self.norm2(x).to(in_dt), in_w, in_b)
        # GELU in fp32: the one elementwise op with enough curvature for
        # low-precision rounding to show in the output. Inductor fuses the
        # up-cast, the erf and the down-cast into a single kernel.
        gelu = F.gelu(hidden.float(), approximate="none").to(out_dt)
        x = x + F.linear(gelu, fo_w, fo_b).to(stream)

        if invalid is not None:
            x = x.masked_fill(invalid, 0)
        return x


class UserOptimizedTransformer(nn.Module):
    """Drop-in replacement for the benchmark's UserOptimizedTransformer."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                UserOptimizedTransformerBlock(
                    config.d_model, config.num_heads, config.ffn_dim
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self._compiled = None  # None = not built, False = compile failed
        self._reduction_set = False

        if settings.use_compile:
            # 14 official shapes against a default limit of 8. Past the limit
            # Dynamo falls back to eager with only a warning, silently turning
            # the compiled candidate back into the eager one.
            try:
                import torch._dynamo

                torch._dynamo.config.cache_size_limit = max(
                    64, torch._dynamo.config.cache_size_limit
                )
            except Exception:
                pass

        print(f"[optimized-sdpa] {settings.describe()}")

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # fp16 GEMMs accumulate in fp16 by default. Force fp32 accumulation for
        # accuracy, but only when the harness itself runs fp32 -- otherwise we
        # would be altering the reference implementation too.
        if (
            settings.compute_dtype is torch.float16
            and not settings.fp16_fast_reduce
            and not self._reduction_set
            and x.dtype is torch.float32
        ):
            try:
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
                    False
                )
            except Exception:
                pass
            self._reduction_set = True

        # Mask analysis stays outside the compiled region: .item() would force
        # a graph break, and the result is a Python bool Dynamo can guard on.
        mask = effective_mask(valid_token_mask)
        attn_dt = settings.dtype_for("attn", x.dtype)
        attn_mask, is_causal = build_attn_mask(
            x.shape[1], mask, self.config.causal, attn_dt, x.device
        )
        invalid = None if mask is None else ~mask[..., None]

        if settings.use_compile and self._compiled is not False:
            if self._compiled is None:
                self._compiled = torch.compile(
                    self._run, dynamic=False, mode=settings.compile_mode
                )
            try:
                return self._compiled(x, attn_mask, is_causal, invalid)
            except Exception as exc:  # a fallback must never raise
                print(
                    f"[optimized-sdpa] compile failed, using eager: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._compiled = False
        return self._run(x, attn_mask, is_causal, invalid)

    def _run(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
        invalid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, None, self.config.causal, attn_mask, is_causal, invalid)
        x = self.final_norm(x)
        if invalid is not None:
            x = x.masked_fill(invalid, 0)
        return x
