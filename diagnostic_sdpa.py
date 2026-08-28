#!/usr/bin/env python3
"""
Fixed diagnostic benchmark for the supplied Transformer benchmark.

This version keeps the original benchmark's numerical rule and timing style,
but adds checkpoints so we can identify exactly where the first numerical
mismatch appears.

Diagnostics include:
  - Q/K/V equality
  - QK^T error on VALID (finite) causal/padding entries only
  - softmax error
  - P@V error using the SAME P and V
  - attention output error
  - output projection error
  - attention residual error
  - norm2 error
  - FFN-in error
  - GELU error
  - FFN-out error
  - complete Transformer-block error
  - error growth after every layer and final norm
  - full-model correctness and performance

This is a debugging script, not the final submission harness.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import math
import statistics
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import optimized_sdpa as opt_mod
from optimized_sdpa import UserOptimizedTransformer


@dataclass(frozen=True)
class TransformerConfig:
    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    ffn_dim: int
    num_layers: int
    causal: bool

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    def validate(self) -> None:
        if self.batch_size <= 0 or self.seq_len <= 0 or self.d_model <= 0:
            raise ValueError("batch_size, seq_len and d_model must be positive")
        if self.num_heads <= 0 or self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.ffn_dim <= 0 or self.num_layers <= 0:
            raise ValueError("ffn_dim and num_layers must be positive")


class BaselineSelfAttention(nn.Module):
    """Exact attention implementation from the user's benchmark."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x, valid_token_mask=None, causal=False):
        b, s, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if causal:
            causal_mask = torch.ones((s, s), device=x.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(causal_mask, float("-inf"))
        if valid_token_mask is not None:
            scores = scores.masked_fill(
                ~valid_token_mask[:, None, None, :], float("-inf")
            )

        probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).contiguous().view(b, s, self.d_model)
        output = self.out_proj(context)
        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class BaselineTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = BaselineSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(self, x, valid_token_mask, causal):
        x = x + self.attention(self.norm1(x), valid_token_mask, causal)
        x = x + self.ffn_out(
            F.gelu(self.ffn_in(self.norm2(x)), approximate="none")
        )
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class BaselineTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                BaselineTransformerBlock(
                    config.d_model, config.num_heads, config.ffn_dim
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(self, x, valid_token_mask=None):
        for layer in self.layers:
            x = layer(x, valid_token_mask, self.config.causal)
        x = self.final_norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


def copy_model_weights(baseline, optimized) -> None:
    optimized.load_state_dict(copy.deepcopy(baseline.state_dict()), strict=True)


def generate_random_case(config, device, dtype, seed, padding_ratio, input_scale):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x = torch.randn(
        config.batch_size,
        config.seq_len,
        config.d_model,
        generator=generator,
        device=device,
        dtype=dtype,
    ) * input_scale

    if padding_ratio <= 0:
        return x, torch.ones(
            (config.batch_size, config.seq_len), device=device, dtype=torch.bool
        )

    min_valid = max(1, int(round(config.seq_len * (1.0 - padding_ratio))))
    lengths = torch.randint(
        low=min_valid,
        high=config.seq_len + 1,
        size=(config.batch_size,),
        generator=generator,
        device=device,
    )
    positions = torch.arange(config.seq_len, device=device)[None, :]
    mask = positions < lengths[:, None]
    x = x.masked_fill(~mask[..., None], 0)
    return x, mask


def error_stats(reference: torch.Tensor, candidate: torch.Tensor, valid: Optional[torch.Tensor] = None) -> dict:
    """Report finite error statistics without letting -inf/-inf create NaNs."""
    ref = reference.detach().float()
    cand = candidate.detach().float()

    finite = torch.isfinite(ref) & torch.isfinite(cand)
    if valid is not None:
        finite = finite & valid

    nonfinite = int((~finite).sum().item())
    if not finite.any():
        return {
            "shape": tuple(reference.shape),
            "max_abs": float("nan"),
            "max_rel": float("nan"),
            "mean_abs": float("nan"),
            "rms": float("nan"),
            "nonfinite": nonfinite,
            "failed_atol_1e3": 0,
            "failed_bench_gate": 0,
        }

    safe_ref = ref[finite]
    safe_cand = cand[finite]
    diff = (safe_cand - safe_ref).abs()
    rel = diff / safe_ref.abs().clamp_min(1e-12)

    return {
        "shape": tuple(reference.shape),
        "max_abs": float(diff.max().item()),
        "max_rel": float(rel.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rms": float(torch.sqrt((diff * diff).mean()).item()),
        "nonfinite": nonfinite,
        "failed_atol_1e3": int((diff > 1e-3).sum().item()),
        "failed_bench_gate": int(
            ((diff > 1e-3) & (diff > 0.01 * safe_ref.abs())).sum().item()
        ),
    }


def print_stats(name: str, stats: dict) -> None:
    print(
        f"{name:32s} shape={stats['shape']} | "
        f"max_abs={stats['max_abs']:.8g} | max_rel={stats['max_rel']:.8g} | "
        f"mean_abs={stats['mean_abs']:.8g} | rms={stats['rms']:.8g} | "
        f">1e-3={stats['failed_atol_1e3']} | "
        f"nonfinite={stats['nonfinite']} | "
        f"gate_fail={stats['failed_bench_gate']}"
    )


def build_reference_attention(q, k, v, valid_mask, causal):
    scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(q.shape[-1]))
    s = q.shape[-2]
    allowed = torch.ones((q.shape[0], 1, s, s), device=q.device, dtype=torch.bool)
    if causal:
        allowed = allowed & torch.ones((s, s), device=q.device, dtype=torch.bool).tril()[None, None]
    if valid_mask is not None:
        allowed = allowed & valid_mask[:, None, None, :]
        allowed = allowed & valid_mask[:, None, :, None]
    scores = scores.masked_fill(~allowed, float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
    pv = torch.matmul(probs, v)
    return scores, probs, pv, allowed


_TRITON_DIAG = None


def get_diag_kernels():
    global _TRITON_DIAG
    if _TRITON_DIAG is not None:
        return _TRITON_DIAG
    import triton
    import triton.language as tl

    @triton.jit
    def qk_kernel(q_ptr, k_ptr, out_ptr,
                   sqb, sqh, sqm, sqd, skb, skh, skn, skd,
                   sob, soh, som, son, n_ctx, head_dim, scale,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
        pid_m = tl.program_id(0)
        head = tl.program_id(1)
        batch = tl.program_id(2)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        dims = tl.arange(0, BLOCK_D)
        row_valid = rows < n_ctx
        col_valid = cols < n_ctx
        dim_valid = dims < head_dim
        q = tl.load(
            q_ptr + batch * sqb + head * sqh + rows[:, None] * sqm + dims[None, :] * sqd,
            mask=row_valid[:, None] & dim_valid[None, :], other=0.0,
        )
        k = tl.load(
            k_ptr + batch * skb + head * skh + cols[:, None] * skn + dims[None, :] * skd,
            mask=col_valid[:, None] & dim_valid[None, :], other=0.0,
        )
        scores = tl.dot(q, tl.trans(k), input_precision="ieee", out_dtype=tl.float32) * scale
        tl.store(
            out_ptr + batch * sob + head * soh + rows[:, None] * som + cols[None, :] * son,
            scores,
            mask=row_valid[:, None] & col_valid[None, :],
        )

    @triton.jit
    def pv_kernel(p_ptr, v_ptr, out_ptr,
                  spb, sph, spm, spn, svb, svh, svn, svd,
                  sob, soh, som, sod, n_ctx, head_dim,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
        pid_m = tl.program_id(0)
        head = tl.program_id(1)
        batch = tl.program_id(2)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        dims = tl.arange(0, BLOCK_D)
        row_valid = rows < n_ctx
        dim_valid = dims < head_dim
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        for start_n in range(0, n_ctx, BLOCK_N):
            cols = start_n + tl.arange(0, BLOCK_N)
            col_valid = cols < n_ctx
            p = tl.load(
                p_ptr + batch * spb + head * sph + rows[:, None] * spm + cols[None, :] * spn,
                mask=row_valid[:, None] & col_valid[None, :], other=0.0,
            )
            v = tl.load(
                v_ptr + batch * svb + head * svh + cols[:, None] * svn + dims[None, :] * svd,
                mask=col_valid[:, None] & dim_valid[None, :], other=0.0,
            )
            acc += tl.dot(p, v, input_precision="ieee", out_dtype=tl.float32)
        tl.store(
            out_ptr + batch * sob + head * soh + rows[:, None] * som + dims[None, :] * sod,
            acc,
            mask=row_valid[:, None] & dim_valid[None, :],
        )

    _TRITON_DIAG = (qk_kernel, pv_kernel, triton)
    return _TRITON_DIAG


def triton_qk(q, k):
    qk_kernel, _, triton = get_diag_kernels()
    b, h, s, d = q.shape
    out = torch.empty((b, h, s, s), device=q.device, dtype=torch.float32)
    bm, bn, bd = 32, 32, triton.next_power_of_2(d)
    grid = (triton.cdiv(s, bm), h, b)
    qk_kernel[grid](
        q, k, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        s, d, 1.0 / math.sqrt(d), BLOCK_M=bm, BLOCK_N=bn, BLOCK_D=bd,
        num_warps=4, num_stages=2,
    )
    return out


def triton_pv(p, v):
    _, pv_kernel, triton = get_diag_kernels()
    b, h, s, d = v.shape
    out = torch.empty((b, h, s, d), device=v.device, dtype=torch.float32)
    bm, bn, bd = 32, 32, triton.next_power_of_2(d)
    grid = (triton.cdiv(s, bm), h, b)
    pv_kernel[grid](
        p, v, out,
        p.stride(0), p.stride(1), p.stride(2), p.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        s, d, BLOCK_M=bm, BLOCK_N=bn, BLOCK_D=bd,
        num_warps=4, num_stages=2,
    )
    return out


def one_layer_diagnostics(baseline, optimized, x, valid_mask, causal, atol, rtol):
    print("\n=== One-layer detailed diagnostics ===")
    s = x.shape[1]
    if s > 2048:
        print("Skipped full score-matrix diagnostics for seq_len > 2048.")
        return

    b_layer = baseline.layers[0]
    o_layer = optimized.layers[0]

    with torch.inference_mode():
        b_norm = b_layer.norm1(x)
        o_norm = o_layer.norm1(x)
        print_stats("norm1", error_stats(b_norm, o_norm))

        q = b_layer.attention._split_heads(b_layer.attention.q_proj(b_norm))
        k = b_layer.attention._split_heads(b_layer.attention.k_proj(b_norm))
        v = b_layer.attention._split_heads(b_layer.attention.v_proj(b_norm))
        print_stats("Q baseline-vs-opt", error_stats(q, o_layer.attention._split_heads(o_layer.attention.q_proj(o_norm))))
        print_stats("K baseline-vs-opt", error_stats(k, o_layer.attention._split_heads(o_layer.attention.k_proj(o_norm))))
        print_stats("V baseline-vs-opt", error_stats(v, o_layer.attention._split_heads(o_layer.attention.v_proj(o_norm))))

        scores_ref, probs_ref, pv_ref, allowed = build_reference_attention(q, k, v, valid_mask, causal)

        # QK^T / softmax / P@V are separate stages only in the Triton
        # implementation. SDPA fuses all three into one kernel and never
        # materializes the score matrix, so there is nothing to sample between
        # them -- the "attention output" row below is the measurable stage.
        try:
            scores_tri = triton_qk(q, k)
            score_valid = allowed.expand_as(scores_ref)
            print_stats("QK^T legal entries", error_stats(scores_ref, scores_tri, score_valid))

            scores_tri_masked = scores_tri.masked_fill(~allowed, float("-inf"))
            probs_tri = torch.softmax(scores_tri_masked.float(), dim=-1).to(dtype=q.dtype)
            print_stats("softmax", error_stats(probs_ref, probs_tri))

            pv_tri = triton_pv(probs_ref, v)
            print_stats("P@V same-P diagnostic", error_stats(pv_ref, pv_tri))
        except Exception as exc:
            print(f"{'QK^T / softmax / P@V':32s} skipped: fused into SDPA ({type(exc).__name__})")

        attn_ref = pv_ref
        attn_opt = opt_mod._attention(q, k, v, valid_mask, causal)
        print_stats("attention output", error_stats(attn_ref, attn_opt))

        # ---- Isolate the rest of the block operation by operation. ----
        out_ref = b_layer.attention.out_proj(attn_ref.transpose(1, 2).contiguous().view_as(x))
        out_opt = o_layer.attention.out_proj(attn_opt.transpose(1, 2).contiguous().view_as(x))
        print_stats("out_proj", error_stats(out_ref, out_opt))

        residual_ref = x + out_ref
        residual_opt = x + out_opt
        if valid_mask is not None:
            residual_ref = residual_ref.masked_fill(~valid_mask[..., None], 0)
            residual_opt = residual_opt.masked_fill(~valid_mask[..., None], 0)
        print_stats("attention residual", error_stats(residual_ref, residual_opt))

        norm2_ref = b_layer.norm2(residual_ref)
        norm2_opt = o_layer.norm2(residual_opt)
        print_stats("norm2", error_stats(norm2_ref, norm2_opt))

        ffn_in_ref = b_layer.ffn_in(norm2_ref)
        ffn_in_opt = o_layer.ffn_in(norm2_opt)
        print_stats("ffn_in", error_stats(ffn_in_ref, ffn_in_opt))

        gelu_ref = F.gelu(ffn_in_ref, approximate="none")
        gelu_opt = F.gelu(ffn_in_opt, approximate="none")
        print_stats("GELU", error_stats(gelu_ref, gelu_opt))

        ffn_out_ref = b_layer.ffn_out(gelu_ref)
        ffn_out_opt = o_layer.ffn_out(gelu_opt)
        print_stats("ffn_out", error_stats(ffn_out_ref, ffn_out_opt))

        block_ref = residual_ref + ffn_out_ref
        block_opt = residual_opt + ffn_out_opt
        if valid_mask is not None:
            block_ref = block_ref.masked_fill(~valid_mask[..., None], 0)
            block_opt = block_opt.masked_fill(~valid_mask[..., None], 0)
        print_stats("complete block", error_stats(block_ref, block_opt))


def per_layer_diagnostics(baseline, optimized, x, valid_mask, causal):
    print("\n=== Per-layer error growth ===")
    xb = x.clone()
    xo = x.clone()
    with torch.inference_mode():
        for i, (lb, lo) in enumerate(zip(baseline.layers, optimized.layers)):
            xb = lb(xb, valid_mask, causal)
            xo = lo(xo, valid_mask, causal)
            print_stats(f"after layer {i}", error_stats(xb, xo))
        xb = baseline.final_norm(xb)
        xo = optimized.final_norm(xo)
        if valid_mask is not None:
            xb = xb.masked_fill(~valid_mask[..., None], 0)
            xo = xo.masked_fill(~valid_mask[..., None], 0)
        print_stats("after final_norm", error_stats(xb, xo))


def warmup(model, x, mask, device, count=10):
    with torch.inference_mode():
        for _ in range(count):
            model(x, mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(model, x, mask, device, repeats=30):
    samples = []
    with torch.inference_mode():
        if device.type == "cuda":
            # Record all repetitions before one synchronization to avoid adding
            # a host-side sync to every individual sample.
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
            torch.cuda.synchronize(device)
            for i in range(repeats):
                starts[i].record()
                model(x, mask)
                ends[i].record()
            torch.cuda.synchronize(device)
            samples = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        else:
            for _ in range(repeats):
                t0 = time.perf_counter_ns()
                model(x, mask)
                t1 = time.perf_counter_ns()
                samples.append((t1 - t0) / 1e6)
    ordered = sorted(samples)
    p90_index = max(0, int(math.ceil(0.9 * len(ordered))) - 1)
    return {
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "min": min(samples),
        "p90": ordered[p90_index],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ffn-dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("float16", "float32", "bfloat16"), default="float16")
    p.add_argument("--padding-ratio", type=float, default=0.0)
    p.add_argument("--input-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--atol", type=float, default=0.001)
    p.add_argument("--rtol", type=float, default=0.01)
    p.add_argument("--repeats", type=int, default=30)
    args = p.parse_args()

    dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    config = TransformerConfig(
        args.batch_size, args.seq_len, args.d_model, args.heads,
        args.ffn_dim, args.layers, args.causal,
    )
    config.validate()

    print("=== Configuration ===")
    print(config)
    print(f"head_dim={config.head_dim}")
    print(f"device={device}, dtype={dtype}, torch={torch.__version__}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
        print(f"compute_capability={torch.cuda.get_device_capability(device)}")
        print(f"cuda_runtime={torch.version.cuda}")
    print(f"optimized_transformer_module={opt_mod.__file__}")
    print(f"triton_installed={importlib.util.find_spec('triton') is not None}")
    if device.type == "cuda":
        print(f"torch.cuda.matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}")
        print(f"torch.backends.cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    baseline = BaselineTransformer(config).to(device=device, dtype=dtype).eval()
    optimized = UserOptimizedTransformer(config).to(device=device, dtype=dtype).eval()
    copy_model_weights(baseline, optimized)

    x, valid_mask = generate_random_case(
        config, device, dtype, args.seed, args.padding_ratio, args.input_scale
    )

    print("\n=== Input statistics ===")
    xf = x.float()
    print(f"x shape={tuple(x.shape)} dtype={x.dtype} device={x.device}")
    print(f"x mean={xf.mean().item():.8g} std={xf.std().item():.8g} min={xf.min().item():.8g} max={xf.max().item():.8g}")
    print(f"valid tokens={int(valid_mask.sum().item())}/{valid_mask.numel()}")

    print("\n=== Optimized path check ===")
    if True:
        try:
            smoke = torch.empty((1, config.num_heads, config.seq_len, config.head_dim), device=device, dtype=dtype)
            mask = torch.ones((1, config.seq_len), device=device, dtype=torch.bool)
            out = opt_mod._attention(smoke, smoke, smoke, mask, config.causal)
            print(f"_attention smoke-test output shape={tuple(out.shape)}")
            print("Optimized attention path is callable.")
        except Exception as exc:
            print("Optimized attention smoke-test FAILED:")
            print(repr(exc))

    print("\n=== Full-model correctness ===")
    with torch.inference_mode():
        ref = baseline(x, valid_mask)
        opt = optimized(x, valid_mask)
    full = error_stats(ref, opt)
    print_stats("full model", full)
    print(f"benchmark gate: {'PASS' if full['failed_bench_gate'] == 0 else 'FAIL'} (atol={args.atol}, rtol={args.rtol})")

    one_layer_diagnostics(baseline, optimized, x, valid_mask, config.causal, args.atol, args.rtol)
    per_layer_diagnostics(baseline, optimized, x, valid_mask, config.causal)

    print("\n=== Performance ===")
    warmup(baseline, x, valid_mask, device)
    warmup(optimized, x, valid_mask, device)
    b = timed(baseline, x, valid_mask, device, args.repeats)
    o = timed(optimized, x, valid_mask, device, args.repeats)
    print(f"baseline : median={b['median']:.6f} ms | mean={b['mean']:.6f} ms | p90={b['p90']:.6f} ms | min={b['min']:.6f} ms")
    print(f"optimized: median={o['median']:.6f} ms | mean={o['mean']:.6f} ms | p90={o['p90']:.6f} ms | min={o['min']:.6f} ms")
    print(f"speedup  : {b['median'] / o['median']:.3f}x")

    return 0 if full["failed_bench_gate"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
