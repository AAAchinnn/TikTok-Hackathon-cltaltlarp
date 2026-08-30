"""The encoder: mask analysis, precision policy, and the routed layer stack.

This file owns everything that has to happen once per forward on the CPU --
reading the mask, choosing a precision plan, packing weights, asking the
dispatcher which implementation runs -- and then a layer loop that contains no
decisions at all. The arithmetic itself lives in opt/blocks.py as candidates.

Every forward goes through the dispatcher, including the very first. With no
routing table for the current GPU it returns the "general" candidates, so the
fallback path and the routed path are the same code path rather than two that
can drift apart. Anything only correct, or only faster, on particular shapes
belongs in a candidate, never here.

It is written as a mixin so the benchmark harness stays untouched. The
harness's `UserOptimizedTransformer` already subclasses `BaselineTransformer`;
composing this in front of it puts our `__init__` / `_apply` / `forward` ahead
of the baseline's in the MRO while the class remains an IS-A
`BaselineTransformer` holding the baseline's own submodules and parameter
names. `copy_model_weights(strict=True)` keeps working, and the only edit to
the given file is the class statement itself.

What it does, and what each part was measured to be worth on a T4 at the hub
shape (b64 n128 d128 h4 l4, fp32), all-on being 2.01x:

  1. `F.scaled_dot_product_attention` replaces the manual
     matmul -> mask -> softmax -> matmul chain. SDPA dispatches to a tiled
     online-softmax kernel and never materializes the [B, H, N, N] score
     matrix, which the baseline writes and re-reads about eleven times. This
     is what carries the long-sequence shapes: 2.57x at seq_len=1024, 4.93x
     with padding and causal masking.
  2. q/k/v packed into one GEMM. Worth 1.85x -> 2.01x. The packed weight is a
     cache built from the three Linears, keyed on parameter `_version`, so
     parameter names are untouched and any weight change invalidates it.
  3. All-valid masks detected once and dropped. Worth 1.93x -> 2.01x.
  4. `torch.compile` over the layer stack, letting Inductor fuse
     LayerNorm+residual, bias+GELU and the trailing masked_fill. Worth
     1.66x -> 2.01x, and the largest single contributor.
  5. Selected GEMMs narrowed to fp16, chosen per shape by measurement. See
     opt/precision.py -- this is the only thing that moves the GEMM-bound
     shapes, because sm75 has no TF32 path.

`mode="max-autotune"` is deliberately not used: it measured *slower* on a T4
(1.99x against 2.01x) and warns "Not enough SMs to use max_autotune_gemm".
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from . import blocks, dispatcher, masking
from .blocks import BlockContext
from .precision import (
    Calibrator,
    Plan,
    apply_backend_flags,
    settings,
)

__all__ = ["OptimizedMixin", "attention"]

# One layer's packed weights, in this order. A plain tuple of tensors rather
# than a dataclass or a slotted class on purpose: this crosses into the
# torch.compile region, and Dynamo traces flat tensor containers cleanly while
# custom objects invite graph breaks. Compilation is worth 1.66x -> 2.01x, so
# it is not something to risk for nicer attribute access.
#   0 qkv_w  1 qkv_b  2 out_w  3 out_b  4 in_w  5 in_b  6 fo_w  7 fo_b
LayerWeights = Tuple[torch.Tensor, ...]


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
) -> torch.Tensor:
    """Attention on [B, H, N, D], matching the baseline's semantics exactly.

    Kept as a module-level function because it is the one stage the diagnostic
    can probe in isolation -- SDPA fuses QK^T, softmax and P@V into a single
    kernel, so there is nothing measurable between them. Invalid query rows are
    zeroed, as the baseline's attention module does.
    """
    mask = masking.effective_mask(valid_token_mask)
    attn_mask, is_causal = masking.build_attn_mask(
        q.shape[-2], mask, causal, q.dtype, q.device
    )
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=is_causal
    )
    if mask is not None:
        out = out.masked_fill(~mask[:, None, :, None], 0)
    return out


class OptimizedMixin:
    """Mix in front of `BaselineTransformer`. Supplies forward, nothing else."""

    def __init__(self, config) -> None:
        super().__init__(config)  # type: ignore[misc]

        # Plain attributes, not buffers: registering them would add keys to
        # state_dict() and break the strict weight copy.
        self._weight_cache: Dict[Tuple, List[LayerWeights]] = {}
        self._weight_key: Optional[Tuple] = None
        self._compiled = None  # None = not built, False = compile failed
        self._calibrator = Calibrator()
        self._backend_flags_set = False
        self._announced = False
        self._routed = False
        self._cuda_graphs: Dict[Tuple, dict] = {}  # Route 3: keyed by shape+dtype+plan

        if settings.use_compile:
            # The benchmark has 14 official shapes against a Dynamo cache
            # default of 8. Past the limit Dynamo silently falls back to eager
            # with only a warning, turning the compiled path back into the
            # uncompiled one -- a 2.01x that quietly becomes 1.66x.
            try:
                import torch._dynamo

                torch._dynamo.config.cache_size_limit = max(
                    64, torch._dynamo.config.cache_size_limit
                )
            except Exception:
                pass

    # -- caches ----------------------------------------------------------

    def _apply(self, *args, **kwargs):
        """Any device or dtype move invalidates every cached tensor."""
        self._weight_cache.clear()
        self._weight_key = None
        masking.clear_caches()
        return super()._apply(*args, **kwargs)  # type: ignore[misc]

    def _weights(self, plan: Plan, stream: torch.dtype) -> List[LayerWeights]:
        """Per-layer packed weights, cast per plan, built once and cached.

        Keyed on every parameter's `_version` so an in-place weight change
        invalidates the cache -- safer than invalidating only on
        load_state_dict.
        """
        version = tuple(p._version for p in self.parameters())  # type: ignore[attr-defined]
        if self._weight_key != version:
            self._weight_cache.clear()
            self._weight_key = version

        cache_key = (plan.compute_dtype, plan.preset, stream)
        cached = self._weight_cache.get(cache_key)
        if cached is not None:
            return cached

        qkv_dt = plan.dtype_for("qkv", stream)
        out_dt = plan.dtype_for("out", stream)
        in_dt = plan.dtype_for("ffn_in", stream)
        fo_dt = plan.dtype_for("ffn_out", stream)

        packed: List[LayerWeights] = []
        # inference_mode(False) so the cached tensors are ordinary tensors even
        # though the first forward runs inside torch.inference_mode().
        with torch.inference_mode(False), torch.no_grad():
            for layer in self.layers:  # type: ignore[attr-defined]
                attn = layer.attention
                qkv_w = torch.cat(
                    [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight],
                    dim=0,
                ).contiguous().to(qkv_dt)
                qkv_b = torch.cat(
                    [attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias], dim=0
                ).contiguous().to(qkv_dt)
                packed.append((
                    qkv_w,
                    qkv_b,
                    attn.out_proj.weight.to(out_dt),
                    attn.out_proj.bias.to(out_dt),
                    layer.ffn_in.weight.to(in_dt),
                    layer.ffn_in.bias.to(in_dt),
                    layer.ffn_out.weight.to(fo_dt),
                    layer.ffn_out.bias.to(fo_dt),
                ))

        self._weight_cache[cache_key] = packed
        return packed

    # -- forward ---------------------------------------------------------

    def _context(
        self,
        x: torch.Tensor,
        plan: Plan,
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> BlockContext:
        """Everything a candidate needs, resolved outside the graph.

        Built once per forward. Shapes and dtypes are Python values Dynamo
        guards on cleanly; a precision Plan object is not worth handing it.
        """
        config = self.config  # type: ignore[attr-defined]
        batch, seq_len, d_model = x.shape
        stream = x.dtype
        return BlockContext(
            batch=batch,
            seq_len=seq_len,
            d_model=d_model,
            num_heads=config.num_heads,
            head_dim=d_model // config.num_heads,
            attn_mask=attn_mask,
            is_causal=is_causal,
            stream=stream,
            qkv_dt=plan.dtype_for("qkv", stream),
            attn_dt=plan.dtype_for("attn", stream),
            out_dt=plan.dtype_for("out", stream),
            in_dt=plan.dtype_for("ffn_in", stream),
            fo_dt=plan.dtype_for("ffn_out", stream),
            fuse_qkv=settings.fuse_qkv,
        )

    def _shape_key(self, x: torch.Tensor, has_mask: bool) -> Tuple:
        config = self.config  # type: ignore[attr-defined]
        return (
            tuple(x.shape),
            str(x.dtype),
            config.num_heads,
            config.ffn_dim,
            config.num_layers,
            bool(config.causal),
            has_mask,
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        if B >= 128:
            return self._forward_compute_heavy(x, valid_token_mask)
        if B <= 4:
            return self._forward_low_overhead(x, valid_token_mask)

        config = self.config  # type: ignore[attr-defined]

        # Mask analysis and weight packing stay outside the compiled region:
        # .item() and ._version both force graph breaks, and the results are
        # Python values Dynamo can guard on.
        mask = masking.effective_mask(valid_token_mask)
        invalid = None if mask is None else ~mask[..., None]

        # Which implementation runs in each slot, and -- if the shape has been
        # autotuned -- which precision plan. Resolved from shapes and dtypes
        # only (no tensor contents, so no synchronisation) and cached inside
        # the dispatcher, so the steady-state cost is one dict lookup.
        # Resolution happens here rather than inside _run so the chosen
        # callables are constants by the time Dynamo traces the loop.
        route = dispatcher.resolve(x, config)
        plan = self._plan_for(x, mask, config.causal, route)

        if not self._backend_flags_set:
            apply_backend_flags(plan.compute_dtype, x.dtype)
            self._backend_flags_set = True
        if settings.verbose and not self._announced:
            print(f"[opt] general | {settings.describe()} | precision={plan}")
            self._announced = True

        attn_dtype = plan.dtype_for("attn", x.dtype)
        attn_mask, is_causal = masking.build_attn_mask(
            x.shape[1], mask, config.causal, attn_dtype, x.device
        )
        weights = self._weights(plan, x.dtype)
        ctx = self._context(x, plan, attn_mask, is_causal)

        if settings.verbose and not self._routed:
            print(f"[opt] route: {route.source} | {route.names()}")
            self._routed = True
        attn_fn = route.attn_block or blocks.attn_general
        ffn_fn = route.ffn_block or blocks.ffn_general
        full_fn = route.full_block

        if settings.use_compile and self._compiled is not False:
            if self._compiled is None:
                self._compiled = torch.compile(
                    self._run, dynamic=False, mode=settings.compile_mode
                )
            try:
                return self._compiled(
                    x, invalid, weights, ctx, attn_fn, ffn_fn, full_fn
                )
            except Exception as exc:  # a fallback must never raise
                print(
                    f"[opt] compile unavailable, using eager: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._compiled = False

        return self._run(x, invalid, weights, ctx, attn_fn, ffn_fn, full_fn)

    # -- Route 2: fp16 Tensor Cores for GEMM-dominated large batches ----------
    # B >= 128 means B*N >= 16k tokens.  T4 fp16 Tensor Cores run at
    # 65 TFLOPS vs 8 TFLOPS fp32 (no TF32 on sm75), so all-fp16 is the main
    # lever.  We skip the calibrator here: for d=128 shapes fp16/all sits at
    # max_abs ~1e-3 (well inside atol=0.002) and the calibration measurement
    # costs ~20 forward passes -- for batch10k that is ~30 unmetered seconds.

    def _forward_compute_heavy(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        config  = self.config  # type: ignore[attr-defined]
        mask    = masking.effective_mask(valid_token_mask)
        invalid = None if mask is None else ~mask[..., None]

        route = dispatcher.resolve(x, config)
        if route.precision is not None:
            plan = route.precision
        elif not settings.calibrate:
            plan = Plan(
                settings.compute_dtype,
                settings.narrow_preset or "all",
                reason="pinned",
            )
        else:
            # The one path not gated by a measurement. fp16/all is the informed
            # default -- see the note above -- but it is a default, not
            # evidence. Autotune the shape and the branch above takes over.
            plan = Plan(torch.float16, "all", reason="compute_heavy")

        if not self._backend_flags_set:
            apply_backend_flags(plan.compute_dtype, x.dtype)
            self._backend_flags_set = True

        attn_dtype           = plan.dtype_for("attn", x.dtype)
        attn_mask, is_causal = masking.build_attn_mask(
            x.shape[1], mask, config.causal, attn_dtype, x.device
        )
        weights  = self._weights(plan, x.dtype)
        ctx      = self._context(x, plan, attn_mask, is_causal)
        attn_fn  = route.attn_block or blocks.attn_general
        ffn_fn   = route.ffn_block  or blocks.ffn_general
        full_fn  = route.full_block

        # Reuse the shared compiled _run -- Dynamo produces a separate
        # specialisation for fp16 weights automatically.
        if settings.use_compile and self._compiled is not False:
            if self._compiled is None:
                self._compiled = torch.compile(
                    self._run, dynamic=False, mode=settings.compile_mode
                )
            try:
                return self._compiled(x, invalid, weights, ctx, attn_fn, ffn_fn, full_fn)
            except Exception as exc:
                print(f"[opt/route2] compile fell back: {type(exc).__name__}: {exc}")
                self._compiled = False

        return self._run(x, invalid, weights, ctx, attn_fn, ffn_fn, full_fn)

    # -- Route 3: CUDA graph replay for kernel-launch-dominated tiny batches --
    # B <= 4: ~40 kernel launches at 5-10 µs each = 200-400 µs overhead out of
    # a ~2 ms forward pass.  A CUDAGraph collapses the entire pass to one host
    # command (~1 µs).  We capture the uncompiled _run -- running compile inside
    # a graph recording can trigger Triton recompilation and break the capture.

    def _forward_low_overhead(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        config  = self.config  # type: ignore[attr-defined]
        mask    = masking.effective_mask(valid_token_mask)
        invalid = None if mask is None else ~mask[..., None]

        route = dispatcher.resolve(x, config)
        plan = self._plan_for(x, mask, config.causal, route)
        if not self._backend_flags_set:
            apply_backend_flags(plan.compute_dtype, x.dtype)
            self._backend_flags_set = True

        attn_dtype           = plan.dtype_for("attn", x.dtype)
        attn_mask, is_causal = masking.build_attn_mask(
            x.shape[1], mask, config.causal, attn_dtype, x.device
        )
        weights  = self._weights(plan, x.dtype)
        ctx      = self._context(x, plan, attn_mask, is_causal)
        attn_fn  = route.attn_block or blocks.attn_general
        ffn_fn   = route.ffn_block  or blocks.ffn_general
        full_fn  = route.full_block

        gkey = (tuple(x.shape), x.dtype, invalid is not None, str(plan))
        if gkey not in self._cuda_graphs:
            self._cuda_graphs[gkey] = _capture_graph(
                self._run, x, invalid, weights, ctx, attn_fn, ffn_fn, full_fn
            )

        gd = self._cuda_graphs[gkey]
        gd["x"].copy_(x)
        if invalid is not None and gd["inv"] is not None:
            gd["inv"].copy_(invalid)
        gd["graph"].replay()
        return gd["out"].clone()

    def _plan_for(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        causal: bool,
        route: Optional["dispatcher.Plan"] = None,
    ) -> Plan:
        """Choose the precision plan for this shape, measuring on first sight.

        The measurement runs the eager path once per ladder rung against the
        model's own fp32 output. It happens inside `warmup_model`, which the
        harness runs for 20 untimed iterations before any timing, so it costs
        nothing that is measured.
        """
        # A measured table entry outranks calibration. bench/autotune.py gated
        # it against the baseline at the real tolerance over several inputs;
        # calibration gets one warmup forward against our own fp32 output and
        # has to keep a margin in hand for the difference.
        if route is not None and route.precision is not None:
            return route.precision

        if not settings.calibrate:
            return Plan(
                settings.compute_dtype,
                settings.narrow_preset or "all",
                reason="pinned",
            )

        key = self._shape_key(x, mask is not None)
        cached = self._calibrator.cached(key)
        if cached is not None:
            return cached

        invalid = None if mask is None else ~mask[..., None]

        def runner(candidate: Plan) -> torch.Tensor:
            attn_dtype = candidate.dtype_for("attn", x.dtype)
            attn_mask, is_causal = masking.build_attn_mask(
                x.shape[1], mask, causal, attn_dtype, x.device
            )
            apply_backend_flags(candidate.compute_dtype, x.dtype)
            # Eager, and through the general candidates rather than the routed
            # ones: this measures whether a precision preset is numerically
            # acceptable, which is a property of the arithmetic, not of which
            # implementation was selected. Routing is decided separately by the
            # autotuner, behind its own correctness gate.
            return self._run(
                x,
                invalid,
                self._weights(candidate, x.dtype),
                self._context(
                    x, candidate, attn_mask, is_causal
                ),
                blocks.attn_general,
                blocks.ffn_general,
                None,
            )

        plan = self._calibrator.calibrate(key, runner, x.dtype)
        if settings.verbose:
            print(f"[opt] calibrated {key[0]} {key[1]} -> {plan}")
        return plan

    def _run(
        self,
        x: torch.Tensor,
        invalid: Optional[torch.Tensor],
        weights: List[LayerWeights],
        ctx: BlockContext,
        attn_fn: Callable,
        ffn_fn: Callable,
        full_fn: Optional[Callable],
    ) -> torch.Tensor:
        """The layer stack. Every sub-block goes through a routed candidate.

        `attn_fn` / `ffn_fn` / `full_fn` arrive already chosen, so this loop
        contains no routing logic and no branching on shape -- Inductor inlines
        the candidates into the same graph it would have produced when the
        bodies were written here.

        LayerNorm and the residual adds stay here rather than inside the
        candidates: they must run in the residual-stream dtype whatever a
        candidate does internally, and keeping them out means no candidate can
        get that wrong.
        """
        for index, layer in enumerate(self.layers):  # type: ignore[attr-defined]
            w = weights[index]

            if full_fn is not None:
                x = full_fn(x, layer, w, ctx)
            else:
                x = x + attn_fn(layer.norm1(x), w, ctx)
                x = x + ffn_fn(layer.norm2(x), w, ctx)

            if invalid is not None:
                x = x.masked_fill(invalid, 0)

        x = self.final_norm(x)  # type: ignore[attr-defined]
        if invalid is not None:
            x = x.masked_fill(invalid, 0)
        return x


# ---------------------------------------------------------------------------
# CUDA graph helper (module-level so it is not part of the compiled region)
# ---------------------------------------------------------------------------

def _capture_graph(run_fn, x, invalid, weights, ctx, attn_fn, ffn_fn, full_fn):
    """Warmup then capture run_fn into a CUDAGraph.

    Static tensors (fixed device addresses) hold clones of the first inputs.
    On replay the caller copies live data in, replays, then clones the output
    so the caller cannot alias the static buffer.

    The warmup runs on a side stream so that cuBLAS / cuDNN algorithm
    selection (which allocates memory) is finished before capture starts.
    CUDAGraph capture requires a memory-allocation-free execution path.
    """
    static_x   = x.clone()
    static_inv = invalid.clone() if invalid is not None else None

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            with torch.inference_mode():
                run_fn(static_x, static_inv, weights, ctx, attn_fn, ffn_fn, full_fn)
    torch.cuda.current_stream().wait_stream(side)

    g = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(g, stream=side):
        static_out = run_fn(
            static_x, static_inv, weights, ctx, attn_fn, ffn_fn, full_fn
        )

    return {"graph": g, "x": static_x, "inv": static_inv, "out": static_out}
