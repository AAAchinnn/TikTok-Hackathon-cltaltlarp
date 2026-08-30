#!/usr/bin/env python3
"""Measure every candidate x precision combination per shape, keep the winner.

This is what turns opt/dispatcher.py from a mechanism into a decision. It
writes opt/configs/<gpu>.json, which the dispatcher loads on first forward.

The rule the whole file exists to enforce: nothing enters the table that has
not beaten the baseline on speed *and* cleared the accuracy gate on several
different inputs. A combination that is faster but wrong is not a winner, and
the gate is the harness's own `compare_outputs`, not an approximation of it.

Why this beats the run-time calibrator it overrides
---------------------------------------------------
`opt/precision.Calibrator` measures one warmup input against the model's own
fp32 output, then keeps a safety margin because a single input under-reports
the worst case. On a T4 that under-report was measured at 1.39x: presets
accepted at <= 7e-4 produced up to 9.7e-4 once the harness ran five inputs.
The margin covers that, but only just, and it costs real speed -- a shape whose
true error is 1.2e-3 is rejected against a 2e-3 gate because the budget was
0.7e-3.

The autotuner has no such problem. It runs the actual baseline, at the actual
tolerance, over `--trials` different inputs, and stores what it saw. The
headroom check below is then about input variation beyond the sample, not about
the gap between a proxy and the truth.

Usage
-----
    python bench/autotune.py                       # every shape that fits
    python bench/autotune.py --rows 1 8 13         # just these
    python bench/autotune.py --atol 0.002 --rtol 0.02
    python bench/autotune.py --dry-run             # print, do not write

Rows are 1-based indices into dispatcher.OFFICIAL_SHAPES.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

import torch_transformer_benchmark as harness
from opt import dispatcher
from opt.precision import NARROW_LADDER, Plan as PrecisionPlan


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------

def check_environment() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit(
            "autotune needs a GPU: the table it writes is named after one, and "
            "a table measured on CPU would be actively misleading."
        )

    wired = any(
        cls.__name__ == "OptimizedMixin"
        for cls in harness.UserOptimizedTransformer.__mro__
    )
    if not wired:
        raise SystemExit(
            "torch_transformer_benchmark.UserOptimizedTransformer is still the "
            "baseline passthrough. Mix in OptimizedMixin first:\n\n"
            "    from opt import OptimizedMixin\n\n"
            "    class UserOptimizedTransformer(OptimizedMixin, "
            "BaselineTransformer):\n\n"
            "Without that, every measurement below would be 1.0x."
        )

    return torch.device("cuda")


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

def time_median_ms(model, x, mask, warmup: int, iters: int, device) -> float:
    """Median latency, timed the way the harness times it.

    CUDA events on the current stream, not wall clock: the launch queue makes
    wall clock measure the host, not the GPU.
    """
    with torch.inference_mode():
        for _ in range(warmup):
            model(x, mask)
        torch.cuda.synchronize(device)

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            model(x, mask)
            ends[i].record()
        torch.cuda.synchronize(device)

    return statistics.median(
        s.elapsed_time(e) for s, e in zip(starts, ends)
    )


def accuracy(
    baseline, model, config, device, dtype, trials, seed, pad, scale, rtol, atol
) -> Tuple[bool, float, float]:
    """Worst-case error over `trials` distinct inputs, judged by the harness.

    Returns (passed_every_trial, worst_max_abs, worst_max_rel). Uses
    `harness.compare_outputs` directly so the gate here and the gate the
    submission is scored by cannot drift apart.
    """
    passed = True
    worst_abs = 0.0
    worst_rel = 0.0

    with torch.inference_mode():
        for trial in range(trials):
            x, mask = harness.generate_random_case(
                config=config,
                device=device,
                dtype=dtype,
                seed=seed + trial,
                padding_ratio=pad,
                input_scale=scale,
            )
            reference = baseline(x, mask)
            candidate = model(x, mask)
            result = harness.compare_outputs(
                reference, candidate, rtol=rtol, atol=atol
            )
            passed &= result.passed
            worst_abs = max(worst_abs, result.max_abs_error)
            worst_rel = max(worst_rel, result.max_relative_error)

    return passed, worst_abs, worst_rel


def force_route(
    key: dispatcher.Key,
    attn_fn,
    ffn_fn,
    precision: PrecisionPlan,
) -> None:
    """Pin one combination by seeding the dispatcher's plan cache.

    Driving the model through its own routing path -- rather than reaching in
    and setting attributes -- is what makes the measurement mean something: the
    batch-size routes in general.py, the CUDA graph capture, the compiled
    region and the weight packing all behave exactly as they will in the real
    run.
    """
    dispatcher._plan_cache[key] = dispatcher.Plan(
        attn_block=attn_fn,
        ffn_block=ffn_fn,
        precision=precision,
        source="autotune",
    )


def reset_model(model) -> None:
    """Drop everything cached under the previous combination."""
    model._weight_cache.clear()
    model._weight_key = None
    model._cuda_graphs.clear()
    model._backend_flags_set = False
    # `_compiled` is deliberately kept: Dynamo guards on weight dtype and will
    # specialise on its own. Rebuilding it per combination would spend minutes
    # in Inductor measuring nothing.


# --------------------------------------------------------------------------
# Per-shape search
# --------------------------------------------------------------------------

def presets_to_try(dtype: torch.dtype) -> List[str]:
    if dtype is not torch.float32:
        # The stream is already narrow; the harness's own reference runs in it,
        # so there is nothing left to trade.
        return ["off"]
    return list(NARROW_LADDER)


def precision_plan(preset: str) -> PrecisionPlan:
    if preset == "off":
        return PrecisionPlan(None, "off", reason="autotune")
    return PrecisionPlan(torch.float16, preset, reason="autotune")


def tune_shape(shape: dict, args, device) -> Optional[Tuple[dispatcher.Key, dict]]:
    dtype = harness.resolve_dtype(args.dtype)
    config = harness.TransformerConfig(
        batch_size=shape["batch"],
        seq_len=shape["seq_len"],
        d_model=shape["d_model"],
        num_heads=shape["num_heads"],
        ffn_dim=shape["ffn_dim"],
        num_layers=shape["num_layers"],
        causal=shape["causal"],
    )
    config.validate()

    torch.manual_seed(args.seed)
    baseline = harness.BaselineTransformer(config)
    model = harness.UserOptimizedTransformer(config)
    harness.copy_model_weights(baseline, model, strict=True)
    baseline = baseline.to(device=device, dtype=dtype).eval()
    model = model.to(device=device, dtype=dtype).eval()

    key = dispatcher.Key(dtype=args.dtype, **shape)
    x, mask = harness.generate_random_case(
        config, device, dtype, args.seed + 100000, args.padding_ratio, 1.0
    )

    base_ms = time_median_ms(baseline, x, mask, args.warmup, args.iters, device)
    print(f"  baseline               {base_ms:9.3f} ms")

    attn_cands = dispatcher.candidates("attn_block")
    ffn_cands = dispatcher.candidates("ffn_block")

    best: Optional[dict] = None
    rows: List[dict] = []

    for attn_name, attn_fn in attn_cands.items():
        for ffn_name, ffn_fn in ffn_cands.items():
            for preset in presets_to_try(dtype):
                pp = precision_plan(preset)
                force_route(key, attn_fn, ffn_fn, pp)
                reset_model(model)

                label = f"{attn_name}/{ffn_name} fp16:{preset}"
                try:
                    ok, max_abs, max_rel = accuracy(
                        baseline, model, config, device, dtype,
                        args.trials, args.seed, args.padding_ratio, 1.0,
                        args.rtol, args.atol,
                    )
                except torch.cuda.OutOfMemoryError:
                    print(f"  {label:34s} OOM")
                    torch.cuda.empty_cache()
                    continue
                except Exception as exc:
                    print(f"  {label:34s} error: {type(exc).__name__}: {exc}")
                    continue

                headroom = args.atol * args.headroom
                if not ok:
                    print(f"  {label:34s}   FAIL  max_abs={max_abs:.3g}")
                    continue
                if max_abs > headroom:
                    # Passed today, but with less margin than we are willing to
                    # ship. Five inputs is a sample, not a proof.
                    print(
                        f"  {label:34s}   thin  max_abs={max_abs:.3g} "
                        f"> {headroom:.3g}"
                    )
                    continue

                ms = time_median_ms(model, x, mask, args.warmup, args.iters, device)
                speedup = base_ms / ms
                print(
                    f"  {label:34s} {ms:9.3f} ms  {speedup:6.3f}x  "
                    f"max_abs={max_abs:.3g}"
                )

                row = {
                    "attn_block": attn_name,
                    "ffn_block": ffn_name,
                    "preset": preset,
                    "ms": ms,
                    "speedup": speedup,
                    "max_abs": max_abs,
                    "max_rel": max_rel,
                }
                rows.append(row)
                if best is None or ms < best["ms"]:
                    best = row

    dispatcher._plan_cache.pop(key, None)
    del baseline, model
    torch.cuda.empty_cache()

    if best is None:
        print("  -> no combination cleared the gate; leaving this shape unrouted")
        return None

    print(
        f"  -> {best['attn_block']}/{best['ffn_block']} fp16:{best['preset']} "
        f"at {best['speedup']:.3f}x"
    )

    record = {
        "plan": {
            "attn_block": best["attn_block"],
            "ffn_block": best["ffn_block"],
        },
        "precision": {
            "compute_dtype": "off" if best["preset"] == "off" else "float16",
            "preset": best["preset"],
            "measured_max_abs": best["max_abs"],
        },
        "evidence": {
            "baseline_ms": base_ms,
            "chosen_ms": best["ms"],
            "speedup": best["speedup"],
            "max_abs": best["max_abs"],
            "max_rel": best["max_rel"],
            "gate_atol": args.atol,
            "gate_rtol": args.rtol,
            "headroom": args.headroom,
            "trials": args.trials,
            "padding_ratio": args.padding_ratio,
            "considered": rows,
        },
    }
    return key, record


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--rows", type=int, nargs="*", default=None,
        help="1-based indices into dispatcher.OFFICIAL_SHAPES (default: all)",
    )
    p.add_argument("--dtype", default="float32",
                   choices=("float32", "float16", "bfloat16"))
    p.add_argument("--padding-ratio", type=float, default=0.0)
    p.add_argument("--rtol", type=float, default=0.02)
    p.add_argument("--atol", type=float, default=0.002)
    p.add_argument(
        "--headroom", type=float, default=0.8,
        help="fraction of atol a winner may spend (default 0.8)",
    )
    p.add_argument("--trials", type=int, default=5,
                   help="distinct inputs per accuracy check")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = check_environment()

    shapes = dispatcher.OFFICIAL_SHAPES
    indices = (
        [i - 1 for i in args.rows] if args.rows
        else list(range(len(shapes)))
    )

    print(f"gpu    : {torch.cuda.get_device_name()}")
    print(f"torch  : {torch.__version__}")
    print(f"gate   : abs <= {args.atol:g} OR rel <= {args.rtol:g}"
          f"  (winner must stay under {args.atol * args.headroom:g})")
    print(f"shapes : {[i + 1 for i in indices]}")

    table: Dict[dispatcher.Key, dict] = {}
    started = time.time()

    for i in indices:
        if not 0 <= i < len(shapes):
            print(f"\n[row {i + 1}] out of range, skipping")
            continue
        shape = shapes[i]
        print(f"\n[row {i + 1}] {dispatcher.Key(dtype=args.dtype, **shape)}")
        try:
            result = tune_shape(shape, args, device)
        except torch.cuda.OutOfMemoryError:
            print("  OOM building this shape; skipping")
            torch.cuda.empty_cache()
            continue
        if result is not None:
            key, record = result
            table[key] = record

    print(f"\ntuned {len(table)} shape(s) in {time.time() - started:.0f}s")

    if args.dry_run:
        print("dry run: nothing written")
        return 0
    if not table:
        print("nothing to write")
        return 1

    path = dispatcher.save_table(table, args.out)
    print(f"wrote {path}")
    print("Re-run the benchmark; [opt] route should now read 'table'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
