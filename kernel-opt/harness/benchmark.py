#!/usr/bin/env python3
"""
Wall-clock + Nsight Compute benchmark over the fixed shape registry.

Two phases per shape:

  1. Timing. CUDA-event timed, alternating baseline/candidate rounds to cancel
     clock drift -- the same scheme the organizers' script uses, for the same
     reason. Reports median/mean/p90/min plus speedup vs the baseline.

  2. Profiling (optional, --ncu). Re-launches this script as a child under
     `ncu --set full` for ONE representative shape, then pipes the raw output
     through tools/trim_ncu.py to get the 10-15 metrics that matter.

Output is a single structured JSON document on stdout, appended to
lineage.jsonl. Nothing here prints a speedup for an unverified kernel: the
--require-pass interlock (on by default) demands a passing correctness record
for the *same kernel SHA* before any timing runs.

Usage
-----
  python benchmark.py --impl cuda
  python benchmark.py --impl cuda --mode recurrent --ncu
  python benchmark.py --impl sdpa --shapes P1_long_seq --ncu --json-out out.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import common
from common import HarnessError

OK, GATED, ERROR = 0, 1, 2

# ncu is expensive (full replay of every kernel). Profile one shape per mode by
# default rather than all of them; timing still covers the whole registry.
DEFAULT_NCU_SHAPES = {"parallel": "P1_long_seq", "recurrent": "R1_batched_token"}


# --------------------------------------------------------------------------
# Correctness interlock
# --------------------------------------------------------------------------

def latest_correctness(impl: str, kernel_sha: Optional[str]) -> Optional[Dict[str, Any]]:
    """Most recent correctness record in lineage.jsonl for this exact build."""
    matches = [
        r for r in common.read_lineage()
        if r.get("record_type") == "correctness"
        and r.get("impl") == impl
        and r.get("kernel_sha") == kernel_sha
    ]
    return matches[-1] if matches else None


def check_gate(impl: str, kernel_sha: Optional[str]) -> Dict[str, Any]:
    """Decide whether this variant is allowed to be timed."""
    record = latest_correctness(impl, kernel_sha)
    if record is None:
        return {
            "allowed": False,
            "reason": (
                f"no correctness record for impl={impl} kernel_sha={kernel_sha}. "
                "Run correctness_check.py first."
            ),
        }
    if record.get("status") != "pass":
        return {
            "allowed": False,
            "reason": (
                f"last correctness record for this build is "
                f"status={record.get('status')!r} "
                f"(failed shapes: {record.get('failed_shapes')}). "
                "A variant that fails correctness is never benchmarked."
            ),
            "correctness_record": record,
        }
    return {"allowed": True, "correctness_record": record}


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

def time_shape(
    shape: Dict[str, Any],
    impl: str,
    warmup: int,
    repeats: int,
    rounds: int,
    seed: int,
    time_baseline: bool,
) -> Dict[str, Any]:
    import torch

    tb = common.import_reference_module()

    device = torch.device("cuda")
    dtype = tb.resolve_dtype(shape["dtype"])
    config = common.shape_to_config(shape)

    baseline, candidate = common.build_pair(shape, impl, device, dtype, seed=seed)

    x, valid_mask = tb.generate_random_case(
        config=config, device=device, dtype=dtype, seed=seed + 100000,
        padding_ratio=shape["padding_ratio"], input_scale=1.0,
    )

    tb.warmup_model(candidate, x, valid_mask, warmup, device)
    if time_baseline:
        tb.warmup_model(baseline, x, valid_mask, warmup, device)

    cand_samples: List[float] = []
    base_samples: List[float] = []

    # Alternate order across rounds so thermal/clock drift hits both equally.
    for round_index in range(rounds):
        cand_first = round_index % 2 == 0
        if cand_first:
            cand_samples += tb.benchmark_once(candidate, x, valid_mask, repeats, device)
            if time_baseline:
                base_samples += tb.benchmark_once(baseline, x, valid_mask, repeats, device)
        else:
            if time_baseline:
                base_samples += tb.benchmark_once(baseline, x, valid_mask, repeats, device)
            cand_samples += tb.benchmark_once(candidate, x, valid_mask, repeats, device)

    cand = tb.TimingResult(cand_samples)
    tokens = shape["batch_size"] * shape["seq_len"]

    record: Dict[str, Any] = {
        "shape_id": shape["id"],
        "mode": shape["mode"],
        "dtype": shape["dtype"],
        "samples": len(cand_samples),
        "candidate": {
            "median_ms": cand.median_ms,
            "mean_ms": cand.mean_ms,
            "p90_ms": cand.p90_ms,
            "min_ms": cand.min_ms,
            "tokens_per_s": tokens * 1000.0 / cand.median_ms,
        },
        "peak_mem_mb": round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 1),
    }

    if time_baseline:
        base = tb.TimingResult(base_samples)
        record["baseline"] = {
            "median_ms": base.median_ms,
            "mean_ms": base.mean_ms,
            "p90_ms": base.p90_ms,
            "min_ms": base.min_ms,
            "tokens_per_s": tokens * 1000.0 / base.median_ms,
        }
        record["speedup_median"] = base.median_ms / cand.median_ms

    torch.cuda.reset_peak_memory_stats(device)
    return record


# --------------------------------------------------------------------------
# ncu profiling
# --------------------------------------------------------------------------

def profile_shape(
    shape_id: str,
    impl: str,
    ncu_path: str,
    iterations: int,
    timeout: int,
    keep_raw: Optional[Path],
) -> Dict[str, Any]:
    """Run this script as an ncu child, then trim the output."""
    child_cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--profile-child", shape_id, "--impl", impl,
        "--child-iters", str(iterations),
    ]
    ncu_cmd = [
        ncu_path,
        "--set", "full",
        "--target-processes", "all",
        "--csv",
        "--page", "raw",
        # Skip warmup launches; profile steady state.
        "--launch-skip", "20",
        "--launch-count", "40",
    ] + child_cmd

    env = dict(os.environ)
    env["KERNEL_OPT_NCU_CHILD"] = "1"

    try:
        proc = subprocess.run(
            ncu_cmd, capture_output=True, text=True, timeout=timeout, env=env
        )
    except subprocess.TimeoutExpired:
        return {"available": False, "reason": f"ncu timed out after {timeout}s"}
    except OSError as exc:
        return {"available": False, "reason": f"could not exec ncu: {exc}"}

    raw = proc.stdout
    if keep_raw:
        keep_raw.write_text(raw + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")

    if proc.returncode != 0 and not raw.strip():
        tail = (proc.stderr or "").strip().splitlines()[-6:]
        return {
            "available": False,
            "reason": f"ncu exited {proc.returncode}",
            "stderr_tail": tail,
            "hint": (
                "Profiling usually needs elevated GPU counter permissions. "
                "See NVIDIA ERR_NVGPUCTRPERM."
            ),
        }

    # Hand the raw text to the trimmer rather than parsing it here, so the
    # loop and the harness always see identically-shaped metrics.
    sys.path.insert(0, str(common.TOOLS_DIR))
    try:
        import trim_ncu
    except ImportError as exc:
        return {"available": False, "reason": f"cannot import trim_ncu: {exc}"}

    try:
        metrics = trim_ncu.trim(raw)
    except Exception as exc:
        return {
            "available": False,
            "reason": f"trim_ncu failed: {type(exc).__name__}: {exc}",
        }

    metrics["available"] = True
    metrics["shape_id"] = shape_id
    return metrics


def run_profile_child(shape_id: str, impl: str, iterations: int) -> int:
    """Body profiled by ncu: N steady-state forward passes, nothing else."""
    import torch

    tb = common.import_reference_module()
    shape = common.SHAPES_BY_ID[shape_id]
    device = torch.device("cuda")
    dtype = tb.resolve_dtype(shape["dtype"])
    config = common.shape_to_config(shape)

    _, candidate = common.build_pair(shape, impl, device, dtype)
    x, valid_mask = tb.generate_random_case(
        config=config, device=device, dtype=dtype, seed=1234 + 100000,
        padding_ratio=shape["padding_ratio"], input_scale=1.0,
    )

    with torch.inference_mode():
        for _ in range(iterations):
            candidate(x, valid_mask)
    torch.cuda.synchronize(device)
    return 0


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run(args: argparse.Namespace) -> Dict[str, Any]:
    shapes = common.select_shapes(args.shapes, args.mode)
    kernel_sha = common.kernel_hash() if args.impl == "cuda" else None

    report: Dict[str, Any] = {
        "kind": "benchmark",
        "impl": args.impl,
        "kernel_sha": kernel_sha,
        "variant": args.variant,
        "env": common.env_fingerprint(),
        "timing": {
            "warmup": args.warmup, "repeats": args.repeats, "rounds": args.rounds,
            "method": "torch.cuda.Event on the current stream",
        },
        "shapes": [],
        "ncu": {"available": False, "reason": "not requested"},
    }

    gate = check_gate(args.impl, kernel_sha)
    report["gate"] = gate
    if args.require_pass and not gate["allowed"]:
        report["status"] = "gated"
        return report

    import torch

    if not torch.cuda.is_available():
        report["status"] = "error"
        report["error"] = "CUDA not available"
        return report

    for shape in shapes:
        try:
            report["shapes"].append(time_shape(
                shape=shape, impl=args.impl, warmup=args.warmup,
                repeats=args.repeats, rounds=args.rounds, seed=args.seed,
                time_baseline=not args.no_baseline,
            ))
        except torch.cuda.OutOfMemoryError as exc:
            report["shapes"].append({
                "shape_id": shape["id"], "mode": shape["mode"],
                "error": f"OOM: {exc}",
            })
            torch.cuda.empty_cache()
        except Exception as exc:
            report["shapes"].append({
                "shape_id": shape["id"], "mode": shape["mode"],
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=6),
            })

    if args.ncu:
        ncu_path = common.find_ncu()
        if ncu_path is None:
            report["ncu"] = {
                "available": False,
                "reason": "ncu not found on PATH, in $NCU_PATH, or in the "
                          "standard CUDA/Nsight install dirs",
            }
        else:
            targets = args.ncu_shapes or sorted({
                DEFAULT_NCU_SHAPES[s["mode"]] for s in shapes
                if s["mode"] in DEFAULT_NCU_SHAPES
            })
            profiles = {}
            for shape_id in targets:
                keep = Path(args.keep_raw_ncu) / f"ncu_{shape_id}.csv" if args.keep_raw_ncu else None
                if keep:
                    keep.parent.mkdir(parents=True, exist_ok=True)
                profiles[shape_id] = profile_shape(
                    shape_id=shape_id, impl=args.impl, ncu_path=ncu_path,
                    iterations=args.child_iters, timeout=args.ncu_timeout,
                    keep_raw=keep,
                )
            report["ncu"] = {"available": True, "ncu_path": ncu_path, "profiles": profiles}

    errored = [s for s in report["shapes"] if "error" in s]
    report["status"] = "ok" if not errored else "partial"
    return report


def summarize(report: Dict[str, Any]) -> str:
    lines = []
    for shape in report.get("shapes", []):
        if "error" in shape:
            lines.append(f"  {shape['shape_id']:<20} ERROR  {shape['error'][:70]}")
            continue
        cand = shape["candidate"]
        speed = shape.get("speedup_median")
        speed_txt = f"{speed:.3f}x" if speed else "n/a"
        lines.append(
            f"  {shape['shape_id']:<20} {cand['median_ms']:>9.4f} ms  "
            f"speedup={speed_txt:<8} {cand['tokens_per_s']:>12,.0f} tok/s"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wall-clock + ncu benchmark over fixed shapes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--impl", choices=common.IMPLEMENTATIONS, default="cuda")
    parser.add_argument("--shapes", nargs="*", default=None)
    parser.add_argument("--mode", choices=("all", "parallel", "recurrent"), default="all")

    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-baseline", action="store_true",
                        help="skip baseline timing (no speedup column)")

    parser.add_argument("--ncu", action="store_true", help="collect ncu metrics")
    parser.add_argument("--ncu-shapes", nargs="*", default=None)
    parser.add_argument("--ncu-timeout", type=int, default=900)
    parser.add_argument("--child-iters", type=int, default=64)
    parser.add_argument("--keep-raw-ncu", default=None,
                        help="directory to save raw ncu CSV for debugging")

    parser.add_argument("--variant", default=None)
    parser.add_argument("--require-pass", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="refuse to time a build without a passing "
                             "correctness record (keep this on)")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--no-lineage", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")

    # Internal: the process ncu actually profiles.
    parser.add_argument("--profile-child", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.profile_child:
        return run_profile_child(args.profile_child, args.impl, args.child_iters)

    try:
        report = run(args)
    except HarnessError as exc:
        report = {"kind": "benchmark", "status": "error", "error": str(exc),
                  "env": common.env_fingerprint()}
    except ImportError as exc:
        report = {"kind": "benchmark", "status": "error",
                  "error": f"missing dependency: {exc}",
                  "env": common.env_fingerprint()}

    payload = json.dumps(report, indent=2, sort_keys=True, default=str)
    print(payload)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")

    if not args.quiet and report.get("shapes"):
        print("\n=== timing summary ===", file=sys.stderr)
        print(summarize(report), file=sys.stderr)
    if report["status"] == "gated":
        print(f"\nGATED: {report['gate']['reason']}", file=sys.stderr)

    if not args.no_lineage:
        common.append_lineage({
            "record_type": "benchmark",
            "variant": args.variant,
            "impl": args.impl,
            "kernel_sha": report.get("kernel_sha"),
            "status": report["status"],
            "median_ms": {
                s["shape_id"]: s["candidate"]["median_ms"]
                for s in report.get("shapes", []) if "candidate" in s
            },
            "speedup": {
                s["shape_id"]: s.get("speedup_median")
                for s in report.get("shapes", []) if "candidate" in s
            },
            "ncu": {
                sid: prof.get("summary")
                for sid, prof in report.get("ncu", {}).get("profiles", {}).items()
            } or None,
            "error": report.get("error"),
        })

    return {"ok": OK, "partial": OK, "gated": GATED, "error": ERROR}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
