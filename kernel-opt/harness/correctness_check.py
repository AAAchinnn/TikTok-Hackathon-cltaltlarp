#!/usr/bin/env python3
"""
Hard pass/fail correctness gate for a kernel variant.

This wraps the equivalence check that already exists in
torch_transformer_benchmark.py -- it does not reimplement it. Specifically it
calls the organizers' `generate_random_case()` and `compare_outputs()`, so the
element-wise criterion stays exactly theirs:

    abs(user - ref) <= atol   OR   abs(user - ref) <= rtol * abs(ref)

(Deliberately NOT torch.isclose, which uses atol + rtol*|ref| and is more
permissive. The reference module says so in a comment; we inherit that.)

Contract for the optimization loop
----------------------------------
  exit 0  -> PASS. The variant is eligible for benchmarking.
  exit 1  -> FAIL. Numerical mismatch. DO NOT benchmark this variant.
  exit 2  -> ERROR. Could not evaluate (no CUDA, compile failure, OOM, ...).
             Also not eligible; distinguished from FAIL so the loop can tell
             "this kernel is wrong" from "this machine is broken".

Nothing in this file measures or reports speed. That separation is the point:
benchmark.py refuses to run unless this has returned 0 for the same kernel hash.

Usage
-----
  python correctness_check.py --impl cuda
  python correctness_check.py --impl cuda --mode parallel --trials 5
  python correctness_check.py --impl sdpa --shapes P1_long_seq R1_batched_token
  python correctness_check.py --list-shapes
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import asdict
from typing import Any, Dict, List

import common
from common import HarnessError

# Problem statement 3.2: relative error < 0.02, abs error < 0.002.
# (The reference module's docstring says 0.01/0.001, but its argparse defaults
# and the problem statement agree on these. Overridable via --rtol/--atol.)
DEFAULT_RTOL = 0.02
DEFAULT_ATOL = 0.002

PASS, FAIL, ERROR = 0, 1, 2


def check_one_shape(
    shape: Dict[str, Any],
    impl: str,
    trials: int,
    seed: int,
    rtol: float,
    atol: float,
    input_scale: float,
    verbose: bool,
) -> Dict[str, Any]:
    """Run `trials` random cases for one shape. Returns a result record."""
    import torch

    tb = common.import_reference_module()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = tb.resolve_dtype(shape["dtype"])
    config = common.shape_to_config(shape)

    baseline, candidate = common.build_pair(shape, impl, device, dtype, seed=seed)

    trial_records: List[Dict[str, Any]] = []
    worst_abs = 0.0
    worst_rel = 0.0
    failed_elements = 0
    total_elements = 0
    all_passed = True

    with torch.inference_mode():
        for trial in range(trials):
            x, valid_mask = tb.generate_random_case(
                config=config,
                device=device,
                dtype=dtype,
                seed=seed + trial,
                padding_ratio=shape["padding_ratio"],
                input_scale=input_scale,
            )
            reference = baseline(x, valid_mask)
            produced = candidate(x, valid_mask)

            # The organizers' comparator, verbatim.
            result = tb.compare_outputs(reference, produced, rtol=rtol, atol=atol)

            all_passed &= result.passed
            worst_abs = max(worst_abs, result.max_abs_error)
            worst_rel = max(worst_rel, result.max_relative_error)
            failed_elements += result.failed_elements
            total_elements += result.total_elements

            record = {
                "trial": trial,
                "passed": result.passed,
                "max_abs_error": result.max_abs_error,
                "max_relative_error": result.max_relative_error,
                "mean_abs_error": result.mean_abs_error,
                "failed_elements": result.failed_elements,
                "total_elements": result.total_elements,
            }
            if not result.passed:
                # Keep the diagnostic payload small but actionable: where it
                # broke and by how much. This text is what a future prompt sees.
                record["worst_index"] = list(result.worst_index)
                record["reference_at_worst"] = result.reference_at_worst
                record["candidate_at_worst"] = result.optimized_at_worst
                record["failed_feature_dims"] = result.failed_feature_dims[:16]
            trial_records.append(record)

            if verbose:
                status = "PASS" if result.passed else "FAIL"
                print(
                    f"  [{shape['id']}] trial {trial + 1}/{trials}: {status} "
                    f"max_abs={result.max_abs_error:.3e} "
                    f"max_rel={result.max_relative_error:.3e} "
                    f"failed={result.failed_elements}/{result.total_elements}",
                    file=sys.stderr,
                )

    return {
        "shape_id": shape["id"],
        "mode": shape["mode"],
        "dtype": shape["dtype"],
        "passed": all_passed,
        "max_abs_error": worst_abs,
        "max_relative_error": worst_rel,
        "failed_elements": failed_elements,
        "total_elements": total_elements,
        "trials": trial_records,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    shapes = common.select_shapes(args.shapes, args.mode)

    report: Dict[str, Any] = {
        "kind": "correctness",
        "impl": args.impl,
        "kernel_sha": common.kernel_hash() if args.impl == "cuda" else None,
        "variant": args.variant,
        "rtol": args.rtol,
        "atol": args.atol,
        "criterion": "abs<=atol OR abs<=rtol*|ref| (per element, all must hold)",
        "trials_per_shape": args.trials,
        "seed": args.seed,
        "env": common.env_fingerprint(),
        "shapes": [],
    }

    import torch

    if not torch.cuda.is_available():
        # Not a FAIL: the kernel was never given a chance to be wrong.
        report["status"] = "error"
        report["error"] = "CUDA not available; cannot evaluate a GPU kernel"
        return report

    overall = True
    for shape in shapes:
        try:
            result = check_one_shape(
                shape=shape,
                impl=args.impl,
                trials=args.trials,
                seed=args.seed,
                rtol=args.rtol,
                atol=args.atol,
                input_scale=args.input_scale,
                verbose=args.verbose,
            )
        except torch.cuda.OutOfMemoryError as exc:
            report["status"] = "error"
            report["error"] = f"OOM on shape {shape['id']}: {exc}"
            return report
        except Exception as exc:
            report["status"] = "error"
            report["error"] = f"{type(exc).__name__} on shape {shape['id']}: {exc}"
            report["traceback"] = traceback.format_exc(limit=8)
            return report

        report["shapes"].append(result)
        overall &= result["passed"]
        if not result["passed"] and args.fail_fast:
            break

    report["status"] = "pass" if overall else "fail"
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hard pass/fail correctness gate (no timing).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--impl", choices=common.IMPLEMENTATIONS, default="cuda",
        help="which candidate to verify against the baseline",
    )
    parser.add_argument("--shapes", nargs="*", default=None, help="shape ids")
    parser.add_argument(
        "--mode", choices=("all", "parallel", "recurrent"), default="all"
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument(
        "--variant", default=None,
        help="variant label recorded in lineage.jsonl (e.g. v007)",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--json-out", default=None, help="write report JSON here")
    parser.add_argument(
        "--no-lineage", action="store_true", help="skip the lineage.jsonl append"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--list-shapes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_shapes:
        for shape in common.SHAPES:
            print(
                f"{shape['id']:<20} {shape['mode']:<10} "
                f"B={shape['batch_size']:<3} S={shape['seq_len']:<5} "
                f"d={shape['d_model']:<5} H={shape['num_heads']:<3} "
                f"L={shape['num_layers']} {shape['dtype']:<9} "
                f"causal={str(shape['causal']):<5} pad={shape['padding_ratio']}"
            )
        return PASS

    try:
        report = run(args)
    except HarnessError as exc:
        report = {
            "kind": "correctness",
            "status": "error",
            "error": str(exc),
            "env": common.env_fingerprint(),
        }
    except ImportError as exc:
        report = {
            "kind": "correctness",
            "status": "error",
            "error": f"missing dependency: {exc}",
            "env": common.env_fingerprint(),
        }

    payload = json.dumps(report, indent=2, sort_keys=True, default=str)
    print(payload)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")

    if not args.no_lineage:
        common.append_lineage({
            "record_type": "correctness",
            "variant": args.variant,
            "impl": args.impl,
            "kernel_sha": report.get("kernel_sha"),
            "status": report["status"],
            "rtol": args.rtol,
            "atol": args.atol,
            "max_abs_error": max(
                (s["max_abs_error"] for s in report.get("shapes", [])), default=None
            ),
            "max_relative_error": max(
                (s["max_relative_error"] for s in report.get("shapes", [])), default=None
            ),
            "failed_shapes": [
                s["shape_id"] for s in report.get("shapes", []) if not s["passed"]
            ],
            "error": report.get("error"),
        })

    return {"pass": PASS, "fail": FAIL, "error": ERROR}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
