#!/usr/bin/env python3
"""Run the harness across the official shapes and collect the evidence.

One subprocess per (shape, padding) so a crash or an OOM on one shape cannot
take the sweep with it. Each run gets its own log; the run also prints a
Markdown table, which is what goes in the tech report.

`--atol` / `--rtol` are passed to the harness *and* exported as
OPT_TARGET_ATOL / OPT_TARGET_RTOL, so the run-time calibrator targets the same
gate the run is scored against. opt/precision.py promises exactly this, and
keeping it in one place is what stops the two from drifting apart.

Usage
-----
    python tools/sweep.py                          # all runnable shapes
    python tools/sweep.py --rows 1 8 13
    python tools/sweep.py --padding 0.0 0.3
    python tools/sweep.py --out results_t4
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opt import dispatcher


# Shapes the sweep declines to run by default, with the reason. Row 14 is not a
# tuning problem: at b32 x n100000 x d1024 the fp32 input alone is 13.1 GB, and
# the baseline it is compared against materializes [B, H, N, N]. Nothing runs
# it, so a report should say so rather than leave a silent gap.
HEAVY = {
    6: "batch=10000 -- baseline needs ~2.6 GB of scores; run it alone",
    14: "seq_len=100000 -- 13.1 GB input, baseline cannot run on any GPU",
}

SPEEDUP_RE = re.compile(r"^speedup\s*:\s*([\d.]+)x", re.M)
SUMMARY_RE = re.compile(r"^summary:\s*(PASS|FAIL)\s*\|\s*max_abs=([\d.eE+-]+)", re.M)
ROUTE_RE = re.compile(r"^\[opt\] route: (\S+)", re.M)
PREC_RE = re.compile(r"^\[opt\].*precision=(.+)$", re.M)


def label(index: int, shape: dict) -> str:
    bits = [f"row{index:02d}"]
    defaults = dict(batch=64, d_model=128, num_heads=4, seq_len=128,
                    num_layers=4, ffn_dim=128)
    for field, short in (("batch", "b"), ("d_model", "d"),
                         ("num_heads", "h"), ("seq_len", "n"),
                         ("num_layers", "l")):
        if shape[field] != defaults[field]:
            bits.append(f"{short}{shape[field]}")
    return "_".join(bits) if len(bits) > 1 else f"row{index:02d}_hub"


def command(shape: dict, pad: float, args) -> List[str]:
    cmd = [
        sys.executable, str(ROOT / "torch_transformer_benchmark.py"),
        "--batch-size", str(shape["batch"]),
        "--seq-len", str(shape["seq_len"]),
        "--d-model", str(shape["d_model"]),
        "--heads", str(shape["num_heads"]),
        "--ffn-dim", str(shape["ffn_dim"]),
        "--layers", str(shape["num_layers"]),
        "--device", args.device,
        "--dtype", args.dtype,
        "--padding-ratio", str(pad),
        "--rtol", str(args.rtol),
        "--atol", str(args.atol),
        "--accuracy-trials", str(args.trials),
        "--repeats", str(args.repeats),
        "--benchmark-rounds", str(args.rounds),
    ]
    if shape["causal"]:
        cmd.append("--causal")
    return cmd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rows", type=int, nargs="*", default=None,
                   help="1-based indices into dispatcher.OFFICIAL_SHAPES")
    p.add_argument("--padding", type=float, nargs="*", default=[0.0, 0.3])
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32",
                   choices=("float32", "float16", "bfloat16"))
    p.add_argument("--rtol", type=float, default=0.02)
    p.add_argument("--atol", type=float, default=0.002)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--out", type=Path, default=ROOT / "results_sweep")
    p.add_argument("--include-heavy", action="store_true",
                   help="also run the shapes listed in HEAVY")
    p.add_argument("--timeout", type=int, default=1800)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    shapes = dispatcher.OFFICIAL_SHAPES
    if args.rows:
        indices = [i - 1 for i in args.rows]
    else:
        indices = [
            i for i in range(len(shapes))
            if args.include_heavy or (i + 1) not in HEAVY
        ]

    for row, reason in sorted(HEAVY.items()):
        if (row - 1) not in indices:
            print(f"skipping row {row}: {reason}")

    # The calibrator should aim at the gate this sweep is judged by, not at its
    # own default. See opt/precision.py.
    env = {
        **os.environ,
        "OPT_VERBOSE": "1",
        "OPT_TARGET_ATOL": str(args.atol),
        "OPT_TARGET_RTOL": str(args.rtol),
    }

    results = []
    for pad in args.padding:
        for i in indices:
            if not 0 <= i < len(shapes):
                continue
            shape = shapes[i]
            tag = label(i + 1, shape) + ("_pad" if pad else "")
            cmd = command(shape, pad, args)

            started = time.time()
            try:
                run = subprocess.run(
                    cmd, capture_output=True, text=True,
                    env=env, timeout=args.timeout,
                )
                out, err, code = run.stdout, run.stderr, run.returncode
            except subprocess.TimeoutExpired:
                out, err, code = "", f"timed out after {args.timeout}s", -1
            elapsed = time.time() - started

            (args.out / f"{tag}.log").write_text(
                f"$ {' '.join(cmd)}\nexit={code}\nelapsed={elapsed:.0f}s\n\n"
                f"{out}\n--- STDERR ---\n{err}"
            )

            speed = SPEEDUP_RE.search(out)
            summary = SUMMARY_RE.search(out)
            route = ROUTE_RE.search(out)
            prec = PREC_RE.search(out)

            row_data = {
                "tag": tag,
                "speedup": float(speed.group(1)) if speed else None,
                "status": summary.group(1) if summary else "ERROR",
                "max_abs": float(summary.group(2)) if summary else None,
                "route": route.group(1) if route else "-",
                "precision": prec.group(1).strip() if prec else "-",
                "elapsed": elapsed,
            }
            results.append(row_data)
            # Built with plain concatenation, not nested f-strings: Colab has
            # shipped several Python versions and PEP 701 quote reuse only
            # works from 3.12.
            sp = "%.3fx" % row_data["speedup"] if row_data["speedup"] else "--"
            ma = ("%.3g" % row_data["max_abs"]
                  if row_data["max_abs"] is not None else "--")
            print(
                "%-26s %6.0fs  %9s  %-5s  max_abs=%10s  route=%s"
                % (tag, elapsed, sp, row_data["status"], ma, row_data["route"])
            )

    ok = [r for r in results if r["speedup"] and r["status"] == "PASS"]
    lines = [
        "| shape | speedup | status | max_abs | route |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        sp = "%.3fx" % r["speedup"] if r["speedup"] else "--"
        ma = "%.3g" % r["max_abs"] if r["max_abs"] is not None else "--"
        lines.append(
            "| `%s` | %s | %s | %s | %s |"
            % (r["tag"], sp, r["status"], ma, r["route"])
        )

    if ok:
        # Geometric mean: these are ratios, so the arithmetic mean would
        # overweight the big wins and overstate the result.
        product = 1.0
        for r in ok:
            product *= r["speedup"]
        geo = product ** (1.0 / len(ok))
        low = min(r["speedup"] for r in ok)
        high = max(r["speedup"] for r in ok)
        lines += [
            "",
            "**%d/%d passing** | geometric mean **%.3fx** | range %.3fx - %.3fx"
            % (len(ok), len(results), geo, low, high),
        ]

    table = "\n".join(lines)
    (args.out / "summary.md").write_text(table + "\n")
    print("\n" + table)
    print(f"\nlogs and summary.md in {args.out}")
    return 0 if all(r["status"] == "PASS" for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
