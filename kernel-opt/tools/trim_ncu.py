#!/usr/bin/env python3
"""
Raw `ncu --set full` output  ->  a tight, prompt-sized metric bundle.

`--set full` emits thousands of metric rows across dozens of sections. Almost
all of it is noise for a mutation loop. This extracts the ~14 numbers that
actually discriminate between kernel variants, aggregates them per kernel, and
emits JSON small enough to paste into a prompt without crowding out the code.

What survives, and why each one earns its place
-----------------------------------------------
  duration_us            what you are minimizing
  compute_pct            SM throughput vs peak      -+ together these two say
  memory_pct             memory throughput vs peak  -+ compute- or mem-bound
  dram_gbps / dram_pct   achieved DRAM bandwidth vs the roofline ceiling
  arithmetic_intensity   FLOP/byte -- which side of the roofline ridge you are on
  achieved_occupancy     warps resident vs max
  theoretical_occupancy  the ceiling occupancy set by your launch config
                         (achieved << theoretical => tail effect / imbalance)
  l1_hit_pct / l2_hit_pct  is the data actually being reused
  sectors_per_request    coalescing. 4.0 is perfect for 32-bit; >4 = scattered
  tensor_pct             tensor-core pipe utilization (0 on an fp32 path)
  registers_per_thread   the usual occupancy limiter
  smem_per_block         the other usual occupancy limiter
  waves_per_sm           <1 means you cannot fill the GPU; quantization loss
  top_stalls             ranked warp stall reasons -- the single most useful
                         signal for "why is this slow", kept to the top 4

Anything not on that list is dropped on purpose. Resist growing it.

Usage
-----
  ncu --set full --csv --page raw python bench_child.py > raw.csv
  python trim_ncu.py raw.csv
  python trim_ncu.py raw.csv --kernel-filter attention --top-kernels 3
  cat raw.csv | python trim_ncu.py -
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Metric name -> output key.
#
# ncu metric names drift between CUDA versions, so each output key maps to a
# list of candidate metric names tried in order. Add aliases here rather than
# teaching the rest of the harness about version differences.
# --------------------------------------------------------------------------

METRIC_ALIASES: Dict[str, List[str]] = {
    "duration_us": [
        "gpu__time_duration.sum",
        "sm__cycles_elapsed.avg",  # fallback; converted below if needed
    ],
    "compute_pct": [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    ],
    "memory_pct": [
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    ],
    "dram_pct": [
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    ],
    "dram_bytes": [
        "dram__bytes.sum",
    ],
    "achieved_occupancy": [
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    ],
    "theoretical_occupancy": [
        "sm__maximum_warps_per_active_cycle_pct",
    ],
    "l1_hit_pct": [
        "l1tex__t_sector_hit_rate.pct",
    ],
    "l2_hit_pct": [
        "lts__t_sector_hit_rate.pct",
    ],
    "sectors_per_request": [
        "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    ],
    "tensor_pct": [
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    ],
    "registers_per_thread": [
        "launch__registers_per_thread",
    ],
    "smem_per_block": [
        "launch__shared_mem_per_block",
        "launch__shared_mem_per_block_static",
    ],
    "waves_per_sm": [
        "launch__waves_per_multiprocessor",
    ],
    "grid_size": ["launch__grid_size"],
    "block_size": ["launch__block_size"],
    "fp32_inst": ["smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
                  "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
                  "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"],
}

# Warp stall reasons. ncu names these
# smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio
STALL_RE = re.compile(
    r"smsp__average_warps?_issue_stalled_(?P<reason>[a-z0-9_]+?)_per_issue_active"
)

# Human-readable gloss for the stall reasons that actually show up, so the
# metric bundle is self-explanatory in a prompt without a lookup table.
STALL_GLOSS = {
    "long_scoreboard": "waiting on global/local memory (the classic mem-bound stall)",
    "short_scoreboard": "waiting on shared memory / MIO queue",
    "wait": "fixed-latency ALU dependency; needs more ILP",
    "barrier": "blocked at __syncthreads(); warp imbalance in the block",
    "membar": "memory barrier",
    "imc_miss": "immediate-constant cache miss",
    "mio_throttle": "MIO instruction queue full (often heavy shared-memory traffic)",
    "lg_throttle": "local/global instruction queue full",
    "tex_throttle": "texture/L1 pipe throttle",
    "drain": "draining at kernel exit",
    "dispatch_stall": "dispatch stall",
    "no_instruction": "instruction-cache miss / fetch starvation",
    "selected": "(issued this cycle -- not a stall)",
    "sleeping": "warp asleep",
    "misc": "miscellaneous",
}

# Roofline ceilings. Only needed to turn achieved GB/s into a % of peak when
# ncu does not report dram_pct directly. Keyed by a substring of the GPU name.
DEVICE_PEAKS = {
    "RTX 3070": {"dram_gbps": 448.0, "fp32_tflops": 20.3},
    "RTX 3080": {"dram_gbps": 760.0, "fp32_tflops": 29.8},
    "RTX 3090": {"dram_gbps": 936.0, "fp32_tflops": 35.6},
    "RTX 4090": {"dram_gbps": 1008.0, "fp32_tflops": 82.6},
    "A100": {"dram_gbps": 1555.0, "fp32_tflops": 19.5},
    "H100": {"dram_gbps": 3350.0, "fp32_tflops": 67.0},
}


def _to_float(text: str) -> Optional[float]:
    """ncu CSV numbers may carry thousands separators or be 'n/a'."""
    if text is None:
        return None
    cleaned = text.strip().strip('"').replace(",", "")
    if not cleaned or cleaned.lower() in {"n/a", "na", "nan", "-"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_csv(raw: str) -> List[Dict[str, str]]:
    """Pull the metric rows out of ncu's CSV output.

    ncu prefixes the CSV with ==PROF== banner lines and may interleave
    warnings, so we locate the header row rather than assuming line 0.
    """
    lines = raw.splitlines()
    header_index = None
    for index, line in enumerate(lines):
        if line.startswith("=="):
            continue
        lowered = line.lower()
        if '"id"' in lowered or lowered.startswith("id,"):
            if "metric name" in lowered:
                header_index = index
                break
    if header_index is None:
        return []

    body = "\n".join(
        line for line in lines[header_index:] if not line.startswith("==")
    )
    return list(csv.DictReader(io.StringIO(body)))


def _column(row: Dict[str, str], *names: str) -> Optional[str]:
    """Case-insensitive column lookup; ncu capitalizes inconsistently."""
    lowered = {k.lower().strip(): v for k, v in row.items() if k}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def group_by_kernel(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    """Collapse the flat metric rows into {kernel_name: {metric: value}}.

    A kernel launched many times appears many times; we average the metric
    across launches and keep the launch count.
    """
    kernels: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        name = _column(row, "Kernel Name", "kernel name")
        metric = _column(row, "Metric Name", "metric name")
        value = _to_float(_column(row, "Metric Value", "metric value") or "")
        unit = (_column(row, "Metric Unit", "metric unit") or "").strip()
        if not name or not metric:
            continue
        entry = kernels.setdefault(
            name.strip(), {"_sums": {}, "_counts": {}, "_units": {}, "launches": set()}
        )
        launch_id = _column(row, "ID", "id")
        if launch_id is not None:
            entry["launches"].add(launch_id)
        if value is None:
            continue
        metric = metric.strip()
        entry["_sums"][metric] = entry["_sums"].get(metric, 0.0) + value
        entry["_counts"][metric] = entry["_counts"].get(metric, 0) + 1
        entry["_units"][metric] = unit
    return kernels


def _mean(entry: Dict[str, Any], metric: str) -> Optional[float]:
    count = entry["_counts"].get(metric)
    if not count:
        return None
    return entry["_sums"][metric] / count


def _first_available(entry: Dict[str, Any], keys: List[str]) -> Tuple[Optional[float], Optional[str]]:
    for key in keys:
        value = _mean(entry, key)
        if value is not None:
            return value, key
    return None, None


def extract_stalls(entry: Dict[str, Any], top_n: int = 4) -> List[Dict[str, Any]]:
    """Rank warp stall reasons by warps-stalled-per-issue-active."""
    found: Dict[str, float] = {}
    for metric in entry["_counts"]:
        match = STALL_RE.search(metric)
        if not match:
            continue
        reason = match.group("reason")
        if reason in {"selected", "sleeping"}:
            continue
        value = _mean(entry, metric)
        if value is not None:
            found[reason] = max(found.get(reason, 0.0), value)

    total = sum(found.values()) or 1.0
    ranked = sorted(found.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [
        {
            "reason": reason,
            "warps_per_issue": round(value, 3),
            "pct_of_stalls": round(100.0 * value / total, 1),
            "means": STALL_GLOSS.get(reason, ""),
        }
        for reason, value in ranked
    ]


def summarize_kernel(name: str, entry: Dict[str, Any], gpu: Optional[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"kernel": name, "launches": len(entry["launches"]) or None}

    for key, aliases in METRIC_ALIASES.items():
        if key in {"dram_bytes", "fp32_inst"}:
            continue
        value, source = _first_available(entry, aliases)
        if value is None:
            continue
        if key == "duration_us":
            unit = entry["_units"].get(source, "")
            if unit in {"nsecond", "ns"}:
                value /= 1000.0
            elif unit in {"msecond", "ms"}:
                value *= 1000.0
            elif unit in {"second", "s"}:
                value *= 1e6
        out[key] = round(value, 4)

    # ---- roofline -------------------------------------------------------
    dram_bytes, _ = _first_available(entry, METRIC_ALIASES["dram_bytes"])
    duration_us = out.get("duration_us")
    if dram_bytes and duration_us:
        gbps = dram_bytes / (duration_us * 1e-6) / 1e9
        out["dram_gbps"] = round(gbps, 1)
        peak = None
        if gpu:
            for token, values in DEVICE_PEAKS.items():
                if token.lower() in gpu.lower():
                    peak = values["dram_gbps"]
                    break
        if peak:
            out["dram_peak_gbps"] = peak
            out["dram_pct_of_roofline"] = round(100.0 * gbps / peak, 1)

    # Arithmetic intensity: FLOP per DRAM byte. FADD/FMUL = 1 flop,
    # FFMA = 2. Only meaningful on fp32 paths; tensor-core kernels report
    # near-zero here and should be read via tensor_pct instead.
    if dram_bytes:
        flops = 0.0
        for metric in METRIC_ALIASES["fp32_inst"]:
            value = _mean(entry, metric)
            if value:
                flops += value * (2.0 if "ffma" in metric else 1.0)
        if flops:
            out["arithmetic_intensity"] = round(flops / dram_bytes, 3)

    stalls = extract_stalls(entry)
    if stalls:
        out["top_stalls"] = stalls

    # ---- one-line verdict ----------------------------------------------
    compute = out.get("compute_pct")
    memory = out.get("memory_pct")
    if compute is not None and memory is not None:
        if max(compute, memory) < 25:
            verdict = "latency-bound (both SM and memory throughput low)"
        elif compute > memory + 10:
            verdict = "compute-bound"
        elif memory > compute + 10:
            verdict = "memory-bound"
        else:
            verdict = "balanced / possibly latency-bound"
        out["bound_by"] = verdict

    return out


def trim(
    raw: str,
    kernel_filter: Optional[str] = None,
    top_kernels: int = 5,
    gpu: Optional[str] = None,
) -> Dict[str, Any]:
    """Main entry point. Raw ncu text in, compact metric dict out."""
    rows = parse_csv(raw)
    if not rows:
        return {
            "error": "no ncu CSV metric rows found",
            "hint": "run ncu with --csv --page raw",
            "raw_head": raw[:400],
        }

    if gpu is None:
        match = re.search(r"Device\s+.*?\((.*?)\)", raw)
        gpu = match.group(1) if match else None

    kernels = group_by_kernel(rows)
    if kernel_filter:
        pattern = re.compile(kernel_filter, re.IGNORECASE)
        kernels = {k: v for k, v in kernels.items() if pattern.search(k)}

    summaries = [summarize_kernel(name, entry, gpu) for name, entry in kernels.items()]
    # Rank by total time (duration x launches): the kernel worth optimizing.
    summaries.sort(
        key=lambda s: (s.get("duration_us") or 0) * (s.get("launches") or 1),
        reverse=True,
    )
    summaries = summaries[:top_kernels]

    total_us = sum(
        (s.get("duration_us") or 0) * (s.get("launches") or 1) for s in summaries
    )
    hottest = summaries[0] if summaries else {}

    return {
        "gpu": gpu,
        "kernels_profiled": len(kernels),
        "total_kernel_us": round(total_us, 2),
        "summary": {
            "hottest_kernel": hottest.get("kernel"),
            "bound_by": hottest.get("bound_by"),
            "compute_pct": hottest.get("compute_pct"),
            "memory_pct": hottest.get("memory_pct"),
            "achieved_occupancy": hottest.get("achieved_occupancy"),
            "dram_pct_of_roofline": hottest.get("dram_pct_of_roofline"),
            "top_stall": (hottest.get("top_stalls") or [{}])[0].get("reason"),
        },
        "kernels": summaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trim raw `ncu --set full` output to 10-15 key metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="raw ncu output file, or - for stdin")
    parser.add_argument("--kernel-filter", default=None,
                        help="regex; keep only matching kernel names")
    parser.add_argument("--top-kernels", type=int, default=5)
    parser.add_argument("--gpu", default=None,
                        help="GPU name override for roofline peaks")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = sys.stdin.read() if args.input == "-" else open(
        args.input, "r", encoding="utf-8", errors="replace"
    ).read()

    result = trim(raw, args.kernel_filter, args.top_kernels, args.gpu)
    payload = json.dumps(result, indent=2, sort_keys=False)
    print(payload)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
