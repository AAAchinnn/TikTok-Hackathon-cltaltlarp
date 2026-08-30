"""Compute-precision policy for the optimized encoder.

Why this file exists
--------------------
On a Tesla T4 (sm75) there is no TF32 path, so an fp32 GEMM gets no Tensor
Core help at all. Narrowing selected GEMMs to fp16 is the only lever that
moves the GEMM-bound shapes. Measured on T4 at row 8 (b64 n128 d1024 h4 l4,
causal), where attention is only ~4% of the FLOPs and the rest is projection
GEMM:

    preset   max_abs    speedup
    fp32     3.1e-06    1.097x
    qkv      (unmeasured -- added after the sweep below)
    attn     9.75e-04   1.926x
    safe     1.14e-03   2.505x
    all      1.46e-03   5.115x

Narrowing is therefore not optional, but it cannot be unconditional either,
and which presets are admissible depends entirely on the gate. All four rows
above clear atol=2e-3. Only `attn` clears 1e-3, and only by 2%.

So the policy is measured, not asserted: `Calibrator` runs the narrow path
and the model's own fp32 path once per shape and keeps the fastest preset
whose disagreement stays inside CALIBRATION_MARGIN * TARGET_ATOL. It needs
no access to the baseline -- it is a self-consistency check -- and it runs
inside the untimed accuracy trials and warmup, so it costs nothing measured.

On tolerance
------------
The harness's own docstring states `atol=0.001, rtol=0.01`; its argparse
defaults are `0.002` / `0.02`. The two disagree, in the file as shipped.
Earlier revisions of this project resolved that by editing the defaults, which
hid the discrepancy inside a file that is supposed to stay pristine. It lives
here instead so the choice is visible, and `OPT_TARGET_ATOL` lets a run
calibrate against whatever gate it will actually be judged by.

That choice is worth more than any remaining kernel work on the GEMM-heavy
shapes: at 2e-3 row 8 calibrates to `safe` and runs 2.5x, at 1e-3 it finds
nothing wider than `attn` and may fall to fp32 and 1.10x. Ask the organisers
which gate applies before optimising against either.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import torch

__all__ = [
    "TARGET_ATOL",
    "TARGET_RTOL",
    "CALIBRATION_MARGIN",
    "NARROW_LADDER",
    "Settings",
    "settings",
    "Calibrator",
    "fp16_tensor_cores_available",
    "bf16_tensor_cores_available",
]


# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------

# Problem statement, not the harness default (which is 2e-3 / 2e-2). Override
# with OPT_TARGET_ATOL to calibrate against the gate a run will actually be
# judged by -- tools/sweep.py sets it whenever it passes --atol, so the two
# cannot drift apart.
#
# This is worth being deliberate about. On a T4 at d_model=1024 the measured
# errors are attn 9.8e-4, safe 1.14e-3, all 1.46e-3. Every one of those clears
# a 2e-3 gate; only `attn` clears 1e-3, and only by 2%. So the target chosen
# here decides whether that shape runs at 1.10x or 2.51x.
TARGET_ATOL = float(os.environ.get("OPT_TARGET_ATOL", 1e-3))
TARGET_RTOL = float(os.environ.get("OPT_TARGET_RTOL", 1e-2))

# What fraction of the budget a preset may spend and still be accepted.
#
# Calibration measures one input; the accuracy harness then runs five different
# ones through the same shape, so the margin has to cover input-to-input
# variation. Measured across trials, that variation is small: fp16/safe at
# d1024 ranged 7.62e-4 to 8.53e-4 (12%), fp16/all ranged 1.08e-3 to 1.16e-3
# (8%). 0.7 leaves roughly six times the observed spread in hand.
#
# An earlier value of 0.25 was chosen without checking it against those
# numbers, and rejected every preset on every shape -- the model silently ran
# fp32 everywhere and gave up 2.3x on d_model=1024. If you tighten this, check
# it against the measured errors above first.
CALIBRATION_MARGIN = float(os.environ.get("OPT_CALIBRATION_MARGIN", 0.7))


# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------
# Stage names match the five GEMMs in a block: qkv projection, attention
# itself, the attention output projection, and the two FFN matmuls.
#
# out_proj and ffn_out are the two GEMMs whose result is added straight into
# the residual stream, so their output rounding lands undiluted in the final
# comparison. That is the whole reason `safe` exists: sparing those two costs
# about half the available speedup (2.51x against 5.12x at d1024) and roughly
# halves the error (1.14e-3 against 1.46e-3).

NARROW_PRESETS: Dict[str, Tuple[str, ...]] = {
    "off": (),
    "qkv": ("qkv",),
    "attn": ("qkv", "attn"),
    "safe": ("qkv", "attn", "ffn_in"),
    "all": ("qkv", "attn", "out", "ffn_in", "ffn_out"),
}

# Tried in this order, fastest first. The calibrator keeps the first that fits.
# `qkv` exists as a rung between `attn` and giving up entirely: at a strict
# 1e-3 gate the wider presets are all rejected on the GEMM-heavy shapes, and
# without a narrow rung the ladder falls straight to fp32 and its 1.10x.
NARROW_LADDER: Tuple[str, ...] = ("all", "safe", "attn", "qkv", "off")


def fp16_tensor_cores_available(device: Optional[torch.device] = None) -> bool:
    """fp16 Tensor Cores exist from sm70 (Volta) onward."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 7


def bf16_tensor_cores_available(device: Optional[torch.device] = None) -> bool:
    """bf16 Tensor Cores need sm80 (Ampere).

    Below that, PyTorch will happily accept bfloat16 and emulate it. On a T4
    the measured result was 0.94x -- slower than doing nothing -- while also
    failing accuracy at max_abs=0.0072. Treating bf16 as unavailable here is
    what stops the ladder from ever selecting that trap.
    """
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 8


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "off": None}


@dataclass
class Settings:
    """Knobs, all optional.

    The general block's contract is that it needs no configuration, so every
    field here has a default that is correct everywhere. The environment
    overrides exist for sweeps and ablations, not for normal operation.
    """

    # None means "let the calibrator choose"; a dtype pins it.
    compute_dtype: Optional[torch.dtype] = None
    # None means "let the calibrator choose"; a name pins the preset.
    narrow_preset: Optional[str] = None
    calibrate: bool = True
    fp16_fast_reduce: bool = False
    use_compile: bool = True
    compile_mode: str = "default"
    fuse_qkv: bool = True
    elide_mask: bool = True
    # Verify rather than assume; see opt/masking.py.
    verify_suffix_padding: bool = True
    verbose: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        def flag(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            return default if raw is None else raw not in ("0", "false", "False")

        dtype_name = os.environ.get("OPT_COMPUTE_DTYPE", "").lower()
        preset = os.environ.get("OPT_NARROW", "").lower() or None
        if preset is not None and preset not in NARROW_PRESETS:
            preset = None

        return cls(
            compute_dtype=_DTYPES.get(dtype_name),
            narrow_preset=preset,
            # Pinning either one turns calibration off: the point of pinning is
            # to measure that exact configuration, which a ladder would undo.
            calibrate=flag(
                "OPT_CALIBRATE", dtype_name == "" and preset is None
            ),
            fp16_fast_reduce=flag("OPT_FP16_FAST_REDUCE", False),
            use_compile=flag("OPT_COMPILE", True),
            compile_mode=os.environ.get("OPT_COMPILE_MODE", "default"),
            fuse_qkv=flag("OPT_FUSE_QKV", True),
            elide_mask=flag("OPT_ELIDE_MASK", True),
            verify_suffix_padding=flag("OPT_VERIFY_SUFFIX", True),
            verbose=flag("OPT_VERBOSE", True),
        )

    def describe(self) -> str:
        if self.calibrate:
            compute = "calibrated"
        elif self.compute_dtype is None:
            compute = "off"
        else:
            name = str(self.compute_dtype).split(".")[-1]
            compute = f"{name}/{self.narrow_preset or 'all'}"
        compiled = (
            f"{self.compile_mode}" if self.use_compile else "off"
        )
        return (
            f"compute={compute} | compile={compiled} | "
            f"fuse_qkv={self.fuse_qkv} | elide_mask={self.elide_mask} | "
            f"verify_suffix={self.verify_suffix_padding}"
        )


settings = Settings.from_env()


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    """The precision decision for one shape."""

    compute_dtype: Optional[torch.dtype]
    preset: str
    measured_max_abs: float = 0.0
    reason: str = "default"

    @property
    def stages(self) -> Tuple[str, ...]:
        if self.compute_dtype is None:
            return ()
        return NARROW_PRESETS[self.preset]

    def dtype_for(self, stage: str, stream: torch.dtype) -> torch.dtype:
        """Which dtype stage `stage` should compute in, given the stream dtype."""
        if self.compute_dtype is None or stage not in self.stages:
            return stream
        return self.compute_dtype

    def __str__(self) -> str:
        if self.compute_dtype is None:
            return f"fp32 ({self.reason})"
        name = str(self.compute_dtype).split(".")[-1]
        return (
            f"{name}/{self.preset} "
            f"(max_abs={self.measured_max_abs:.3g}, {self.reason})"
        )


FP32_PLAN = Plan(compute_dtype=None, preset="off", reason="fp32")


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


class Calibrator:
    """Picks the fastest numerically-acceptable plan, per shape, by measuring.

    `runner(plan)` must evaluate the model under `plan` and return the output
    tensor. The caller owns that closure because only the model knows how to
    run itself; the calibrator only knows how to judge the result.
    """

    def __init__(
        self,
        atol: float = TARGET_ATOL,
        margin: float = CALIBRATION_MARGIN,
        ladder: Sequence[str] = NARROW_LADDER,
    ) -> None:
        self.atol = atol
        self.margin = margin
        self.ladder = tuple(ladder)
        self._plans: Dict[Tuple, Plan] = {}

    @property
    def budget(self) -> float:
        return self.atol * self.margin

    def cached(self, key: Tuple) -> Optional[Plan]:
        return self._plans.get(key)

    def plans(self) -> Dict[Tuple, Plan]:
        return dict(self._plans)

    def calibrate(self, key: Tuple, runner, stream_dtype: torch.dtype) -> Plan:
        """Return the plan for `key`, measuring it on first sight.

        One extra forward per rung, all of them inside warmup. The reference is
        the model's own fp32 output, so nothing here needs the baseline.
        """
        cached = self._plans.get(key)
        if cached is not None:
            return cached

        plan = self._measure(runner, stream_dtype)
        self._plans[key] = plan
        return plan

    def _measure(self, runner, stream_dtype: torch.dtype) -> Plan:
        # Narrowing an fp16/bf16 stream is meaningless -- it is already narrow,
        # and the harness's own reference runs in that dtype.
        if stream_dtype is not torch.float32:
            return Plan(None, "off", reason="stream is not fp32")
        if not fp16_tensor_cores_available():
            return Plan(None, "off", reason="no fp16 tensor cores")

        try:
            reference = runner(FP32_PLAN).detach().float()
        except Exception as exc:  # calibration must never break the model
            return Plan(None, "off", reason=f"reference failed: {type(exc).__name__}")

        for preset in self.ladder:
            if preset == "off":
                break
            candidate = Plan(torch.float16, preset, reason="calibrated")
            try:
                got = runner(candidate)
            except Exception:
                continue
            max_abs = float((got.detach().float() - reference).abs().max().item())
            if max_abs <= self.budget:
                return Plan(
                    torch.float16, preset, max_abs, reason="calibrated"
                )

        return Plan(None, "off", reason="no preset within budget")


def apply_backend_flags(compute_dtype: Optional[torch.dtype], stream: torch.dtype) -> None:
    """Force fp32 accumulation for fp16 GEMMs.

    fp16 matmuls may accumulate in fp16 by default, and that default has moved
    between torch versions -- so set it explicitly rather than inherit it. Only
    touched when the harness itself is running fp32; when the harness asks for
    fp16 end to end, changing this would alter the reference too.
    """
    if compute_dtype is not torch.float16 or stream is not torch.float32:
        return
    if settings.fp16_fast_reduce:
        return
    try:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    except Exception:
        pass
