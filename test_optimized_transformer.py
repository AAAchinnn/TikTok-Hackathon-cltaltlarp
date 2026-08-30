"""
Basic correctness test for UserOptimizedTransformer.

Place this file next to torch_transformer_benchmark.py and run:

    python test_optimized_transformer.py

It copies the benchmark model's weights into the optimized model and compares
outputs for several small cases, including causal and padded inputs.
"""

from __future__ import annotations

import os
import sys

import torch

# Make sure the benchmark file can be imported when this test is run from the
# repository root or from this directory.
ROOT = os.path.dirname(os.path.abspath(__file__))

# Add both this directory and its parent to the import path.  This supports
# either layout:
#
#   repo/
#       torch_transformer_benchmark.py
#       optimized_transformer.py
#       test_optimized_transformer.py
#
# or a small subdirectory containing these two files.
for candidate in (ROOT, os.path.dirname(ROOT), os.getcwd()):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from optimized_transformer import UserOptimizedTransformer  # noqa: E402

# The supplied benchmark imports its own dispatcher for the placeholder
# UserOptimizedTransformer.  We do not instantiate that placeholder here, so
# a tiny import-only shim avoids requiring your team's dispatcher package just
# to run this standalone test.
import types
if "dispatcher" not in sys.modules:
    dispatcher_shim = types.ModuleType("dispatcher")
    dispatcher_shim.attention = None
    dispatcher_shim.select_path = lambda *args, **kwargs: "fallback"
    sys.modules["dispatcher"] = dispatcher_shim

try:
    from torch_transformer_benchmark import BaselineTransformer, TransformerConfig  # noqa: E402
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Could not import torch_transformer_benchmark.py. Place the supplied "
        "benchmark where Python can import it (for example, the repository root)."
    ) from exc


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float]:
    """Return max absolute and max relative error."""
    ref = reference.float()
    cand = candidate.float()
    abs_error = (cand - ref).abs()
    rel_error = abs_error / ref.abs().clamp_min(1e-12)
    return float(abs_error.max()), float(rel_error.max())


def run_case(
    *,
    seq_len: int,
    causal: bool,
    padding_ratio: float,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Run one baseline-vs-optimized correctness case."""
    config = TransformerConfig(
        batch_size=2,
        seq_len=seq_len,
        d_model=64,
        num_heads=4,
        ffn_dim=256,
        num_layers=2,
        causal=causal,
    )

    torch.manual_seed(1234)

    baseline = BaselineTransformer(config).to(device=device, dtype=dtype).eval()
    optimized = UserOptimizedTransformer(config).to(device=device, dtype=dtype).eval()

    # Make the optimized model identical to the baseline model so that the
    # only difference is the implementation of attention.
    optimized.load_state_dict(baseline.state_dict(), strict=True)

    x = torch.randn(
        config.batch_size,
        config.seq_len,
        config.d_model,
        device=device,
        dtype=dtype,
    )

    if padding_ratio == 0:
        mask = torch.ones(
            config.batch_size,
            config.seq_len,
            device=device,
            dtype=torch.bool,
        )
    else:
        valid_len = max(1, int(round(seq_len * (1.0 - padding_ratio))))
        mask = torch.zeros(
            config.batch_size,
            config.seq_len,
            device=device,
            dtype=torch.bool,
        )
        mask[:, :valid_len] = True

    with torch.inference_mode():
        reference = baseline(x, mask)
        candidate = optimized(x, mask)

    max_abs, max_rel = compare(reference, candidate)

    # These are deliberately aligned with the benchmark's default tolerances.
    passed = torch.allclose(reference.float(), candidate.float(), atol=0.001, rtol=0.01)

    status = "PASS" if passed else "FAIL"
    print(
        f"{status} | seq={seq_len:4d} | causal={str(causal):5s} | "
        f"padding={padding_ratio:.1f} | max_abs={max_abs:.6g} | "
        f"max_rel={max_rel:.6g}"
    )

    if not passed:
        raise AssertionError(
            f"optimized output failed accuracy check: "
            f"max_abs={max_abs:.6g}, max_rel={max_rel:.6g}"
        )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"device={device}, dtype={dtype}")

    for seq_len in (32, 128, 256):
        for causal in (False, True):
            run_case(
                seq_len=seq_len,
                causal=causal,
                padding_ratio=0.0,
                dtype=dtype,
                device=device,
            )

    # One padded case makes sure the query/key masking behavior is covered.
    run_case(
        seq_len=128,
        causal=True,
        padding_ratio=0.25,
        dtype=dtype,
        device=device,
    )

    print("all basic correctness tests: PASS")


if __name__ == "__main__":
    main()
