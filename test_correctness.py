"""Correctness tests for the exact supplied Transformer benchmark.

These tests compare UserOptimizedTransformer against BaselineTransformer using
the benchmark's own output comparison rule.  On a CUDA + Triton machine they
exercise the small, medium and large dispatcher paths.  On a CPU-only machine
this file exits cleanly because the custom Triton kernels cannot run there.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Import the benchmark as a module without executing its CLI main().
BENCHMARK_DIR = Path(__file__).resolve().parent.parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from torch_transformer_benchmark import (  # noqa: E402
    BaselineTransformer,
    TransformerConfig,
    UserOptimizedTransformer,
    compare_outputs,
    copy_model_weights,
    generate_random_case,
)
from dispatcher import attention as triton_attention  # noqa: E402


ATOL = 0.001
RTOL = 0.01


def check_transformer(
    seq_len: int,
    *,
    causal: bool,
    padding_ratio: float,
    dtype: torch.dtype,
) -> None:
    """Build identical models and compare their complete Transformer outputs."""

    config = TransformerConfig(
        batch_size=2,
        seq_len=seq_len,
        d_model=512,
        num_heads=8,
        ffn_dim=2048,
        num_layers=2,
        causal=causal,
    )

    baseline = BaselineTransformer(config).cuda().to(dtype=dtype).eval()
    optimized = UserOptimizedTransformer(config).cuda().to(dtype=dtype).eval()
    copy_model_weights(baseline, optimized)

    x, valid_mask = generate_random_case(
        config,
        device=torch.device("cuda"),
        dtype=dtype,
        seed=1234 + seq_len,
        padding_ratio=padding_ratio,
        input_scale=1.0,
    )

    with torch.inference_mode():
        reference = baseline(x, valid_mask)
        candidate = optimized(x, valid_mask)

    result = compare_outputs(reference, candidate, rtol=RTOL, atol=ATOL)
    status = "PASS" if result.passed else "FAIL"
    print(
        f"full transformer N={seq_len:4d} causal={str(causal):5s} "
        f"padding={padding_ratio:.2f}: {status} "
        f"max_abs={result.max_abs_error:.6g} "
        f"max_rel={result.max_relative_error:.6g}"
    )
    if not result.passed:
        raise AssertionError(
            f"Transformer correctness failed for N={seq_len}, causal={causal}"
        )


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is required; no Triton correctness tests were run.")
        return 0

    # These sequence lengths intentionally hit all three dispatcher families.
    for seq_len in (32, 128, 1024):
        check_transformer(
            seq_len,
            causal=False,
            padding_ratio=0.0,
            dtype=torch.float16,
        )

    # Exercise the benchmark's two masking requirements as well.
    check_transformer(
        128,
        causal=True,
        padding_ratio=0.25,
        dtype=torch.float16,
    )

    # Also test the low-precision route that is often relevant for Tensor Cores.
    check_transformer(
        128,
        causal=False,
        padding_ratio=0.0,
        dtype=torch.bfloat16,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
