#!/usr/bin/env python3
"""
Shared plumbing for the kernel-opt harness.

Everything here is deliberately thin: it imports the *existing* benchmark
module (torch_transformer_benchmark.py) rather than re-implementing any of it,
so the reference model, the random-case generator and the equivalence check
stay single-sourced. If the organizers update that file, the harness follows.

Contents:
  * repo/import bootstrap
  * SHAPES: the fixed shape registry (parallel + recurrent modes)
  * build_pair(): baseline + candidate with identical weights
  * candidate implementations ("torch_ref", "sdpa", "cuda")
  * load_cuda_extension(): JIT-compiles kernels/current.cu
  * append_lineage(): append-only lineage.jsonl writer
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------
# Paths / bootstrap
# --------------------------------------------------------------------------

HARNESS_DIR = Path(__file__).resolve().parent
KERNEL_OPT_DIR = HARNESS_DIR.parent
REPO_ROOT = KERNEL_OPT_DIR.parent

KERNELS_DIR = KERNEL_OPT_DIR / "kernels"
TOOLS_DIR = KERNEL_OPT_DIR / "tools"
LINEAGE_PATH = KERNEL_OPT_DIR / "lineage.jsonl"
CURRENT_KERNEL = KERNELS_DIR / "current.cu"
BENCHMARK_MODULE_PATH = REPO_ROOT / "torch_transformer_benchmark.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class HarnessError(RuntimeError):
    """Raised for environment problems the harness cannot work around."""


def import_reference_module():
    """Import the organizers' benchmark script as `tb`.

    Kept as a function (not a top-level import) so that --list-shapes and
    other metadata paths work on a machine without torch installed.
    """
    if not BENCHMARK_MODULE_PATH.exists():
        raise HarnessError(
            f"reference benchmark not found at {BENCHMARK_MODULE_PATH}; "
            "the harness must sit in <repo>/kernel-opt/harness/"
        )
    try:
        import torch_transformer_benchmark as tb  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise HarnessError(
            f"could not import torch_transformer_benchmark ({exc}). "
            "Is PyTorch installed in this interpreter?"
        ) from exc
    return tb


# --------------------------------------------------------------------------
# Fixed shape registry
# --------------------------------------------------------------------------
#
# NOTE: the official Test Shapes appendix (problem statement 3.7) lives in a
# Feishu doc that was not available when this was written. These shapes are
# derived from the benchmark's argparse defaults plus the axes the problem
# statement names explicitly (large/small batch, seq_len, d_model). Replace the
# entries below with the official list when you have it -- everything else in
# the harness keys off `id`, so nothing else needs to change.
#
# mode="parallel"  : prefill / chunked-parallel regime. Long sequences, full
#                    S x S attention, compute- and bandwidth-bound.
# mode="recurrent" : decode / incremental regime. seq_len 1-8, dominated by
#                    kernel-launch overhead and weight-load bandwidth.
#
# `dtype` is the tensor dtype used for BOTH baseline and candidate. `causal`
# and `padding_ratio` map onto the reference model's two masking paths.

SHAPES: List[Dict[str, Any]] = [
    # ---- parallel / prefill ------------------------------------------------
    {
        "id": "P0_default",
        "mode": "parallel",
        "note": "benchmark argparse defaults; the canonical smoke shape",
        "batch_size": 8, "seq_len": 128, "d_model": 512, "num_heads": 8,
        "ffn_dim": 2048, "num_layers": 6, "causal": False,
        "dtype": "float32", "padding_ratio": 0.0,
    },
    {
        "id": "P1_long_seq",
        "mode": "parallel",
        "note": "long sequence, causal; attention dominates over FFN",
        "batch_size": 4, "seq_len": 1024, "d_model": 512, "num_heads": 8,
        "ffn_dim": 2048, "num_layers": 6, "causal": True,
        "dtype": "float16", "padding_ratio": 0.0,
    },
    {
        "id": "P2_large_batch",
        "mode": "parallel",
        "note": "large batch, medium seq; occupancy-friendly, FFN-heavy",
        "batch_size": 32, "seq_len": 256, "d_model": 768, "num_heads": 12,
        "ffn_dim": 3072, "num_layers": 6, "causal": False,
        "dtype": "float16", "padding_ratio": 0.0,
    },
    {
        "id": "P3_wide_model",
        "mode": "parallel",
        "note": "wide d_model + very long seq; peak memory pressure on 8GB",
        "batch_size": 2, "seq_len": 2048, "d_model": 1024, "num_heads": 16,
        "ffn_dim": 4096, "num_layers": 4, "causal": True,
        "dtype": "bfloat16", "padding_ratio": 0.0,
    },
    {
        "id": "P4_ragged_padding",
        "mode": "parallel",
        "note": "30% padding; exercises the valid_token_mask path end-to-end",
        "batch_size": 8, "seq_len": 512, "d_model": 512, "num_heads": 8,
        "ffn_dim": 2048, "num_layers": 6, "causal": False,
        "dtype": "float16", "padding_ratio": 0.3,
    },
    # ---- recurrent / decode ------------------------------------------------
    #
    # Caveat, stated plainly: the reference model has no KV cache, so a
    # seq_len=1 call recomputes a 1x1 attention rather than attending to
    # history. These shapes therefore measure the *launch-bound, memory-bound
    # small-tensor regime* -- which is the thing worth optimizing for decode --
    # but they are not a true incremental-decode benchmark. If you add a KV
    # cache to the candidate, add matching shapes here.
    {
        "id": "R0_single_token",
        "mode": "recurrent",
        "note": "batch 1, one token: pure launch-overhead / latency floor",
        "batch_size": 1, "seq_len": 1, "d_model": 512, "num_heads": 8,
        "ffn_dim": 2048, "num_layers": 6, "causal": True,
        "dtype": "float16", "padding_ratio": 0.0,
    },
    {
        "id": "R1_batched_token",
        "mode": "recurrent",
        "note": "batched decode: GEMV-shaped projections, bandwidth-bound",
        "batch_size": 32, "seq_len": 1, "d_model": 512, "num_heads": 8,
        "ffn_dim": 2048, "num_layers": 6, "causal": True,
        "dtype": "float16", "padding_ratio": 0.0,
    },
    {
        "id": "R2_short_chunk",
        "mode": "recurrent",
        "note": "speculative-decode sized chunk; 8 tokens x large batch",
        "batch_size": 64, "seq_len": 8, "d_model": 512, "num_heads": 8,
        "ffn_dim": 2048, "num_layers": 6, "causal": True,
        "dtype": "float16", "padding_ratio": 0.0,
    },
]

SHAPES_BY_ID: Dict[str, Dict[str, Any]] = {s["id"]: s for s in SHAPES}


def select_shapes(
    ids: Optional[List[str]] = None,
    mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Resolve a --shapes / --mode selection into shape dicts."""
    if ids:
        missing = [i for i in ids if i not in SHAPES_BY_ID]
        if missing:
            raise HarnessError(
                f"unknown shape id(s): {missing}. "
                f"Known: {sorted(SHAPES_BY_ID)}"
            )
        chosen = [SHAPES_BY_ID[i] for i in ids]
    else:
        chosen = list(SHAPES)
    if mode and mode != "all":
        chosen = [s for s in chosen if s["mode"] == mode]
    if not chosen:
        raise HarnessError(f"no shapes matched (ids={ids}, mode={mode})")
    return chosen


def shape_to_config(shape: Dict[str, Any]):
    """Build the reference module's TransformerConfig from a shape dict."""
    tb = import_reference_module()
    cfg = tb.TransformerConfig(
        batch_size=shape["batch_size"],
        seq_len=shape["seq_len"],
        d_model=shape["d_model"],
        num_heads=shape["num_heads"],
        ffn_dim=shape["ffn_dim"],
        num_layers=shape["num_layers"],
        causal=shape["causal"],
    )
    cfg.validate()
    return cfg


# --------------------------------------------------------------------------
# Candidate implementations
# --------------------------------------------------------------------------

IMPLEMENTATIONS = ("torch_ref", "sdpa", "cuda")

_EXTENSION_CACHE: Dict[str, Any] = {}


_KERNEL_HASH_CACHE: Dict[Path, str] = {}


def kernel_hash(path: Path = CURRENT_KERNEL, *, refresh: bool = False) -> Optional[str]:
    """SHA256 of the kernel source, so lineage rows are pinned to a build.

    Cached per resolved path for the process lifetime. `_CudaAttention.forward`
    calls `load_cuda_extension()` -> `kernel_hash()` once per attention layer,
    every forward pass; re-reading and re-hashing the source that often is
    measurable overhead on a slow filesystem (this repo is on a 9p-mounted
    /mnt/c path under WSL2, ~2.5ms per hash -- see experiments.md Discovery
    D4), and was previously inflating every latency measurement by
    ~num_layers x 2.5ms regardless of the kernel's actual GPU cost. Pass
    refresh=True if the source may have changed within a long-lived process.
    """
    resolved = path.resolve()
    if not refresh and resolved in _KERNEL_HASH_CACHE:
        return _KERNEL_HASH_CACHE[resolved]
    if not path.exists():
        return None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    _KERNEL_HASH_CACHE[resolved] = digest
    return digest


def load_cuda_extension(verbose: bool = False, rebuild: bool = False):
    """JIT-compile kernels/current.cu via torch.utils.cpp_extension.load.

    Cached per source hash so repeated harness runs in one process (and, via
    the on-disk build dir, across processes) don't recompile.
    """
    import torch
    from torch.utils.cpp_extension import load

    if not CURRENT_KERNEL.exists():
        raise HarnessError(f"kernel source not found: {CURRENT_KERNEL}")

    digest = kernel_hash(refresh=rebuild)
    if not rebuild and digest in _EXTENSION_CACHE:
        return _EXTENSION_CACHE[digest]

    build_dir = KERNELS_DIR / "_build" / str(digest)
    build_dir.mkdir(parents=True, exist_ok=True)

    # Target the local device only; keeps compile time down in the loop.
    arch_flags = []
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        arch_flags = [f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"]

    module = load(
        name=f"kernel_opt_current_{digest}",
        sources=[str(CURRENT_KERNEL)],
        build_directory=str(build_dir),
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"] + arch_flags,
        extra_cflags=["-O3"],
        verbose=verbose,
    )
    _EXTENSION_CACHE[digest] = module
    return module


def build_pair(
    shape: Dict[str, Any],
    impl: str,
    device,
    dtype,
    seed: int = 1234,
):
    """Construct (baseline, candidate) with byte-identical weights.

    Returns the reference BaselineTransformer and a candidate module whose
    forward signature matches it exactly.
    """
    import torch

    tb = import_reference_module()
    if impl not in IMPLEMENTATIONS:
        raise HarnessError(f"unknown impl {impl!r}; expected one of {IMPLEMENTATIONS}")

    config = shape_to_config(shape)
    torch.manual_seed(seed)

    baseline = tb.BaselineTransformer(config)

    if impl == "torch_ref":
        candidate = tb.UserOptimizedTransformer(config)
    elif impl == "sdpa":
        candidate = SdpaTransformer(config)
    else:
        candidate = CudaFusedTransformer(config)

    # Reuse the organizers' weight copy so parameter-name compatibility is
    # enforced the same way the official script enforces it.
    tb.copy_model_weights(baseline, candidate, strict=True)

    baseline = baseline.to(device=device, dtype=dtype).eval()
    candidate = candidate.to(device=device, dtype=dtype).eval()
    return baseline, candidate


def _make_candidate_classes():
    """Define candidate model classes lazily (they subclass torch modules)."""
    tb = import_reference_module()
    import torch
    import torch.nn.functional as F

    class _SdpaAttention(tb.BaselineSelfAttention):
        """Reference math, but routed through F.scaled_dot_product_attention.

        This is the 'free' baseline optimization -- useful as the control that
        every custom kernel must beat, not as the end goal.
        """

        def forward(self, x, valid_token_mask=None, causal=False):
            batch, seq_len, _ = x.shape
            q = self._split_heads(self.q_proj(x))
            k = self._split_heads(self.k_proj(x))
            v = self._split_heads(self.v_proj(x))

            attn_mask = None
            if valid_token_mask is not None:
                # [B, 1, 1, S] boolean: True = attend.
                attn_mask = valid_token_mask[:, None, None, :]
                if causal:
                    causal_ok = torch.ones(
                        (seq_len, seq_len), device=x.device, dtype=torch.bool
                    ).tril()
                    attn_mask = attn_mask & causal_ok
                    causal = False

            context = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=causal, scale=self.scale
            )
            context = (
                context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
            )
            output = self.out_proj(context)
            if valid_token_mask is not None:
                output = output.masked_fill(~valid_token_mask[..., None], 0)
            return output

    class _CudaAttention(tb.BaselineSelfAttention):
        """Projections in torch; the attention core in kernels/current.cu."""

        def forward(self, x, valid_token_mask=None, causal=False):
            batch, seq_len, _ = x.shape
            q = self._split_heads(self.q_proj(x))
            k = self._split_heads(self.k_proj(x))
            v = self._split_heads(self.v_proj(x))

            ext = load_cuda_extension()
            mask = valid_token_mask if valid_token_mask is not None else torch.empty(0)
            context = ext.fused_attention(
                q.contiguous(), k.contiguous(), v.contiguous(),
                mask, bool(causal), float(self.scale),
            )
            context = (
                context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
            )
            output = self.out_proj(context)
            if valid_token_mask is not None:
                output = output.masked_fill(~valid_token_mask[..., None], 0)
            return output

    def _swap_attention(model, attn_cls):
        for layer in model.layers:
            old = layer.attention
            new = attn_cls(old.d_model, old.num_heads)
            layer.attention = new
        return model

    class _SdpaTransformer(tb.BaselineTransformer):
        def __init__(self, config):
            super().__init__(config)
            _swap_attention(self, _SdpaAttention)

    class _CudaFusedTransformer(tb.BaselineTransformer):
        def __init__(self, config):
            super().__init__(config)
            _swap_attention(self, _CudaAttention)

    return _SdpaTransformer, _CudaFusedTransformer


class _LazyClass:
    """Defers subclassing nn.Module until torch is known to be importable."""

    def __init__(self, index: int):
        self._index = index
        self._cls = None

    def __call__(self, *args, **kwargs):
        if self._cls is None:
            self._cls = _make_candidate_classes()[self._index]
        return self._cls(*args, **kwargs)


SdpaTransformer = _LazyClass(0)
CudaFusedTransformer = _LazyClass(1)


# --------------------------------------------------------------------------
# Environment fingerprint
# --------------------------------------------------------------------------

def env_fingerprint() -> Dict[str, Any]:
    """Everything needed to know whether two runs are comparable."""
    info: Dict[str, Any] = {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["sm"] = f"{props.major}.{props.minor}"
            info["sm_count"] = props.multi_processor_count
            info["vram_mb"] = round(props.total_memory / 1024 ** 2)
    except Exception as exc:  # torch missing or broken
        info["torch"] = None
        info["torch_error"] = str(exc)
    return info


def find_ncu() -> Optional[str]:
    """Locate the Nsight Compute CLI, or return None.

    Checked in order: $NCU_PATH, PATH, the standard Linux install location,
    and the Windows install dirs (reachable from WSL via /mnt/c).
    """
    explicit = os.environ.get("NCU_PATH")
    if explicit and Path(explicit).exists():
        return explicit

    from shutil import which

    found = which("ncu") or which("nv-nsight-cu-cli")
    if found:
        return found

    globs = [
        "/usr/local/cuda*/bin/ncu",
        "/opt/nvidia/nsight-compute/*/ncu",
        "/mnt/c/Program Files/NVIDIA Corporation/Nsight Compute*/ncu.exe",
        "/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/*/bin/ncu.exe",
    ]
    for pattern in globs:
        try:
            matches = sorted(Path("/").glob(pattern.lstrip("/")))
        except OSError:
            continue
        if matches:
            return str(matches[-1])
    return None


# --------------------------------------------------------------------------
# Lineage
# --------------------------------------------------------------------------

def append_lineage(record: Dict[str, Any], path: Optional[Path] = None) -> None:
    """Append one JSON object as a line. Never rewrites existing lines.

    `path` resolves at call time, not at import time, so tests and alternate
    runs can redirect LINEAGE_PATH without the default silently pinning the
    original file.
    """
    path = Path(path) if path is not None else LINEAGE_PATH
    record = dict(record)
    record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def read_lineage(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Read lineage.jsonl, skipping blank lines. Malformed lines raise.

    `path` resolves at call time; see append_lineage().
    """
    path = Path(path) if path is not None else LINEAGE_PATH
    if not path.exists():
        return []
    out = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise HarnessError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
    return out
