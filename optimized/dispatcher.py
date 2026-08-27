"""
Shape-aware dispatcher for the Transformer GPU kernel task.

Given an input, chooses which registered candidate implementation runs in each
slot of the encoder. Resolution happens once per unique shape and is cached, so
the steady-state cost is a single dict lookup.

Three rules here are load-bearing. Do not relax them:

  1. The Key is built ONLY from values the CPU already knows - tensor shapes,
     dtypes, config flags. Never from tensor *contents*. Reading a value off the
     GPU (e.g. `if mask.all()`) forces a synchronisation that drains the launch
     pipeline, costing far more than any routing decision can win back.

  2. Unknown shapes fall back to a safe plan instead of raising. The 14 official
     shape combinations are known, but dtype, padding ratio and tolerance are
     not pinned, so unmeasured keys will happen.

  3. The routing table is data, not code. bench/autotune.py regenerates it by
     measuring every candidate against every shape behind a correctness gate.
     Nothing in this file is hand-tuned, and no thresholds are hardcoded.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch


# --------------------------------------------------------------------------
# Slots
# --------------------------------------------------------------------------
# A slot is a swappable position in the encoder pipeline. `full_block` covers
# both of the others; when a plan supplies one, the encoder ignores attn/ffn.

SLOTS = ("attn_block", "ffn_block", "full_block")

_REGISTRY: Dict[str, Dict[str, Callable]] = {slot: {} for slot in SLOTS}


def register(slot: str, name: str):
    """Decorator that adds a candidate to the registry.

        @register("attn_block", "sdpa_fused_qkv")
        def sdpa_fused_qkv(x, w, mask, causal): ...

    Kernel authors add one function plus one line; nothing else changes.
    """
    if slot not in SLOTS:
        raise ValueError(f"unknown slot {slot!r}, expected one of {SLOTS}")

    def decorate(fn: Callable) -> Callable:
        if name in _REGISTRY[slot]:
            raise ValueError(f"duplicate candidate {slot}/{name}")
        _REGISTRY[slot][name] = fn
        fn._slot = slot          # type: ignore[attr-defined]
        fn._name = name          # type: ignore[attr-defined]
        return fn

    return decorate


def candidates(slot: str) -> Dict[str, Callable]:
    """All registered candidates for a slot. Used by the autotuner."""
    return dict(_REGISTRY[slot])


# --------------------------------------------------------------------------
# Key
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Key:
    batch: int
    seq_len: int
    d_model: int
    num_heads: int
    ffn_dim: int
    num_layers: int
    causal: bool
    dtype: str          # "float32" / "float16" / "bfloat16"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def tokens(self) -> int:
        """batch * seq_len - the best single predictor of which regime we are in."""
        return self.batch * self.seq_len

    @classmethod
    def from_input(cls, x: torch.Tensor, config) -> "Key":
        # x.shape and x.dtype live on the CPU already; reading them is free.
        batch, seq_len, _ = x.shape
        return cls(
            batch=int(batch),
            seq_len=int(seq_len),
            d_model=int(config.d_model),
            num_heads=int(config.num_heads),
            ffn_dim=int(config.ffn_dim),
            num_layers=int(config.num_layers),
            causal=bool(config.causal),
            dtype=str(x.dtype).replace("torch.", ""),
        )

    def as_dict(self) -> dict:
        return {
            "batch": self.batch, "seq_len": self.seq_len,
            "d_model": self.d_model, "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim, "num_layers": self.num_layers,
            "causal": self.causal, "dtype": self.dtype,
        }

    def __str__(self) -> str:
        return (f"b{self.batch}_n{self.seq_len}_d{self.d_model}"
                f"_h{self.num_heads}_f{self.ffn_dim}_l{self.num_layers}"
                f"_{'causal' if self.causal else 'full'}_{self.dtype}")


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Plan:
    """The chosen implementations. `None` means 'use the baseline path'."""
    attn_block: Optional[Callable] = None
    ffn_block: Optional[Callable] = None
    full_block: Optional[Callable] = None
    source: str = "fallback"

    def names(self) -> Dict[str, str]:
        out = {}
        for slot in SLOTS:
            fn = getattr(self, slot)
            out[slot] = getattr(fn, "_name", None) if fn else None
        return out


FALLBACK_PLAN = Plan(source="fallback")


# --------------------------------------------------------------------------
# Routing table
# --------------------------------------------------------------------------
# One JSON per GPU, named after the device. Same code, different tables - this
# is what makes "our dispatcher retunes itself per hardware" a true statement
# rather than a claim.

CONFIG_DIR = Path(__file__).resolve().parent / "configs"

_TABLE: Optional[Dict[Key, Dict[str, str]]] = None
_TABLE_SOURCE: str = "none"


def device_slug(device: Optional[torch.device] = None) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name(device)
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_table(path: Optional[Path] = None) -> int:
    """Load a routing table. Returns how many entries were read.

    Missing file is not an error - it just means nothing has been autotuned for
    this GPU yet, and every shape takes the fallback path.
    """
    global _TABLE, _TABLE_SOURCE

    if path is None:
        path = CONFIG_DIR / f"{device_slug()}.json"

    entries: Dict[Key, Dict[str, str]] = {}
    if path.exists():
        data = json.loads(path.read_text())
        for entry in data.get("entries", []):
            shape = entry["shape"]
            entries[Key(**shape)] = entry["plan"]
        _TABLE_SOURCE = str(path)
    else:
        _TABLE_SOURCE = f"{path} (not found)"

    _TABLE = entries
    _plan_cache.clear()
    return len(entries)


def save_table(table: Dict[Key, dict], path: Optional[Path] = None) -> Path:
    """Written by bench/autotune.py. Carries the evidence, not just the winner."""
    if path is None:
        path = CONFIG_DIR / f"{device_slug()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "entries": [
            {"shape": key.as_dict(), **record} for key, record in table.items()
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------

_plan_cache: Dict[Key, Plan] = {}


def resolve(x: torch.Tensor, config) -> Plan:
    """Called once per forward. Cache hit is a single dict lookup."""
    key = Key.from_input(x, config)
    plan = _plan_cache.get(key)
    if plan is None:
        plan = _build_plan(key)
        _plan_cache[key] = plan
    return plan


def _build_plan(key: Key) -> Plan:
    if _TABLE is None:
        load_table()

    spec = _TABLE.get(key) if _TABLE else None
    if spec is None:
        return _default_plan(key)

    chosen: Dict[str, Callable] = {}
    for slot, name in spec.items():
        if slot not in SLOTS:
            continue
        fn = _REGISTRY[slot].get(name)
        if fn is None:
            # Table names a candidate that no longer exists (renamed, deleted,
            # or its module was not imported). Degrade rather than crash.
            return replace(_default_plan(key), source=f"missing:{slot}/{name}")
        chosen[slot] = fn

    return Plan(**chosen, source="table")


def _default_plan(key: Key) -> Plan:
    """Safe path for any shape the autotuner has not measured.

    Prefers candidates registered as "general" - correct everywhere, tuned for
    nothing. If none are registered, every slot is None and the encoder runs the
    unmodified baseline.
    """
    return Plan(
        attn_block=_REGISTRY["attn_block"].get("general"),
        ffn_block=_REGISTRY["ffn_block"].get("general"),
        source="fallback",
    )


# --------------------------------------------------------------------------
# Introspection - for the demo video and the tech report
# --------------------------------------------------------------------------

def explain(x: torch.Tensor, config) -> str:
    key = Key.from_input(x, config)
    plan = resolve(x, config)
    names = plan.names()
    return (
        f"shape   : {key}\n"
        f"tokens  : {key.tokens:,}   head_dim: {key.head_dim}\n"
        f"table   : {_TABLE_SOURCE}\n"
        f"source  : {plan.source}\n"
        f"attn    : {names['attn_block']}\n"
        f"ffn     : {names['ffn_block']}\n"
        f"full    : {names['full_block']}"
    )


def stats() -> dict:
    return {
        "table_source": _TABLE_SOURCE,
        "table_entries": len(_TABLE) if _TABLE else 0,
        "cached_plans": len(_plan_cache),
        "registered": {slot: sorted(_REGISTRY[slot]) for slot in SLOTS},
    }


# --------------------------------------------------------------------------
# Official test shapes (problem statement, appendix 3.7)
# --------------------------------------------------------------------------
# dtype is deliberately absent: the appendix does not pin it, so the autotuner
# crosses these with whichever dtypes we decide to cover.

OFFICIAL_SHAPES: List[dict] = [
    dict(batch=64,    d_model=128,  num_heads=4,  seq_len=128,    num_layers=4, causal=True, ffn_dim=128),
    dict(batch=1,     d_model=128,  num_heads=4,  seq_len=128,    num_layers=4, causal=True, ffn_dim=128),
    dict(batch=4,     d_model=128,  num_heads=4,  seq_len=128,    num_layers=4, causal=True, ffn_dim=128),
    dict(batch=16,    d_model=128,  num_heads=4,  seq_len=128,    num_layers=4, causal=True, ffn_dim=128),
    dict(batch=128,   d_model=128,  num_heads=4,  seq_len=128,    num_layers=4, causal=True, ffn_dim=128),
    dict(batch=10000, d_model=128,  num_heads=4,  seq_len=128,    num_layers=4, causal=True, ffn_dim=128),
    dict(batch=64,    d_model=32,   num_heads=4,  seq_len=128,    num_layers=4, causal=True, ffn_dim=32),
    dict(batch=64,    d_model=1024, num_heads=4,  seq_len=128,    num_layers=4, causal=True, ffn_dim=1024),
    dict(batch=64,    d_model=128,  num_heads=1,  seq_len=128,    num_layers=4, causal=True, ffn_dim=128),
    dict(batch=64,    d_model=128,  num_heads=2,  seq_len=128,    num_layers=4, causal=True, ffn_dim=128),
    dict(batch=64,    d_model=128,  num_heads=16, seq_len=128,    num_layers=4, causal=True, ffn_dim=128),
    dict(batch=64,    d_model=128,  num_heads=4,  seq_len=32,     num_layers=4, causal=True, ffn_dim=128),
    dict(batch=64,    d_model=128,  num_heads=4,  seq_len=1024,   num_layers=4, causal=True, ffn_dim=128),
    dict(batch=32,    d_model=1024, num_heads=16, seq_len=100000, num_layers=2, causal=True, ffn_dim=1024),
]


def official_keys(dtype: str = "float32") -> List[Key]:
    return [Key(dtype=dtype, **shape) for shape in OFFICIAL_SHAPES]
