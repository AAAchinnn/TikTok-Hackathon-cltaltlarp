"""Optimized Transformer implementations for the TikTok GPU kernel task.

The whole package exists to be reachable from two lines in the benchmark
harness, which is otherwise left exactly as the organisers shipped it:

    from opt import OptimizedMixin

    class UserOptimizedTransformer(OptimizedMixin, BaselineTransformer):
        pass

`OptimizedMixin` goes first in the bases so its `forward` wins the MRO, while
the class stays an IS-A `BaselineTransformer` -- same submodules, same
parameter names, so `copy_model_weights(strict=True)` is unaffected.

Layout:

    general.py     the encoder: mask analysis, precision, the layer stack
    blocks.py      candidate implementations, one per dispatcher slot
    precision.py   which GEMMs get narrowed, decided by measurement
    masking.py     all-valid elision and suffix-padding verification
    dispatcher.py  shape -> candidate routing
    triton_attn.py hand-written Triton attention, parked

Every forward goes through the dispatcher. With no routing table present for
the current GPU it returns the "general" candidates, which are the permanent
fallback -- correct on any shape, tuned for none. Registering a specialised
candidate and measuring it into a table is what changes that, and nothing in
the encoder has to change for it.

Importing this package registers the general candidates as a side effect, via
`blocks`. That import is load-bearing: without it the registry is empty and
`dispatcher._default_plan` has nothing to return.

Run `python tools/diagnose.py` for per-stage error tables and
`python tools/sweep.py` for the shape x preset matrix.
"""

from . import blocks  # noqa: F401  -- registers the "general" candidates
from . import dispatcher
from .blocks import BlockContext
from .dispatcher import Key, Plan, candidates, explain, register, resolve, stats
from .general import OptimizedMixin, attention
from .precision import TARGET_ATOL, TARGET_RTOL, Settings, settings

__all__ = [
    "OptimizedMixin",
    "attention",
    "settings",
    "Settings",
    "TARGET_ATOL",
    "TARGET_RTOL",
    # dispatcher surface, for candidate authors and the demo
    "dispatcher",
    "register",
    "resolve",
    "explain",
    "stats",
    "candidates",
    "Key",
    "Plan",
    "BlockContext",
]

__version__ = "0.3.0"
