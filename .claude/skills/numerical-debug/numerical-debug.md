---
name: numerical-debug
description: Isolate numerical divergence between reference and optimized kernels.
---

# Numerical Debugging Workflow

Compare:

- inputs
- Q
- K
- V
- QK^T
- masked scores
- softmax
- P@V
- attention output
- output projection
- residual
- LayerNorm
- FFN
- GELU
- final output

Find the earliest divergence.

Do not change more than one precision boundary at a time.

Do not loosen the benchmark tolerance.
