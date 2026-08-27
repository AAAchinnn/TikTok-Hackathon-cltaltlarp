# Integration quick reference

The modified benchmark already imports the dispatcher. The only custom call
needed by the Transformer is:

```python
from dispatcher import attention

context = attention(
    q,
    k,
    v,
    valid_token_mask=valid_token_mask,
    causal=config.causal,
)
```

Input/output contract:

```text
q:    [B, H, S, D]
k:    [B, H, S, D]
v:    [B, H, S, D]
mask: [B, S] or None
out:  [B, H, S, D]
```

The higher-level benchmark then transposes the output back to `[B, S, d_model]`
and applies the unchanged output projection.
