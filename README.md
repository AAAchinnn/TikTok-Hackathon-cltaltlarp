# Shape-specialized Triton kernels for the supplied Transformer benchmark

This package is wired to the exact structure of `torch_transformer_benchmark.py`.
The benchmark is a pre-LN Transformer with:

- separate learned Q, K and V projections,
- multi-head attention,
- an optional causal mask,
- an optional valid-token/padding mask,
- an output projection,
- residual connections,
- LayerNorm,
- two FFN Linear layers with exact GELU,
- and a final LayerNorm.

The implementation deliberately keeps all of those existing parameter modules
unchanged. Only the attention computation is replaced with Triton, so the
benchmark's existing `copy_model_weights()` continues to work with a strict
state-dict copy.

## Repository layout

```text
transformer_triton_kernels/
  dispatcher.py
  test_correctness.py
  kernels/
    __init__.py
    _attention_triton.py
    small_attention.py
    medium_attention.py
    large_attention.py
    fallback.py
```

The supplied benchmark script is also updated at:

```text
../torch_transformer_benchmark.py
```

## Integration point

Inside `UserOptimizedTransformer.forward()`, the benchmark now does:

```python
context = triton_attention(
    q,
    k,
    v,
    valid_token_mask=valid_token_mask,
    causal=self.config.causal,
)
```

where Q/K/V have shape `[B, H, S, D]`.

## Dispatcher policy

The current starting thresholds are:

```text
S <= 64    -> small Triton path
S <= 512   -> medium Triton path
S > 512    -> large Triton path
unsupported CUDA/dtype/head_dim -> PyTorch SDPA fallback
```

These are deliberately tunable starting points. The benchmark document says
shape specialization should be based on measured latency for the known test
shapes, not on a universal heuristic.

## Why there is an online-softmax kernel

The Triton kernel processes K/V in tiles and updates the softmax statistics
incrementally. It therefore avoids writing an `S x S` attention matrix to
GPU global memory. That is the important large-sequence optimization suggested
by the supplied architecture document.

## Correctness gate

The benchmark itself remains authoritative. Run its normal accuracy phase
before trusting any performance number. The included test uses the benchmark's
`compare_outputs()` helper and exercises:

- small / medium / large dispatcher paths,
- causal attention,
- padding masks,
- FP16,
- BF16.

## Running

From the project directory on a CUDA + Triton environment:

```bash
python test_correctness.py
python ../torch_transformer_benchmark.py --device cuda --dtype float16
```

The benchmark also supports its existing warm-up, repeat, compilation and
accuracy arguments.

## Tuning

The most important parameters to sweep on the actual target GPU are:

- `SMALL_MAX_N` and `MEDIUM_MAX_N` in `dispatcher.py`,
- `BLOCK_M`, `BLOCK_N`, `num_warps`, and `num_stages` in the three family files.

Do not promote a configuration to the final dispatcher just because it is
fast: it should first pass the benchmark's numerical correctness gate.
