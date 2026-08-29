# Triton Transformer Optimization Project

## Goal

Optimize the supplied Transformer benchmark for the official benchmark
configuration set while preserving the benchmark's correctness requirements.


The goal is:

1. Pass the benchmark correctness gate.
2. Improve latency on the official benchmark configurations.
3. Only introduce custom kernels where they provide a measurable benefit.
4. Preserve the benchmark's externally visible model interface and parameter names.

## Benchmark correctness

The benchmark accepts each output element only when:

abs(user - reference) <= atol
OR
abs(user - reference) <= rtol * abs(reference)

Default:
- atol = 0.001
- rtol = 0.01

Never loosen the benchmark tolerance.

Never hide correctness failures.

## Official configurations

Optimize specifically for these configurations:

1. B=64,   D=128,  H=4,  N=128, layers=4, causal=True, FFN=128
2. B=1,    D=128,  H=4,  N=128, layers=4, causal=True, FFN=128
3. B=4,    D=128,  H=4,  N=128, layers=4, causal=True, FFN=128
4. B=16,   D=128,  H=4,  N=128, layers=4, causal=True, FFN=128
5. B=128,  D=128,  H=4,  N=128, layers=4, causal=True, FFN=128
6. B=10000,D=128,  H=4,  N=128, layers=4, causal=True, FFN=128
7. B=64,   D=32,   H=4,  N=128, layers=4, causal=True, FFN=32
8. B=64,   D=1024, H=4,  N=128, layers=4, causal=True, FFN=1024
9. B=64,   D=128,  H=1,  N=128, layers=4, causal=True, FFN=128
10. B=64,  D=128,  H=2,  N=128, layers=4, causal=True, FFN=128
11. B=64,  D=128,  H=16, N=128, layers=4, causal=True, FFN=128
12. B=64,  D=128,  H=4,  N=32,  layers=4, causal=True, FFN=128
13. B=64,  D=128,  H=4,  N=1024,layers=4, causal=True, FFN=128
14. B=32,  D=1024, H=16, N=100000,layers=2, causal=True, FFN=1024

## Architecture rules

Keep these parameter names compatible with the benchmark:

- q_proj
- k_proj
- v_proj
- out_proj
- norm1
- norm2
- ffn_in
- ffn_out
- layers
- final_norm

Do not rewrite the model architecture unless there is a measured reason.


## Optimization methodology

Every optimization must follow:

1. State the hypothesis.
2. Make the smallest possible change.
3. Run correctness.
4. Run performance.
5. Record the result.
6. Keep or revert based on evidence.

Do not make multiple unrelated changes in one experiment.

## Performance

Report:
- median latency
- mean latency
- p90 latency
- minimum latency
- speedup versus baseline

Use CUDA event timing for GPU measurements.

## Numerical debugging

When correctness fails:
- isolate the first operation where outputs diverge
- compare intermediate tensors
- distinguish absolute error from relative error
- inspect reduction order and dtype boundaries
- do not immediately increase precision everywhere

## Long sequence

For N=100000:
- never materialize an N x N attention matrix
- use streaming/tiled/online-softmax attention

## Code quality

Use descriptive names.
Prefer simple kernels over clever kernels until correctness is established.
