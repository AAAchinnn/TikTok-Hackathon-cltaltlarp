# Shape-Adaptive Transformer Kernels for the TikTok GPU Hackathon

An optimized GPU implementation of the benchmark Transformer layer that picks a
different implementation and a different numerical precision **per input shape**,
using measurements taken on the machine it will actually run on.

Against the provided PyTorch baseline on a Tesla T4, all official shapes pass the
accuracy gate (`abs <= 2e-3` OR `rel <= 2e-2`).

## Project overview

The problem statement invites per-shape specialisation: *"participants can choose
different implementations for different shapes by adding shape checks."* Rather
than write a chain of `if seq_len > N` branches, we built the general form of that
idea and let measurement fill it in.

Three things happen on every forward:

1. **Routing.** `opt/dispatcher.py` maps `(shape, dtype, config)` to an
   implementation and a precision plan. The key is built only from values the CPU
   already knows — never from tensor contents, because reading a value off the GPU
   forces a synchronisation that costs more than any routing decision can win back.
   Resolution is cached, so steady-state cost is one dict lookup.
2. **Precision selection.** A T4 (sm75) has no TF32 path, so an fp32 GEMM gets no
   Tensor Core help at all. Narrowing selected GEMMs to fp16 is the single largest
   lever available, but which GEMMs can be narrowed depends on the shape. That
   decision is measured, never asserted — see *Measured, not asserted* below.
3. **Execution.** Three routes, chosen by batch size, because the bottleneck moves:
   large batches are GEMM-bound, tiny batches are kernel-launch-bound, and the
   middle is neither.

| route | when | what dominates | what we do |
|---|---|---|---|
| compute-heavy | `B >= 128` | GEMM throughput | fp16 Tensor Cores |
| general | `4 < B < 128` | mixed | `torch.compile` + calibrated precision |
| low-overhead | `B <= 4` | kernel launch overhead | CUDA graph replay |

Underneath all three, the arithmetic itself uses
`F.scaled_dot_product_attention`, which never materializes the `[B, H, N, N]`
score matrix that the baseline writes and re-reads roughly eleven times.

## Measured, not asserted

The design rule the project is built around: **nothing ships on a claim that could
have been measured.**

- `bench/autotune.py` runs every registered candidate against every precision
  preset, on every official shape, and keeps the fastest that clears the accuracy
  gate with headroom to spare. The gate is the harness's own `compare_outputs`, at
  the real tolerance, over several distinct inputs — not an approximation of it.
  The result is a routing table in `opt/configs/<gpu>.json` that carries its own
  evidence: every combination considered, its latency, and its worst observed error.
- `opt/masking.py` *verifies* that padding is a suffix rather than assuming it.
  Under causal attention with suffix padding, a valid query at position `i` attends
  only keys `j <= i`, all of which are valid — so the key-padding mask is redundant
  and can be dropped entirely. That is a statement about the data, not the shape,
  and if it were ever false the output would be silently wrong. Verifying costs one
  extra reduction on a synchronisation already being paid.
- `mode="max-autotune"` is deliberately **not** used: it measured *slower* on a T4
  (1.99x against 2.01x) and warns "Not enough SMs to use max_autotune_gemm".
- bfloat16 is treated as unavailable below sm80. PyTorch will accept it on a T4 and
  emulate it — measured at 0.94x, slower than doing nothing, while also failing
  accuracy at `max_abs=7.2e-3`. Refusing it is what stops the ladder selecting that trap.

## What each optimization is worth

Ablations at the hub shape (`b64 n128 d128 h4 l4`, fp32, non-causal), each measured
by disabling one thing:

| configuration | speedup |
|---|---|
| all optimizations | **2.013x** |
| without `torch.compile` | 1.661x |
| without packed QKV projection | 1.851x |
| without all-valid mask elision | 1.932x |
| with `max-autotune` instead of `default` | 1.991x |

`torch.compile` is the largest single contributor — Inductor fuses
LayerNorm+residual, bias+GELU and the trailing `masked_fill`. Note the Dynamo cache
size limit is raised at construction: the benchmark has 14 official shapes against
a default of 8, and past the limit Dynamo silently falls back to eager, turning a
2.01x quietly into a 1.66x.

Precision presets at `d_model=1024`, where attention is only ~4% of the FLOPs and
the rest is projection GEMM:

| preset | narrowed GEMMs | max_abs | speedup |
|---|---|---|---|
| `off` | none | 3.1e-06 | 1.097x |
| `attn` | qkv, attn | 9.75e-04 | 1.926x |
| `safe` | qkv, attn, ffn_in | 1.14e-03 | 2.505x |
| `all` | + out, ffn_out | 1.46e-03 | 5.115x |

`safe` exists because `out_proj` and `ffn_out` write straight into the residual
stream, so their rounding lands undiluted in the final comparison. Sparing those
two costs about half the available speedup and roughly halves the error.

## Setup and installation

Requires an NVIDIA GPU. Developed and measured on a Tesla T4 via Google Colab.

```bash
git clone https://github.com/AAAchinnn/TikTok-Hackathon-cltaltlarp
cd TikTok-Hackathon-cltaltlarp
pip install torch          # CUDA build; preinstalled on Colab
```

No other dependencies.

## Steps to reproduce

**1. Confirm the optimized path is live.**

```bash
python torch_transformer_benchmark.py \
  --batch-size 64 --seq-len 128 --d-model 128 --heads 4 \
  --ffn-dim 128 --layers 4 --causal --device cuda --dtype float32 \
  --rtol 0.02 --atol 0.002
```

With `OPT_VERBOSE=1` this prints the precision plan and the route taken.

**2. Build the routing table for your GPU** (once per GPU, ~30-45 min):

```bash
python bench/autotune.py --atol 0.002 --rtol 0.02
```

Writes `opt/configs/<gpu>.json`. Add `--dry-run` to see the full candidate x preset
ladder without writing anything. Without this step the code still runs correctly
everywhere — it just takes the untuned fallback path on every shape.

**3. Run the full shape sweep:**

```bash
python tools/sweep.py --out results --atol 0.002 --rtol 0.02
```

Writes one log per run plus `results/summary.md`, a Markdown table of speedup,
pass/fail, worst error and the route taken for each shape.

The tolerances are passed explicitly rather than edited into the harness, so
`torch_transformer_benchmark.py` differs from the file as shipped by exactly the
two lines that mix in our implementation.

## Repository layout

```
torch_transformer_benchmark.py   organisers' harness, +2 lines to mix us in
opt/
  __init__.py       package surface; importing it registers the candidates
  dispatcher.py     shape -> (implementation, precision) routing
  general.py        the encoder: mask analysis, precision policy, layer stack
  blocks.py         candidate implementations, one per dispatcher slot
  precision.py      which GEMMs get narrowed, decided by measurement
  masking.py        all-valid elision and suffix-padding verification
  configs/          per-GPU routing tables written by the autotuner
bench/autotune.py   measures candidates x presets, writes the routing table
tools/sweep.py      runs the official shapes, writes results/summary.md
```

## Limitations and what we would improve

**Row 14 of the official shapes cannot be run by anyone.** At
`b32 x n100000 x d1024 x h16`, the score matrix the *baseline* builds is
`32 x 16 x 100000 x 100000` — 5.12 x 10^12 elements, **20.5 TB** in fp32. No machine
can produce a reference output, so the harness has nothing to compare against. Our
own path never materializes that matrix and would need roughly 26 GB in fp16 with
batch chunking, which is an A100-class requirement. With more time we would write a
block-wise reference — mathematically identical, tiled over the query dimension —
validate it against the shipped baseline at small shapes, and use it to verify row
14 where the shipped baseline cannot run.

**A hand-written Triton attention kernel did not make the cut.** We built one and
measured it; on sm75, `F.scaled_dot_product_attention` dispatches to a
memory-efficient kernel that it did not beat on our shapes, so it is not on the
shipped path. The work is preserved on the `triton` branch rather than carried as
dead code here. If it were revisited, the dispatcher makes adopting it a one-line
change -- register it as a candidate and `bench/autotune.py` measures it against the
general block on every shape automatically, behind the same correctness gate.

**The candidate slot has only one occupant.** The routing mechanism cross-products
candidates against precision presets, but today only the `general` candidate is
registered per slot, so the routing table's real work is precision selection. The
architecture is built for a second candidate; we have not yet earned one.

**CUDA graph replay is skipped when an attention mask tensor survives to the
kernel.** A capture would bake the mask into the recording and silently replay it
for later inputs, so those cases run the same routed path eagerly instead. No
official shape is affected — they are all causal, and causal plus verified suffix
padding elides the mask entirely — but a non-causal padded workload at `B <= 4`
gives up the launch-overhead win rather than risk a stale mask. Including mask
identity in the graph key would recover it.

**Calibration measures one input.** `opt/precision.Calibrator` is the fallback when
no routing table exists, and it judges a preset from a single warmup forward with a
safety margin. We measured that single-input estimate under-reporting the worst case
by 1.39x. The autotuner exists precisely because that is not good enough evidence to
ship on; where a table exists, calibration is bypassed.

## Team member contributions

<!-- TODO: fill in before submission -->
- **[Name]** — ...
- **[Name]** — ...

## Development tools and AI assistance

<!-- TODO: expand for the Devpost description; see TECH_REPORT.md -->

## License

See `LICENSE`.
