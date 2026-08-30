# Technical Report

Implement a GPU Kernel for a Transformer Layer — TikTok Hackathon

---

## 1. Environment

| | |
|---|---|
| GPU | NVIDIA Tesla T4, 16 GB, compute capability **7.5 (Turing / sm75)** |
| Platform | Google Colab |
| CPU | <!-- TODO --> |
| RAM | <!-- TODO --> |
| Disk | <!-- TODO --> |
| PyTorch | <!-- TODO: from the benchmark's own output line --> |
| CUDA | <!-- TODO --> |
| Python | <!-- TODO --> |

Collect the missing rows with:

```python
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
!lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core"
!free -h | head -2
!df -h / | tail -1
import torch, sys
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| python", sys.version.split()[0])
```

**Why the GPU generation is the central fact of this report.** Turing (sm75) has
fp16 Tensor Cores but **no TF32 path**. On Ampere and later, an fp32 matmul is
silently accelerated by TF32 Tensor Cores; on a T4 it is not. An fp32 GEMM on this
card runs at roughly 8 TFLOPS against 65 TFLOPS for fp16. Every significant decision
below follows from that gap, and most of them would be different on an A100.

---

## 2. Methodology

**Correctness gate.** The problem statement specifies `relative error < 0.02` and
`absolute error < 0.002`, element-wise, as an OR. The harness's own
`compare_outputs` implements exactly that; we call it directly rather than
reimplementing it, so our gate and the scoring gate cannot drift apart.

Note the harness as shipped disagrees with itself — its module docstring says
`atol=0.001` while its argparse defaults to `0.001`/`0.01`, and the problem
statement says `0.002`/`0.02`. We resolved this **outside** the organisers' file:
`torch_transformer_benchmark.py` differs from the shipped version by exactly the two
lines that mix in our implementation, and every command passes `--atol 0.002
--rtol 0.02` explicitly. The choice is therefore visible in the command line rather
than hidden in a modified default.

**Timing.** `torch.cuda.Event` on the current stream, median of 300 samples across
three alternating rounds. Wall-clock timing measures the host, not the GPU, because
the launch queue runs ahead.

**Selection.** `bench/autotune.py` measures every registered candidate against every
precision preset on every official shape, and keeps the fastest that clears the gate
on five distinct inputs with 20% headroom in hand. Nothing enters the routing table
that has not beaten the baseline on speed *and* passed accuracy — a combination that
is faster but wrong is not a winner.

---

## 3. Optimizations

### 3.1 Scaled dot-product attention

The baseline computes `matmul -> mask -> softmax -> matmul`, materializing the
`[B, H, N, N]` score matrix and passing over it roughly eleven times. At row 13
(`b64 n1024 h4`) that matrix is 268M elements — **1.07 GB**, allocated fresh by
`masked_fill` and again by `softmax`.

`F.scaled_dot_product_attention` dispatches to a tiled online-softmax kernel that
never materializes it. This is what carries every long-sequence shape and is the
only reason the largest shapes run at all.

### 3.2 Packed QKV projection

Three `[d, d]` GEMMs become one `[3d, d]` GEMM. The packed weight is a cache built
from the three `nn.Linear` modules, keyed on parameter `_version`, so parameter
names are untouched and `copy_model_weights(strict=True)` keeps working. Worth
1.851x -> 2.013x at the hub shape.

### 3.3 `torch.compile` over the layer stack

Inductor fuses LayerNorm+residual, bias+GELU, and the trailing `masked_fill`. The
largest single contributor: 1.661x -> 2.013x.

Dynamo's cache size limit is raised from 8 to 64 at construction. The benchmark has
14 official shapes; past the limit Dynamo silently falls back to eager with only a
warning, which would turn a 2.01x quietly into a 1.66x.

`mode="max-autotune"` measured *slower* (1.991x) and warns "Not enough SMs to use
max_autotune_gemm". We use `default`.

### 3.4 Reduced precision, selected by measurement

Five GEMMs per block can be independently narrowed to fp16: `qkv`, `attn`, `out`,
`ffn_in`, `ffn_out`. Presets form a ladder, tried fastest-first:

| preset | narrowed | max_abs @ d1024 | speedup @ d1024 |
|---|---|---|---|
| `off` | — | 3.1e-06 | 1.097x |
| `qkv` | qkv | — | — |
| `attn` | qkv, attn | 9.75e-04 | 1.926x |
| `safe` | qkv, attn, ffn_in | 1.14e-03 | 2.505x |
| `all` | all five | 1.46e-03 | 5.115x |

`out_proj` and `ffn_out` write straight into the residual stream, so their rounding
lands undiluted in the output. That is why `safe` exists as a rung.

fp16 GEMMs are forced to accumulate in fp32 (`allow_fp16_reduced_precision_reduction
= False`) rather than inheriting a default that has moved between torch versions.

bfloat16 is treated as unavailable below sm80. A T4 will accept it and emulate it:
measured 0.941x — slower than doing nothing — while failing accuracy at
`max_abs = 7.2e-3`.

### 3.5 Mask elision, and verifying rather than assuming

An all-valid mask is detected once and dropped, removing one `masked_fill` per
layer: 1.932x -> 2.013x.

Under **causal** attention with **suffix** padding, a valid query at position `i`
attends only keys `j <= i`, and since padding sits at the end, every such key is
valid. The key-padding mask is therefore redundant and `is_causal=True` alone
reproduces the baseline exactly — about 7% faster than building a combined bias.

That is a property of the *data*, not the shape, and if it were false the output
would be silently wrong. `opt/masking.py` verifies it (a violation is a `False`
immediately followed by a `True`) on a synchronisation already being paid. This is
directly observable in our results: padded and unpadded runs report *identical*
`max_abs` to six significant figures, because both take the same kernel path.

### 3.6 Three execution routes

The bottleneck moves with batch size, so one implementation cannot be right everywhere.

| route | condition | bottleneck | technique |
|---|---|---|---|
| compute-heavy | `B >= 128` | GEMM throughput | fp16 Tensor Cores |
| general | `4 < B < 128` | mixed | compile + calibrated precision |
| low-overhead | `B <= 4` | ~40 kernel launches at 5-10 µs | CUDA graph replay |

At `B <= 4` launch overhead is 200-400 µs of a ~2 ms pass. A CUDA graph collapses
the whole forward to one host command. Capture is refused when an attention mask
tensor survives to the kernel — a recording would bake it in and silently replay
it for later inputs — so those cases run the same path eagerly. No official shape
is affected: all are causal, and causal + verified suffix padding elides the mask.

### 3.7 Shape-aware dispatch

`opt/dispatcher.py` routes on a key built **only** from CPU-known values — shapes,
dtypes, config flags — never tensor contents, because reading a value off the GPU
forces a synchronisation costing more than the routing decision can win. Unknown
shapes fall back to a safe plan rather than raising. The routing table is data, not
code: `bench/autotune.py` regenerates it per GPU, and it carries its own evidence —
every combination considered, with latency and worst observed error.

---

## 4. Results

Tesla T4, fp32 stream, all 14 shapes causal, gate `abs <= 2e-3 OR rel <= 2e-2`.

### 4.1 Fallback path (no routing table)

| shape | speedup | max_abs | padded speedup |
|---|---|---|---|
| row01 `b64 n128 d128 h4` | 2.535x | 1.43e-06 | 2.451x |
| row02 `b1` | 5.461x | 9.70e-04 | 6.638x |
| row03 `b4` | 4.349x | 8.06e-04 | 4.411x |
| row04 `b16` | 1.954x | 8.27e-04 | 2.209x |
| row05 `b128` | 6.995x | 1.73e-03 | 6.977x |
| row07 `d32` | 4.986x | 1.19e-06 | 4.837x |
| row08 `d1024` | 1.170x | 3.10e-06 | 1.120x |
| row09 `h1` | 1.704x | 1.43e-06 | 1.707x |
| row10 `h2` | 2.100x | 1.43e-06 | 2.101x |
| row11 `h16` | 3.726x | 1.43e-06 | 3.694x |
| row12 `n32` | 2.472x | 1.43e-06 | 2.459x |
| row13 `n1024` | 5.008x | 1.91e-06 | 5.083x |

**24/24 passing. Geometric mean 3.10x** across the twelve unpadded shapes, range
1.17x – 7.00x.

### 4.2 With the autotuned routing table

<!-- TODO: regenerate with tools/sweep.py after bench/autotune.py and paste
     results/summary.md here. Confirmed so far: row 13 moves 5.008x -> 17x. -->

The `max_abs` column in 4.1 shows the selection working. Shapes reading ~1e-06 ran
fp32 — the run-time calibrator rejected every fp16 preset. Shapes reading ~8e-04
accepted one. The calibrator targets its own default `atol` of 1e-3 with a 0.7
margin, giving a 7e-4 budget, so presets whose true error is ~1.2e-3 were rejected
against a gate that actually permits 2e-3. Row 8 sat at 1.170x for exactly this
reason while `fp16/all` measures 1.16e-3 and runs ~4.9x. The autotuner removes that
gap by measuring against the real gate instead of a proxy.

### 4.3 Shapes not measured

**Row 6** (`b10000`): 1.28M tokens. The baseline's score matrix is 2.62 GB and
`masked_fill`/`softmax` each allocate a fresh copy, putting baseline peak near 9 GB
on a 16 GB card. Run in isolation with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Row 14** (`b32 n100000 d1024 h16`): **not runnable by anyone.** The baseline's
score matrix is 5.12 x 10^12 elements — **20.5 TB** in fp32. No hardware can produce
a reference output, so the harness has nothing to compare against. Our path never
materializes it and would need ~26 GB in fp16 with batch chunking (A100-class). See
README *Limitations* for the block-wise reference we would build given more time.

---

## 5. Interpretation

Two honest observations we would rather state than have asked.

**The large speedups are partly a statement about the reference.** The baseline is a
deliberately naive implementation that allocates gigabytes of score matrix per
layer. At long sequence lengths, removing that allocation dominates everything else
we did. Row 13's number is mostly SDPA versus `O(N²)` materialization, amplified by
causal masking — which the baseline pays for (it computes all N² scores, then
discards half) and we do not (SDPA skips the blocks entirely).

**The most interesting result is not the largest one.** Row 8 (`d_model=1024`) is
GEMM-bound; SDPA buys almost nothing there, and the entire win comes from precision
routing on a card with no TF32. That row moving from 1.17x to ~4.9x is a better
demonstration of the dispatcher's value than row 13's headline, because it is the
shape where nothing else helps.

---

## 6. AI tools and skills used

<!-- TODO: complete before submission — explicitly worth bonus points. -->

| tool | how it was used |
|---|---|
| Claude Code (Claude Opus) | Codebase analysis against the problem statement; identified that the optimized package was never wired into the harness; diagnosed the calibrator targeting a stricter tolerance than the task requires; authored `bench/autotune.py` and `tools/sweep.py`; extended the dispatcher to route precision plans |
| <!-- TODO --> | |

**Prompting approach that worked.** <!-- TODO: describe. The most productive
pattern was asking the model to audit measured results against the problem
statement's stated tolerance, rather than asking it to write kernels — that is what
surfaced the 1e-3 vs 2e-3 gap worth ~4x on row 8. -->

---

## 7. Reproducing

See README *Steps to reproduce*. Summary:

```bash
python bench/autotune.py --atol 0.002 --rtol 0.02      # once per GPU
python tools/sweep.py --out results --atol 0.002 --rtol 0.02
```

Raw logs for every run in `results/`; `results/summary.md` is generated.
