# Experimentation Log — Transformer CUDA Kernel Optimization

Project: pure-CUDA fused attention kernel for `torch_transformer_benchmark.py`,
optimized against the 14 official benchmark configurations in
`.claude/CLAUDE.md`. This file is append-only; entries are never rewritten
after the fact (see `MUTATION_MENU.md`).

---

## Current Architecture

```
BaselineTransformer forward (torch_transformer_benchmark.py, UNCHANGED)
    |
    v
per-layer: norm1 -> attention -> +residual -> norm2 -> ffn -> +residual
    |
    v
attention (kernel-opt/harness/common.py::_CudaAttention)
    q_proj / k_proj / v_proj / out_proj  -> plain torch nn.Linear (cuBLAS)
    core attention (QK^T, scale, causal+padding mask, softmax, PV)
        -> kernels/current.cu :: fused_attention   [CUSTOM CUDA, v0]
    |
    v
ffn_in -> GELU(exact) -> ffn_out                    -> plain torch (cuBLAS)
```

Only the attention core is currently custom CUDA. QKV/output/FFN projections
are untouched `nn.Linear` (cuBLAS GEMMs), per the "prefer cuBLAS unless
proven otherwise" rule — no evidence has been gathered yet on whether they're
worth touching.

`kernels/current.cu` v0: a single fused kernel doing tiled online-softmax
attention (FlashAttention-style forward). Known-suboptimal by construction on
6 named axes (M1–M6 in `MUTATION_MENU.md`): no K/V shared-memory staging,
scalar per-lane dot products, no tensor cores, PV parallelized only over
head_dim, softmax rescale on `kBlockM=16` of 128 threads, scalar loads.

---

## Environment

- GPU: NVIDIA GeForce RTX 3070, sm_86, 46 SMs, 8192 MB VRAM (WSL2 passthrough)
- Driver: 610.47 (host), CUDA UMD 13.3
- `nvcc`: 12.0.140
- `nsys`: 2022.4.2.50, `ncu`: 2022.4.1.0 (both present, not yet exercised)
- PyTorch: **2.4.1+cu121** (see Discovery D1 — not the version originally installed)
- `torch_transformer_benchmark.py` default dtype: **float32**
- Working tree lives on `/mnt/c/...` (9p-mounted Windows filesystem under WSL2)
  — see Discovery D4, this has direct performance-measurement consequences.

---

## Discoveries

**D1 — The originally-installed `torch==2.13.0+cu132` cannot run on this
machine; it is not a project bug.**
`torch.cuda.is_available()` / `torch._C._cuda_getDeviceCount()` segfaulted
unconditionally. Isolated via raw `ctypes` calls: the WSL passthrough driver
library (`/usr/lib/wsl/lib/libcuda.so.1`) independently passes `cuInit`/
`cuDeviceGetCount` (1 device found) and a raw `libcudart.so.12` call to
`cudaGetDeviceCount` also succeeds — so the GPU/driver stack itself is fine.
The segfault persisted even under `LD_PRELOAD`/`LD_LIBRARY_PATH` forcing the
correct driver library, which rules out simple shadowing by the also-present,
also-wrong `libnvidia-compute-580` apt package. That leaves the CUDA 13.2
runtime bundled with `torch==2.13.0+cu132` as the actual incompatibility with
this WSL2 driver shim. **Fix:** reinstalled `torch==2.4.1+cu121` (cu121, a
combination with a long WSL2 track record) into `pytorch_env/`; CUDA now
initializes and runs correctly with no LD_* workarounds needed. `ninja` was
also missing (required by `torch.utils.cpp_extension.load`) and has been
installed. Neither prior `-cu132` torch nor the missing `ninja` are committed
project state, so no repo files changed for this fix — only the venv.
No decision needed; this is an environment prerequisite, not a kernel finding.

**D2 — The harness's placeholder `SHAPES` registry (`P0`–`P4`, `R0`–`R2`) is
not the official benchmark and its fp16/bf16 entries are needlessly fragile.**
Running `correctness_check.py --impl cuda` over the placeholder registry
produces `FAIL` on 5 of 8 shapes (`P1`, `P2`, `P3`, `P4`, `R2`) with max
absolute error up to 0.0625 and up to 2.4% of output elements failing
(`P3_wide_model`). This looked like a kernel bug. **It is not**: running the
identical check with `--impl sdpa` (plain `F.scaled_dot_product_attention`,
zero custom CUDA) fails on the *exact same shapes* with near-identical error
magnitudes (0.0078125, 0.01171875, 0.0625, 0.0078125, 0.0078125). Both
candidates are fp16/bf16 end-to-end (baseline is cast to the same dtype), so
this isn't a fp32-vs-fp16 comparison — it's normal reduction-order-dependent
fp16/bf16 rounding compounding across 6 stacked layers, occasionally
exceeding the tight `atol=0.002 / rtol=0.02` gate on a handful of elements,
in *both* the hand-written kernel and PyTorch's own fused SDPA path. The
official 14 shapes (CLAUDE.md / `prob-state.txt`) specify no dtype, and the
reference script's own CLI default is `float32`. Confirmed directly (see D3):
all 13 runnable official shapes pass at float32 with ~1e-6 error on both
`sdpa` and `cuda`. **Conclusion: the placeholder registry's fp16/bf16 shapes
are not representative of the grading configuration and should not be read
as "the kernel has a bug."** They are still worth keeping as a stress test
of fp16 precision margins, but `common.py`'s `SHAPES` list needs the real 14
shapes added (it currently has none of them) before it can be used to
evaluate official-shape performance directly.

**D3 — All 13 runnable official shapes pass correctness at float32; shape
`cfg08` cannot run at all; shape `cfg14` cannot run in its literal reference
form on any GPU.**
Ad-hoc script (not committed) built each of the 14 CLAUDE.md configs at
`dtype=float32`, ran `baseline` vs `sdpa` and `baseline` vs `cuda`:

  - cfg01–cfg07, cfg09–cfg13 (12 shapes): **PASS**, max abs error ~1e-6 to
    ~2e-6 for both `sdpa` and `cuda` — numerically exact to fp32 precision,
    no correctness concern at all for these.
  - **cfg08 (B=64, D=1024, H=4, N=128, FFN=1024): `cuda` impl throws
    `RuntimeError: head_dim 256 exceeds kMaxHeadDim 128`.** `D/H = 1024/4 =
    256`, and `current.cu` hard-codes `kMaxHeadDim = 128` with an explicit
    `TORCH_CHECK` (`current.cu:259-261`) that refuses to run rather than
    silently mis-executing. This is not a performance issue — the kernel
    cannot serve this official shape at all yet. `sdpa` handles it fine
    (50.79 ms median, PASS). This is the single highest-priority functional
    gap.
  - **cfg14 (B=32, D=1024, H=16, N=100000, layers=2): the *reference*
    `BaselineTransformer.forward` cannot execute this shape on any
    realistic GPU.** `BaselineSelfAttention` materializes
    `scores = Q @ K^T` of shape `[B, H, N, N]` in fp32:
    `32 * 16 * 100000 * 100000 * 4 bytes ≈ 19,073.5 GB` for that one tensor.
    This is not a memory-budget nuance to tune around; it is off by roughly
    six orders of magnitude from any GPU that exists. **The correctness gate
    for shape 14 cannot be evaluated against the literal reference
    forward().** Whatever grades this shape must be doing one of: (a)
    comparing against a memory-feasible reference (e.g. a chunked/streaming
    computation of the same math, or `F.scaled_dot_product_attention` with a
    memory-efficient backend, run once, itself never materializing `NxN`),
    or (b) not actually instantiating `BaselineTransformer` for this shape at
    grading time. This needs clarifying before shape 14 can be validated —
    see Risks. The project's own `CLAUDE.md` rule ("never materialize an
    N×N attention matrix" for N=100000) already anticipates this; it's now
    confirmed as a hard requirement, not just good practice.

**D4 — The custom CUDA kernel currently loses to plain PyTorch SDPA on every
official shape that ran, but roughly half of that gap is a harness
measurement artifact, not the kernel.**
Median latency, `sdpa` vs `cuda`, at float32 (see Benchmark Evolution table
below for the full 13-shape breakdown): the custom kernel is 1.3x–5.3x
*slower* than `sdpa` everywhere it ran. Suspicious pattern: shapes with very
different total work (B=1 to B=128, H=1 to H=16) cluster around a nearly
constant ~13–14 ms for the `cuda` impl, while `sdpa` scales down to 2.6 ms
for the same small shapes — consistent with a large *fixed* per-call cost
dominating the `cuda` path at small/medium sizes. Root cause found:
`common.py::load_cuda_extension()` calls `kernel_hash()` on **every** call,
which re-reads `current.cu` from disk and computes a fresh SHA-256 of it —
and `_CudaAttention.forward()` calls `load_cuda_extension()` on every
forward, i.e. **once per transformer layer, every forward pass** (4 layers
here). Measured directly: `kernel_hash()` costs ~2.4–2.7 ms per call on this
machine, because the repo lives on `/mnt/c` (a 9p-mounted Windows filesystem
under WSL2, with known high per-syscall latency). 4 layers × ~2.5 ms ≈ 10 ms
of pure disk-I/O overhead added to every forward pass, independent of GPU
work entirely. This does not fully explain the gap on the largest shapes
(cfg13: 309.8 ms vs 89.2 ms, a 220 ms difference far larger than 10 ms — real
kernel inefficiency also matters there, consistent with the M1–M6
known-suboptimalities already documented in `MUTATION_MENU.md`), but it means
**the current sdpa-vs-cuda comparison at small/medium shapes is not a fair
read of the kernel's actual GPU cost** and should not be used to judge M1–M17
mutations until fixed.

---

## Precision Decisions

| Operation | Storage dtype (official shapes) | Accumulator | Reason |
|---|---|---|---|
| Q/K/V projections | float32 | float32 (cuBLAS default) | matches reference; official shapes carry no dtype override, script defaults to float32 |
| QK^T (`current.cu`) | float32 in / float32 math | float32 (explicit `static_cast<float>` on every load) | kernel always accumulates in fp32 regardless of input dtype, matching reference's `torch.softmax(scores.float(), ...)` |
| softmax | float32 in kernel (`s_prob`, `s_max`, `s_sum` all `float`) | float32 | reference casts scores to `.float()` before softmax; kernel comment states this explicitly (`current.cu:32-33`) |
| PV accumulation | float32 (`s_acc` is `float`) | float32 | same rationale |
| output | cast to `scalar_t` only at final store (`current.cu:230`) | — | preserves storage dtype without narrowing intermediate math |

No precision boundary has been moved yet. The fp16/bf16 marginal failures in
D2 are a property of the reference-vs-any-fp16-implementation comparison
itself (also present in `sdpa`), not a cast point in `current.cu` that needs
correcting for the official (float32) shapes.

---

## Benchmark Evolution

Median latency (ms) at `dtype=float32`, single forward pass, 4-layer
Transformer, `--causal`. `sdpa` = `F.scaled_dot_product_attention` control
(zero custom CUDA). `cuda` = `current.cu` v0, unmodified, first time ever
run. All entries below are correctness-**PASS** except where noted.

| Shape | B | D | H | N | FFN | sdpa median (ms) | cuda median (ms) | cuda/sdpa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cfg01 (base) | 64 | 128 | 4 | 128 | 128 | 7.239 | 14.192 | 1.96x slower |
| cfg02 | 1 | 128 | 4 | 128 | 128 | 2.613 | 13.460 | 5.15x slower |
| cfg03 | 4 | 128 | 4 | 128 | 128 | 2.645 | 14.090 | 5.33x slower |
| cfg04 | 16 | 128 | 4 | 128 | 128 | 5.616 | 13.533 | 2.41x slower |
| cfg05 | 128 | 128 | 4 | 128 | 128 | 8.194 | 17.863 | 2.18x slower |
| cfg06 | 10000 | 128 | 4 | 128 | 128 | 6328.4 | 9189.9 | 1.45x slower |
| cfg07 | 64 | 32 | 4 | 128 | 32 | 10.492 | 13.587 | 1.29x slower |
| cfg08 | 64 | 1024 | 4 | 128 | 1024 | 50.792 | **ERROR** | head_dim 256 > kMaxHeadDim 128 |
| cfg09 | 64 | 128 | 1 | 128 | 128 | 3.272 | 13.752 | 4.20x slower |
| cfg10 | 64 | 128 | 2 | 128 | 128 | 3.809 | 14.574 | 3.83x slower |
| cfg11 | 64 | 128 | 16 | 128 | 128 | 5.906 | 24.134 | 4.09x slower |
| cfg12 | 64 | 128 | 4 | 32 | 128 | 2.636 | 13.805 | 5.24x slower |
| cfg13 | 64 | 128 | 4 | 1024 | 128 | 89.221 | 309.841 | 3.47x slower |
| cfg14 | 32 | 1024 | 16 | 100000 | 1024 | N/A (OOM by ~6 orders of magnitude) | not run | reference cannot execute |

Every row above must be re-measured once D4's hashing overhead is removed
before drawing conclusions about M1–M17 mutations from this table. As-is, it
establishes only: (a) correctness is currently fine at float32 everywhere
except cfg08/cfg14, (b) `sdpa` is the bar to beat, and (c) something in the
harness call path costs ~10ms/forward independent of the kernel.

Also on the placeholder registry (fp16/bf16, not official — see D2), `cuda`
and `sdpa` fail identically on `P1`, `P2`, `P3`, `P4`, `R2` and pass
identically on `P0`, `R0`, `R1`.

---

## Rejected Approaches

(none yet — no mutation has been attempted)

---

## Final Lessons

(to be filled in as the project progresses)

---

## Experiment 000 — Baseline characterization

### Date
2026-08-29

### Objective
Establish whether the environment, the harness, and the seed kernel
(`current.cu` v0, never previously compiled or run per `lineage.jsonl`) are
in a working, measurable state before attempting any optimization mutation.

### Context
Fresh repository inspection. `lineage.jsonl` contained only a `schema` row
and two `variant` rows (`v000` unverified, `baseline_sdpa` unverified) — no
correctness or benchmark records existed anywhere.

### Hypothesis
No hypothesis about the kernel itself yet; the working hypothesis was "the
environment and harness need to be validated end-to-end before any kernel
change can be attributed causally to anything."

### Proposed change
None — read-only investigation, plus one environment fix (D1: torch
reinstall + `ninja`) required to make anything runnable at all.

### Expected outcome
A working `torch.cuda`, a compiling `current.cu`, and a first real
correctness + performance record for the 14 official shapes.

### Measurements
See Discoveries D1–D4 and the Benchmark Evolution table above.

### Result
- Environment: fixed (D1).
- Correctness: seed kernel passes all 12 runnable official shapes at
  float32 (~1e-6 error), fails to run on cfg08 (head_dim limit), and cfg14's
  reference cannot execute in literal form on any GPU (D3).
- Performance: seed kernel is currently slower than `sdpa` on every shape
  that ran, partly a real algorithmic gap (M1–M6, expected — the kernel was
  authored deliberately unoptimized) and partly a harness measurement
  artifact (D4, ~10ms/forward from re-hashing the kernel source on disk on
  every layer's forward call).

### Discovery
See D1–D4 above.

### Decision
**INVESTIGATE FURTHER**, per-item:
- D1 (env): **KEEP** the torch 2.4.1+cu121 + ninja fix — required for anything to run.
- D2/D3 (correctness): **KEEP** float32 as the validated baseline; the
  placeholder fp16/bf16 shapes are not blocking but should be labeled as
  such in `common.py` rather than implying they're official.
- cfg08 head_dim limit: **BLOCKED**, needs a real fix (raise `kMaxHeadDim`
  and re-verify the shared-memory budget noted in the kernel's own comment)
  before cfg08 can be evaluated at all.
- cfg14 reference infeasibility: **BLOCKED**, needs a human/organizer
  decision on what shape-14 correctness is actually graded against before
  a streaming kernel can be validated against it.
- D4 (measurement artifact): **BLOCKED** on a harness fix (cache the kernel
  hash outside the per-forward hot path) before any M1–M17 mutation's
  performance number can be trusted at small/medium shapes.

### Next step
See "Proposed next 3 experiments" below.

---

## Proposed next 3 experiments (ranked by expected payoff)

1. **Fix the benchmark measurement artifact (D4).** Cache
   `kernel_hash()`/`load_cuda_extension()` per-process instead of re-hashing
   `current.cu` from disk on every `_CudaAttention.forward()` call. This is
   a harness-only change (`kernel-opt/harness/common.py`), touches no
   kernel code, and is a prerequisite for every subsequent measurement being
   trustworthy — right now roughly 10ms of every forward-pass timing is disk
   I/O, not GPU work, and it disproportionately hides the real cost on the
   small/fast official shapes (cfg02, cfg03, cfg12 etc. — exactly the shapes
   where launch/dispatch overhead matters most). **Expected payoff: large**
   (changes the apparent speedup of every future mutation) at **near-zero
   risk** (doesn't touch `current.cu`, cannot affect correctness).

2. **Fix cfg08 by raising `kMaxHeadDim` and re-deriving the shared-memory
   budget.** `current.cu` already computes `smem_bytes(head_dim)` generically
   and gates on `TORCH_CHECK(head_dim <= kMaxHeadDim, ...)` — the check
   exists specifically so this failure mode is loud rather than silent
   (comment at `current.cu:257-261`). D=1024/H=4 needs `head_dim=256`; the
   shared-memory formula in the file's own header comment
   (`(2*BLOCK_M*D + BLOCK_M*BLOCK_N + 3*BLOCK_M) * 4` bytes) gives
   `(2*16*256 + 16*64 + 48) * 4 ≈ 37.6 KB` at `head_dim=256`, still under the
   48KB default static-shared-memory limit on sm_86 — so this is expected to
   be a one-line constant change plus a recompute-and-confirm of that budget,
   not a redesign. **Expected payoff: required** (cfg08 is 1 of 14 official
   shapes and currently cannot run at all — this isn't an optimization, it's
   closing a functional gap) at **low risk** (constant change, existing
   budget math already anticipates larger head_dim; must re-run correctness
   on cfg08 and re-check cfg08 doesn't regress shared-memory occupancy for
   the already-passing shapes, since `kMaxHeadDim` only gates the check, not
   the launch config for smaller head_dim shapes).

3. **Profile the largest real algorithmic gap (cfg13, N=1024) with `ncu`
   before touching anything else in `current.cu`.** cfg13 is the shape where
   the harness artifact (D4) is smallest relative to total time (~10ms out
   of 310ms) and the kernel is still 3.47x slower than `sdpa` — this is the
   shape most likely to cleanly isolate a real algorithmic bottleneck (M1
   K/V re-read from global, M2 scalar warp-reduction, or M3 no tensor cores
   are the prime suspects per `MUTATION_MENU.md`, all trigger on
   `bound_by`/`tensor_pct`/stall-reason metrics that haven't been collected
   yet). Run `ncu --set full` on cfg13 through `benchmark.py --ncu` (once
   experiment 1 lands, so the correctness gate + benchmark interlock works
   cleanly), extract the top stall reason via `trim_ncu.py`, and use that —
   not intuition — to pick the first real mutation off the menu. **Expected
   payoff: high** (this is the shape with the most absolute latency to
   recover, and the profiling data will make mutation choice evidence-driven
   rather than guessed) at **zero risk** (profiling only, no code change).

Correctness-wise, cfg14 (the N=100000 shape) is explicitly **not** in this
list yet: it is blocked on a real ambiguity (D3) about what reference it
should even be checked against, and the project's own rules forbid
"skipping computation the reference performs" or "special-casing" a shape —
so a streaming-attention implementation for it needs that ambiguity resolved
first, not guessed around.
