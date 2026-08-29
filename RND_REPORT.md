# R&D Report — Pure-CUDA Transformer Attention Kernel Optimization

Status: **baseline-characterization phase complete; first optimization mutation not yet applied.**
This report was cut short by token/credit budget; it documents everything actually
done, discovered, and measured, not a finished optimization.

---

## 1. What we did

1. Inspected the repo cold: reference benchmark (`torch_transformer_benchmark.py`),
   an existing but never-run CUDA kernel (`kernel-opt/kernels/current.cu`, v0,
   a tiled online-softmax fused-attention kernel), a harness
   (`kernel-opt/harness/{common,correctness_check,benchmark}.py`), a mutation
   menu (`kernel-opt/MUTATION_MENU.md`), and an append-only ledger
   (`kernel-opt/lineage.jsonl`) that had zero real runs recorded.
2. Diagnosed and fixed a blocking environment bug: `torch==2.13.0+cu132`
   segfaulted on `torch.cuda.is_available()`. Isolated via raw `ctypes` calls
   into `libcuda`/`libcudart` that the WSL2 GPU passthrough itself was fine;
   the installed CUDA-13.2 torch build was not. Reinstalled
   `torch==2.4.1+cu121` + `ninja` into the project venv. No repo files
   changed for this — venv only.
3. Compiled `current.cu` for the first time ever and ran the harness's
   correctness gate (`correctness_check.py`) against both the harness's
   placeholder shape registry and, via a throwaway script, all 14 official
   shapes from `.claude/CLAUDE.md` at `dtype=float32`.
4. Found and fixed a real harness bug (`kernel-opt/harness/common.py`):
   `kernel_hash()` re-read and SHA-256'd `current.cu` from disk on **every**
   forward call (once per transformer layer), and the repo lives on a
   9p-mounted `/mnt/c` path under WSL2 where that costs ~2.5ms/call — adding
   ~10ms of pure disk I/O to every 4-layer forward pass, unrelated to GPU
   work. Fixed by caching the digest per-process. This is the only change
   made to any file other than `experiments.md`/this report; `current.cu`
   itself was never touched.
5. Delegated nsys/ncu profiling workflow setup to a `performance-profiler`
   subagent. It found **`ncu` is blocked at the Windows host driver level**
   (GPU performance-counter access denied to this WSL2 guest — the classic
   `ERR_NVGPUCTRPERM`-class restriction) and was mid-way through building a
   CUPTI-based launch-count/duration fallback when it was cut off by an API
   credit exhaustion (`billing_error`, model `claude-opus-5`). **No ncu/nsys
   metrics were obtained.** This is a real, reproducible environment
   constraint, not a workaround-able bug — profiling-by-counter needs a
   different host permission configuration or a different machine.
6. Logged everything as it was found in `kernel-opt/kernels/../experiments.md`
   (Discoveries D1–D4, a benchmark table, precision-decision table, and a
   ranked 3-experiment proposal), following the project's own
   experiment-logging methodology.

## 2. What we discovered

- **The custom kernel v0 is currently slower than plain PyTorch
  `F.scaled_dot_product_attention` on every official shape that ran**, by
  1.3x–5.3x depending on shape. This was expected in part — the kernel's own
  header comment lists 6 known-suboptimalities (M1–M6: no K/V shared-memory
  staging, scalar per-lane dot products, no tensor cores, PV parallelized
  only over head_dim, softmax rescale on 16/128 threads, scalar loads) — but
  part of the gap was a measurement artifact (item 4 above), not real GPU
  cost.
- **Correctness is fine at the official float32 default**: 12 of the 14
  official shapes pass with ~1e-6 error (essentially exact) for both `sdpa`
  and the custom kernel.
- **Official shape 8 (D=1024, H=4 → head_dim=256) cannot run at all**:
  `current.cu` hard-caps `kMaxHeadDim=128` and throws rather than
  mis-executing. This is a functional gap, not a performance one, and is
  the single highest-priority fix — one constant plus a re-check of the
  kernel's own documented shared-memory formula (which already appears to
  fit under 48KB at head_dim=256, per hand calculation in `experiments.md`).
- **Official shape 14 (N=100000) cannot be checked against the literal
  reference at all**: `BaselineSelfAttention` would materialize a `[B,H,N,N]`
  fp32 attention matrix of ~19,073 GB. This isn't a memory-tuning problem;
  it's six orders of magnitude beyond any GPU. Whatever grades this shape
  must use a different, memory-feasible reference — this needs an explicit
  answer before a streaming/tiled implementation for it can be validated,
  and the project's own rules forbid guessing around that ambiguity
  (no "skip computation" or "special-case the shape" shortcuts allowed).
- **The harness's own placeholder fp16/bf16 shapes (not the official 14)
  fail correctness** on both the custom kernel and plain `sdpa`, with nearly
  identical error magnitudes — ordinary fp16 rounding compounding over 6
  stacked layers against a tight 0.002/0.02 gate, not a kernel bug.
- **`ncu`/`nsys`-based counter profiling is blocked in this WSL2
  environment at the host driver level.** Bottleneck classification (compute-
  vs memory- vs latency-bound) for `current.cu` has **not** been empirically
  established — the 3 proposed experiments below are therefore prioritized
  by the kernel's documented known-suboptimalities and by wall-clock shape
  comparisons, not by ncu evidence. This is an explicit, acknowledged gap.

## 3. Benchmark matrix (float32, official shapes, median ms)

Measured with CUDA events, alternating baseline/candidate rounds. First
sweep (pre-harness-fix) vs. a partial re-check (post-fix, contaminated by a
concurrently-running profiler subagent sharing the same GPU — flagged, not
clean). Take the post-fix numbers as directional only.

| Shape | B | D | H | N | FFN | sdpa (ms) | cuda pre-fix (ms) | cuda post-fix (ms, GPU-shared, noisy) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cfg01 | 64 | 128 | 4 | 128 | 128 | 7.24 / 4.22 | 14.19 | 16.38 (noisy) |
| cfg02 | 1 | 128 | 4 | 128 | 128 | 2.61 | 13.46 | 16.31 (noisy) |
| cfg03 | 4 | 128 | 4 | 128 | 128 | 2.64 | 14.09 | — |
| cfg04 | 16 | 128 | 4 | 128 | 128 | 5.62 | 13.53 | — |
| cfg05 | 128 | 128 | 4 | 128 | 128 | 8.19 | 17.86 | — |
| cfg06 | 10000 | 128 | 4 | 128 | 128 | 6328.4 | 9189.9 | — |
| cfg07 | 64 | 32 | 4 | 128 | 32 | 10.49 | 13.59 | — |
| cfg08 | 64 | 1024 | 4 | 128 | 1024 | 50.79 | **ERROR: head_dim 256 > kMaxHeadDim 128** | — |
| cfg09 | 64 | 128 | 1 | 128 | 128 | 3.27 | 13.75 | — |
| cfg10 | 64 | 128 | 2 | 128 | 128 | 3.81 | 14.57 | — |
| cfg11 | 64 | 128 | 16 | 128 | 128 | 5.91 | 24.13 | — |
| cfg12 | 64 | 128 | 4 | 32 | 128 | 2.64 | 13.81 | — |
| cfg13 | 64 | 128 | 4 | 1024 | 128 | 89.22 / 91.12 | 309.84 | 285.72 (noisy, ~8% down) |
| cfg14 | 32 | 1024 | 16 | 100000 | 2 layers | N/A — reference needs ~19,073.5 GB, cannot run | not run | not run |

The one clean-ish signal from the post-fix partial re-check: cfg13 (the
shape with the largest absolute real gap, and where the ~10ms/forward
hashing artifact is smallest relative to total time) improved from 309.8ms
to 285.7ms even while sharing the GPU with another process — consistent
with the fix removing real, if modest, overhead. cfg01/cfg02 appeared to get
*slower* post-fix, which is physically inconsistent with a pure-overhead-
removal fix and is attributed entirely to GPU contention with the
concurrently-running profiler subagent, not the fix itself. **A clean,
uncontended full re-sweep was not completed before the budget ran out.**

## 4. Bottleneck assessment (no ncu evidence — documented-suboptimality basis only)

`ncu`/`nsys` counter access is blocked at the host level (Section 2), so this
is based on the kernel's own documented design gaps and wall-clock shape
scaling, not measured occupancy/stall data:

- Small/medium shapes (cfg01–cfg05, cfg07, cfg09–cfg12): `cuda` latency
  clusters tightly (~13–17ms) across wildly different batch/head counts,
  while `sdpa` scales down to ~2.6ms for the smallest — indicative of a large
  **fixed per-forward cost** dominating at these sizes. Part of this was the
  now-fixed hashing artifact; the residual fixed cost (JIT-extension call
  overhead, per-layer Python/kernel-launch overhead, lack of QKV/attention
  fusion) is unquantified without profiling.
- Large-N shape (cfg13, N=1024): the gap is large in absolute terms (ms) and
  is the best candidate for a genuine algorithmic bottleneck, matching the
  kernel's documented M1 (K/V re-read from global every tile, no shared-
  memory staging) and M2 (scalar per-lane warp-reduction dot products)
  suboptimalities — both scale with sequence length, which lines up with
  cfg13 being both the worst absolute performer and the shape least diluted
  by fixed per-call overhead.
- No tensor-core usage (M3) on an fp32 path is expected to matter most at
  large head_dim/large-N shapes (cfg08, cfg13) — unconfirmed without
  `tensor_pct` from ncu.

## 5. Three proposed experiments (ranked, evidence-based given available data)

### Experiment 1 — Fix `kMaxHeadDim` for official shape 8 (functional gap, not perf)
1. **Hypothesis:** `current.cu`'s `kMaxHeadDim=128` constant is an
   arbitrary compile-time cap, not a hardware limit; raising it to 256 (and
   re-deriving `smem_bytes()`) will let cfg08 run without touching the
   algorithm.
2. **Bottleneck targeted:** functional (shape cannot execute at all), not
   a throughput bottleneck.
3. **Expected mechanism:** the kernel's own shared-memory formula
   `(2*BLOCK_M*D + BLOCK_M*BLOCK_N + 3*BLOCK_M) * 4` bytes gives ≈37.6KB at
   head_dim=256 — under the 48KB default static-shared-memory budget on
   sm_86, so no `cudaFuncSetAttribute` opt-in should be needed.
4. **Numerical risk:** none expected — the softmax/accumulation math is
   unchanged; only the compile-time bound and shared-memory offsets move.
   Must re-run correctness on cfg08 to confirm.
5. **Validating measurement:** `correctness_check.py --impl cuda` on cfg08
   passes with max_abs_error consistent with the other float32 shapes
   (~1e-6), and shared-memory usage stays under budget (compile succeeds,
   no `cudaErrorInvalidValue` at launch).

### Experiment 2 — Stage K/V tiles in shared memory (menu M1)
1. **Hypothesis:** K and V are currently re-read from global memory on every
   score computation within a tile (`current.cu`'s own comment M1); staging
   them once per tile in shared memory should cut redundant global traffic
   substantially, especially at larger N (cfg13).
2. **Bottleneck targeted:** memory traffic / redundant global reads — the
   menu's stated trigger (`bound_by=memory-bound`, `long_scoreboard` stalls)
   could not be confirmed by `ncu` (blocked), so this is applied on the
   documented-suboptimality basis, explicitly flagged as unconfirmed by
   direct evidence per the project's own "don't guess from the trigger
   column" rule.
3. **Expected mechanism:** fewer global-memory transactions per tile,
   better L1/L2 reuse of K/V across the `kBlockM` query rows sharing a tile.
4. **Numerical risk:** low — same FMA order per (row,key) pair, just sourced
   from shared instead of global memory; must re-verify bit-for-bit-enough
   agreement (atol/rtol gate) since shared-memory staging can change bank-
   conflict-driven scheduling but not the arithmetic itself.
5. **Validating measurement:** median latency on cfg13 (the shape with the
   most N-dependent absolute gap) before/after, correctness gate on all 13
   runnable official shapes, and — if `ncu` access is ever restored —
   `sectors_per_request`/`l1_hit_pct` before/after to confirm the traffic
   reduction actually happened rather than assuming it from the latency
   number alone.

### Experiment 3 — Retune launch configuration (menu M9)
1. **Hypothesis:** the fixed `kBlockM=16, kBlockN=64, kThreads=128` launch
   config was chosen once, generically, and may under-fill the GPU (46 SMs)
   at small-batch official shapes (cfg02: B=1, cfg09: H=1), where the flat
   ~13–17ms floor suggests too few blocks/waves rather than genuine compute
   cost.
2. **Bottleneck targeted:** occupancy / grid utilization at small-batch
   shapes specifically.
3. **Expected mechanism:** a smaller `kBlockM` or larger grid decomposition
   increases the number of independently schedulable blocks, filling more
   SMs concurrently when `batch × heads × (seq_len/kBlockM)` is small.
4. **Numerical risk:** none — this changes only parallel decomposition, not
   arithmetic order within a (row, key) pair.
5. **Validating measurement:** median latency on cfg02 and cfg09
   specifically (the smallest-batch/smallest-head-count official shapes)
   before/after a small controlled sweep of 2–3 configs (per
   `MUTATION_MENU.md`'s own guidance: "try 32/64/128 and 32/128/256", not
   exhaustive search), plus confirmation the change doesn't regress cfg13.

**None of these three has been executed yet** — Experiment 1 is the
recommended next action (required functional fix, lowest risk, already
budgeted math), but no code change beyond the harness hashing fix
(Section 1, item 4) has been applied to `current.cu` in this session.

## 6. AI tools, agents, and skills utilized

- **Model:** Claude Sonnet 5 (main session); one subagent run attempted on
  `claude-opus-5` (see below).
- **Skill:** `optimiser-CUDA` (`.claude/skills/optimiser-CUDA/cuda-optimise`)
  — the pure-CUDA, evidence-driven optimization methodology this session
  followed (baseline-first, one-hypothesis-per-change, cuBLAS-before-custom-
  kernel, precision-boundary discipline).
- **Subagent: `performance-profiler`** (model `opus`) — dispatched to
  establish the `nsys`/`ncu` profiling workflow on official shapes cfg01 and
  cfg13. Discovered that `ncu` GPU performance-counter access is blocked at
  the Windows host driver level in this WSL2 configuration, and had begun
  building a CUPTI-based fallback for launch counts/durations when it was
  terminated by an Anthropic API billing error (`credit balance too low`,
  model `claude-opus-5`) before completing. Its partial finding (`ncu`
  blocked; CUPTI fallback in progress) is the only output obtained from it.
- **Agent definitions present but not used this session** (available for
  next steps): `kernel-optimizer` (opus, owns `current.cu` edits),
  `correctness-debugger` (sonnet, numerical divergence isolation),
  `benchmark-analyst` (sonnet, cross-shape dispatch decisions),
  `experimentation-historian`/`logger.md` (sonnet, owns `experiments.md`
  — its schema and required experiment format were followed manually by the
  main session when writing `kernel-opt/experiments.md`, rather than
  delegating, to keep the record consistent with the diagnostic work
  happening in the same thread).
- **Skill: `numerical-debug`** (`.claude/skills/numerical-debug`) — informed
  the diagnostic order used when the placeholder-registry correctness
  failures first appeared (input → Q/K/V → QK^T → mask → softmax → P@V →
  output), before the `sdpa`-control comparison revealed they weren't a
  kernel bug at all.
- **Harness tooling used as-is (not AI, but load-bearing):**
  `kernel-opt/harness/{common,correctness_check,benchmark}.py`,
  `kernel-opt/tools/trim_ncu.py` (never actually exercised, since `ncu` was
  blocked), `kernel-opt/MUTATION_MENU.md` (used to select and justify the
  ranked experiments above), `kernel-opt/lineage.jsonl` (append-only log,
  populated automatically by `correctness_check.py` runs in this session).

## 7. Files changed this session

- `kernel-opt/harness/common.py` — cached `kernel_hash()` per-process
  (bug fix, not a kernel mutation; see Section 1 item 4).
- `kernel-opt/experiments.md` — created; full discovery log, benchmark
  table, precision decisions, ranked experiment proposals.
- `RND_REPORT.md` — this file.
- `pytorch_env/pytorch_env` (venv only, not version-controlled repo logic) —
  `torch` downgraded `2.13.0+cu132` → `2.4.1+cu121`; `ninja` installed.
- **`kernel-opt/kernels/current.cu` was not modified.** No optimization
  mutation has been applied yet.

## 8. Honest limitations of this report

- No `ncu`/`nsys` counter data was obtained — the bottleneck assessment
  (Section 4) is inference from documented kernel design and wall-clock
  shape scaling, explicitly not confirmed by profiler evidence, which the
  project's own methodology treats as a lesser form of evidence
  ("do not optimize based on intuition alone... choosing [a mutation]
  without the trigger metric present is guessing").
- The post-fix benchmark re-check (Section 3) was contaminated by a
  concurrently-running subagent sharing the GPU and should be re-run in
  isolation before being trusted for absolute numbers.
- Shape 14's correctness contract is genuinely unresolved and requires an
  answer from whoever owns the grading harness, not an engineering
  workaround.
