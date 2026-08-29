# Mutation Menu

The bounded action space for the optimization loop. A variant is produced by
applying **exactly one** entry from this menu to `kernels/current.cu`.

This file exists to stop the loop from wandering. An unbounded "make it faster"
prompt produces rewrites that are hard to attribute: when three things change
and the result is 1.4x, you have learned nothing about which of the three did
it, and you cannot undo the one that hurt. One mutation per variant keeps
`lineage.jsonl` a causal record instead of a changelog.

---

## Rules

1. **One mutation per variant.** If a mutation requires a second to be legal
   (e.g. M7 vectorized loads requires an aligned `head_dim`), state the
   dependency and land the prerequisite as its own variant first.
2. **Correctness gates everything.** `correctness_check.py` must exit 0 before
   `benchmark.py` will time the variant. The gate matches on kernel SHA, so
   editing the `.cu` invalidates the previous pass.
3. **Every attempt is logged, including failures.** A mutation that broke
   correctness or regressed latency is the most valuable row in the file --
   it is what stops the loop retrying it. Never delete a lineage row.
4. **Pick from the trigger column, not from the top.** Each mutation lists the
   metric that justifies it. If the trigger metric is not present in the
   current `trim_ncu.py` output, the mutation is not indicated -- choosing it
   anyway is guessing.
5. **Revert on regression.** If median latency worsens on the target shape,
   restore the parent and mark the row `regressed`. Do not stack a second
   mutation onto a regression hoping it cancels out.
6. **Re-profile after every accepted variant.** The bottleneck moves. The
   trigger that justified mutation N is usually not the trigger for N+1.

---

## Menu

Trigger metrics refer to `trim_ncu.py` output keys.

### Memory / data movement

| ID | Mutation | Trigger | Expected effect | Risk |
|----|----------|---------|-----------------|------|
| **M1** | Stage K and V tiles in shared memory before the score loop, instead of reading global per dot product | `bound_by=memory-bound`, `top_stall=long_scoreboard`, low `l1_hit_pct` | Large. K is currently re-read `kBlockM` times per tile | Shared-memory budget; may cut occupancy |
| **M1b** | Double-buffer the K/V shared tiles (prefetch tile n+1 while computing n) | M1 landed and `long_scoreboard` still dominant | Moderate; hides global latency | Doubles smem for K/V; register pressure |
| **M6** | Vectorize global loads: `float4` for fp32, `half2`/`__half2` for fp16 | `sectors_per_request` >> 4.0 | Moderate; fewer, wider transactions | Requires `head_dim % 4 == 0` and 16B-aligned base pointers |
| **M7** | Pad shared-memory row stride to avoid bank conflicts (`head_dim + 1` or `+4`) | `top_stall=mio_throttle` or `short_scoreboard` | Small to moderate | Costs a little smem; easy to get the padding wrong |
| **M8** | Use `__ldg` / `__restrict__` read-only path for K and V | Low `l1_hit_pct` with high re-read | Small | Usually already applied by the compiler |

### Occupancy / parallel decomposition

| ID | Mutation | Trigger | Expected effect | Risk |
|----|----------|---------|-----------------|------|
| **M4** | Parallelize the PV accumulation over `(row, dim)` pairs rather than `dim` alone | `achieved_occupancy` low **and** `head_dim < kThreads` | Moderate; recovers the idle half-block at D=64 | Needs a cross-thread reduction or a different accumulator layout |
| **M5** | Spread the softmax rescale across warps instead of `tid < kBlockM` | `top_stall=barrier` | Small; 16/128 threads currently do this work | Requires a block-level max/sum reduction |
| **M9** | Retune `kBlockM` / `kBlockN` / `kThreads` | `waves_per_sm < 1` (too few blocks) or `achieved_occupancy << theoretical_occupancy` | Varies; often the cheapest real win | Pure retune -- try 32/64/128 and 32/128/256 |
| **M10** | Cut register pressure (shrink live ranges, `__launch_bounds__`) | `registers_per_thread` > 64 with `achieved_occupancy` capped | Moderate when register-limited | May spill to local memory and get worse |
| **M11** | Split-K / flash-decoding: parallelize over the key axis for short sequences | `recurrent` shapes with `waves_per_sm < 1` | Large on `R0`/`R1` where there is not enough work to fill the GPU | Needs a second reduction pass over partial softmax states |

### Compute

| ID | Mutation | Trigger | Expected effect | Risk |
|----|----------|---------|-----------------|------|
| **M3** | Use tensor cores (`mma.sync` / `wmma`) for QK^T and PV | `tensor_pct ≈ 0` with `bound_by=compute-bound` on an fp16/bf16 shape | Large -- the single biggest ceiling raise | High complexity; fragment layouts are easy to get subtly wrong. Land only with a shape-restricted guard |
| **M2** | Compute several scores per warp iteration to amortize the shuffle tree | `compute_pct` moderate, `arithmetic_intensity` low | Small to moderate | More registers per thread |
| **M12** | `__expf` → 2-based `exp2f` with the scale folded into the score | high `compute_pct`, SFU pressure | Small | Slight numerical drift; re-check against the tolerance |
| **M13** | Skip fully-masked key tiles early (causal / padding block skipping) | `P1`, `P3` (causal) and `P4` (30% padding) | Moderate on causal shapes; ~half the tiles are dead at large S | Needs a cheap per-tile validity test |

### Fusion scope (changes what the kernel covers)

| ID | Mutation | Trigger | Expected effect | Risk |
|----|----------|---------|-----------------|------|
| **M14** | Fuse the QKV projections into the kernel (currently torch matmuls) | many small kernels in `trim_ncu` output; `recurrent` shapes | Large on decode shapes where launch overhead dominates | Big rewrite; needs a GEMM inside the kernel |
| **M15** | Fuse LayerNorm + residual into one kernel | elementwise kernels visible in the profile with `bound_by=memory-bound` | Moderate; these are pure bandwidth | Must match the reference's fp32 LayerNorm accumulation |
| **M16** | Fuse the FFN `GELU` with its surrounding matmuls | GELU kernel visible and `memory_pct` high | Moderate on `P2` (FFN-heavy) | Must use exact `erf` GELU, not `tanh` -- the reference passes `approximate="none"` |
| **M17** | Dispatch a different kernel per shape regime | a mutation helps `parallel` but regresses `recurrent` | Enables specialization without a global regression | Explicitly allowed by problem statement 3.2 ("shape checks in the implementation") |

---

## Not on the menu

Out of bounds for the loop. These need a human decision:

- **Lowering precision below the input dtype** (fp16 accumulation, fp8, quantization). It buys speed against the tolerance budget, and the budget is not the loop's to spend.
- **Weakening the tolerance.** `rtol=0.02 / atol=0.002` comes from problem statement 3.2. It is a constant, not a tuning knob.
- **Touching `torch_transformer_benchmark.py`.** The reference must stay the organizers' code, or the comparison means nothing.
- **Skipping computation the reference performs** (approximate/sparse attention, dropped tiles that are not provably masked).
- **Special-casing the benchmark's fixed inputs** -- caching outputs, exploiting the seed, or anything else that would not survive an unseen shape.

---

## Lineage schema

`lineage.jsonl` is append-only; one JSON object per line. Both harness scripts
append automatically. Fields:

| Field | Meaning |
|-------|---------|
| `record_type` | `schema` \| `variant` \| `correctness` \| `benchmark` |
| `ts` | UTC ISO-8601, written by the harness |
| `variant` | label, e.g. `v007` (pass `--variant` to both scripts) |
| `parent` | the variant this was mutated from |
| `mutation` | menu ID applied, e.g. `M1` |
| `kernel_sha` | first 16 hex of the `.cu` SHA256 -- ties every row to a build |
| `status` | `pass` / `fail` / `error` (correctness); `ok` / `partial` / `gated` (benchmark) |
| `max_abs_error`, `max_relative_error` | worst observed across shapes |
| `failed_shapes` | shape ids that failed the gate |
| `median_ms`, `speedup` | per-shape maps, benchmark rows only |
| `ncu` | the `summary` block from `trim_ncu.py` |
| `notes` | free text: why this mutation, what happened |

A `variant` row is written by hand (or by the loop) to record intent *before*
the run; `correctness` and `benchmark` rows are written by the harness after.
