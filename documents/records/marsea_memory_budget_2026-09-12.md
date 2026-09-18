# MarSea — the three profiling fixes verified, and an independent memory budget

*2026-09-12. Read against the code as uploaded 2026-09-12: `primitives.py`, `normalizer.py`,
`chunked.py`, `invariants.py`, `backbone.py`, `heads.py`, `relation.py`, `profile_memory.py`,
`preflight.sh`, `time_step.py`, `train.py`, `run_s2.sh`, and the four tests cited.*

---

## Part 1 — the three fixes: all three are in, with three caveats

### (a) Chunked path in bf16 under autocast — **fixed, and fixed at the right two levels**

`marsea_chunked_attention` wraps the body, and `_chunk_fanout` wraps again *outside* the body that
`cp.checkpoint` recomputes — split into `_chunk_fanout` / `_chunk_fanout_inner` precisely so the
guard survives re-entry. All three checkpoint sites in the codebase
(`chunked._marsea_chunked_attention`, `normalizer._normalize_by_head_block`,
`backbone._checkpoint_layer`) use `use_reentrant=False` and every recompute body lands inside a
guard. The dense guard is still on `_normalize_one`. I found no remaining mixed-dtype matmul.

**Caveat.** `test_chunked_is_fp32_under_autocast` runs under `torch.no_grad()`. The
`_chunk_fanout` guard exists for exactly one scenario — the checkpointed recompute under ambient
autocast — and **no test exercises it.** `test_t11_checkpointed_backward_runs_fp32` runs a backward
but on CPU, fp32, no autocast, asserting only `isfinite`. One test that runs the chunked backward
under `autocast(bf16)` and asserts `torch.equal` on `cbar_j`/`tau_j` would close it; without it, the
fix is correct but unprotected against the next refactor.

### (b) Dense diagnostics rebuilding all heads — **fixed, and it narrows inputs rather than indexing after the fact**

`dense_head` slices `q`, `k_rep`, `vis` and `lse` *before* every allocation, so `E`, `S_full`,
`A_sm`, `Atil`, `a1`, `A`, `p` are each `[B,1,T,T]`; `h_i` is remapped to head-space 0; the eleven
per-head 1-D diagnostics are sliced to match; `_fp32_scores` is narrowed the same way. 2.4 GB
rather than 29 GB at 8K, as the comment claims.

**Caveat, and it matters for question 2.** `dense_head` is a **retained-side** fix, not a peak-side
one: passes 0/1/2 still run all twelve heads, and `keep = (bhi[:,1] == dense_head)` selects from
already-computed all-head sparse arrays. `profile_memory.py`'s docstring draws this distinction
correctly, but "29 GB → 2.4 GB" reads like a peak reduction and is not one. Also untested: the
backbone plumbing (`ctx.keep_dense_head`), `psi_j`/`cap_binds` (both sliced by the code), and the
bf16 shape of that block — the fix's own comment admits "which the fp32-only tests never saw", and
that admission applies to the test too.

### (c) INV-3 vs the unit cap's slack — **fixed, but with literally zero margin, and the cited test is the wrong one**

`tol3 = max(tol_sum, CAP_TOL)` with `CAP_TOL = 1e-6`, tested as `rs <= 1 + tol3`, inclusive. So fp64
now admits a row carrying exactly `1 + 1e-6` — and rejects `1 + 1.0000000001e-6`. But
`proj_le_masked`'s own test is strict (`binds = sw > s + bind_tol`), so the cap genuinely admits
rows *up to and including* `1 + CAP_TOL`, and an fp64 sum over `n_k` terms carries its own
summation error **on top of** that. The tolerance should be `max(tol_sum, CAP_TOL) + tol_sum`: the
cap's documented slack *plus* the summation error, not the max of the two. As written the fp64 path
sits exactly on the boundary it is meant to clear.

Two related loose ends: `INV-5` hard-codes `1.0 + 1e-6` rather than importing `CAP_TOL`, so the two
will drift apart the first time anyone tunes the cap; and **the cited test does not cover this at
all** — `test_a5_a6_hit_defined_only_where_thm2_speaks` exercises `fidelity.column_fidelity`'s
Thm.-2 exclusion rules and never touches `invariants.py`. The only test that calls
`invariant_report` directly is `test_t3_invariants_random`, and `conftest.rand_case` builds fp32
cases with `n_q ≤ 40`, where `tol_sum = 1e-5` already dominates `CAP_TOL` — **so the fp32 path never
had the bug and that test cannot regress the fix.** Real fp64 coverage exists only as a side effect
of `MARSEA_DEBUG=1` running the suite, through tests written for other purposes
(`test_t16_tau_i_one_is_identity_on_step1_row`, which does assert >50 cap-binding rows in fp64, is
the closest thing to a guard). Three lines of direct test would fix that.

---

## Part 2 — the memory budget, computed independently

### 2.1 The model is not the question

bf16 weights + fp32 LoRA masters + AdamW state = **2.9 GB**. There is nothing to shard. Sharding a
2.9 GB model across GPUs to relieve a ~130 GB activation bill is not a lever.

### 2.2 Unit of account

One `[1, 12, T, T]` fp32 tensor: **0.81 GB at 4K, 3.22 GB at 8K, 12.9 GB at 16K.**

An inventory of `_normalize_one` finds **~45 fp32, ~15 bool and 4 int64 tensors of T×T shape per
patched layer per forward** — about 55–65 distinct allocations, of which ~28 fp32 are retained for
backward. `profile_memory.py`'s docstring estimate of "10 fp32 `[1,H,T,T]` tensors" undercounts by
roughly 4×, and spec §6.5's budget `4 · 4B · H · T² · L_patch` undercounts by about the same — which
is why §6.5's "≤ 8 patched layers at 8K on 80 GB" was never going to hold.

Where the count comes from, in case you want to audit it: `g` alone costs 7 T²-shaped tensors;
`column_stats` costs 6; `row_summary` costs 9 to produce a `[1,H,T,4]` feature vector; each
`proj_le_masked` costs ~13 plus a full `sorted_prefix_stat`; and `sorted_prefix_stat` itself holds
~10 fp32 + one int64 (2 fp32-equivalents, 6.4 GB at 8K) + 4 bool live simultaneously at its
`G = torch.where(...)` line.

### 2.3 **Gradient checkpointing means the peak is set by ONE patched layer**

`_checkpoint_layer` wraps each patched decoder layer, so no class-(ii) tensor survives the forward,
and during the backward exactly one layer's set is re-materialised at a time. **Dropping 4 patched
layers to 2 therefore does not halve the peak.** It removes the retained `E` (0.81 GB/layer, on
logging steps only) and converts two layers back to SDPA's O(T·d) activations — call it 2–5 GB. The
binding term is unchanged.

That reframes the decision you were asked for: of the two pre-committed fallbacks, one of them is
not a memory lever at all.

### 2.4 Dense path

Backward peak ≈ one layer's ~28 retained fp32 T² + one `sorted_prefix_stat`'s ~12 fp32-equivalent
transient ≈ **40 tensor-equivalents**:

| T | dense peak (1 layer) | + weights | 80 GB H100 |
|---|---|---|---|
| 4K | 32 GB | ~36 GB | **fits** |
| 8K | 129 GB | ~133 GB | no |
| 16K | 516 GB | ~519 GB | no |

The 8K figure reproduces `run_s2.sh`'s own comment ("the DENSE path needs ~133 GB") to within a
percent, from an independent count. It does **not** reproduce the 200 GB you were quoted — see 2.6.

### 2.5 Chunked path at 8K — it fits, until a trained quantity says otherwise

| term | 8K, 4 layers, ρ=0.05 | shape |
|---|---|---|
| per-chunk transient (C=1024) | 24 GB | O(H·T·C) — genuinely linear |
| differentiable-`lse` loop | 3.2 GB | **O(T²), un-checkpointed — a bug** |
| sparse relation store | 2.0 GB | O(ρ·H·T²), ~44–100 B/entry — this one is the mechanism |
| ragged fan-in pad | 1.3 GB | O(T·L_max) |
| retained `E`, logging steps | 3.2 GB | never dropped by the backbone's filter |
| weights + optimizer | 2.9 GB | |
| **pass-2 densification** | **8.3 GB at frac_over = 1 %** | **O(frac_over · H · T² · d), uncapped** |

Sum at frac_over = 1 %: **~45–54 GB** — which matches `run_s2.sh`'s "~54 GB" comment, and **fits
under `preflight.sh`'s 72 GB gate.**

**But the last row is sized by a quantity nothing caps, and its own pre-registered healthy range
blows the budget.** Pass 2 gathers `k_rep_full[b_r, h_r]` and `v_rep_full[b_r, h_r]`, each
`[n_over, n_k, d]` — 128× larger than the `[n_over, n_k]` score tensor beside them. With
`n_over = frac_over · H · T`:

| frac_over | pass-2 gathers at 8K | at 16K |
|---|---|---|
| 1 % | 8.3 GB | 33 GB |
| **10 %** | **82 GB** | **330 GB** |

The training procedure §11 lists "no cap-binding on more than ~10 % of rows" as what a *healthy*
run looks like. So **the S2 queue can OOM mid-run on a run that is behaving exactly as designed**,
and the failure scales as T², arriving first at 16K where E3 lives. `test_t11_chunked_equals_dense`
asserts the branch fires, but only at T ≤ 1024, where the same 10 % costs 1.3 GB and is invisible.

This is the one item I would not ship without fixing. It is also cheap: group `ridx` by `(b_r, h_r)`
and use a plain `matmul` against `k_rep_full[b, h]` per group, which removes the
`[n_over, n_k, d]` gathers entirely; or failing that, loop pass 2 over fixed-size row blocks.

### 2.6 Where the 200 GB figure comes from, and why I don't use it

`profile_memory.py` fits `peak − weights = a·T^p` by least squares in log space from measurements at
T ≤ 2048. Extrapolating a T² term 16× has a 16× lever arm on any error in `p`, which the caveat
already notes — and the result disagrees with the repo's own `runs/memory_model.json` (133 GB dense
at 8K) by 1.5×. Two further reasons to distrust the extrapolated table before `preflight.sh` runs on
the H100:

- `n4 = 4 / len(layers)` rescales the **entire** activation term to 4 layers, i.e. it assumes zero
  backbone-fixed activation. Harmless at `--layers 4`, a 2× overstatement at the default 2.
- `retained_GB` misses the chunked path's sparse store entirely: `extra["sparse"]` is a *dict* of
  tensors, so `torch.is_tensor` is False and none of it is counted (~2 GB at 8K, ~12 GB at ρ=0.3).

And the header prints "fitted as base + a*T^2" while the code fits a free exponent — worth fixing
before anyone reads the output twice.

**More importantly: "does not fit at 8K on either the dense or the chunked path" contradicts the
number written into `run_s2.sh`.** One of the two is wrong, and `preflight.sh` settles it on the
target GPU in twenty minutes with no lever arm. I would not take a scientific decision until it has
run.

---

## Part 3 — what I'd do

### Four engineering fixes, ranked by GB per line of diff

1. **Cap or restructure pass-2's densification** (~8 GB at 1 %, ~82 GB at the healthy 10 %, and the
   only unbounded term). Also make `frac_rows_over_unit` a logged *gate* in `train_step`, not just a
   diagnostic — it is currently the one number that can end a run and nothing watches it.
2. **Delete the duplicate `sorted_prefix_stat`.** It is called **four** times per `normalize`, and
   call 2 —
   `with torch.no_grad(): kstar_l, psi_j, _, _, _ = sorted_prefix_stat(zcol, ET)` — recomputes, on
   the identical input, a statistic call 1 already produced and threw away inside
   `SparsemaxMasked.forward`. Thread `(kstar, psi)` out of the Function. ~12 fp32-equivalent T²
   transients and one full sort per layer per forward; ~25 % of the dense normaliser's transient
   peak, ~5 GB of the chunked peak because it is doubled per chunk. Zero numerical cost.
3. **Checkpoint the differentiable-`lse` loop** (3.2 GB at 8K, 12.9 GB at 16K). `torch.exp` saves
   its output and the loop has no `cp.checkpoint`, so it retains exactly one full dense
   `[1,H,T,T]` — the chunked path's most avoidable quadratic.
4. **Fix `want_dense_diag`'s gate.** `backbone.py` sets it on `ctx.collect or keep_dense or
   layer in keep_dense_layers`, then immediately nulls `p, a1, Atil, A_sm, u, A, logits` unless
   `keep_dense`. So on every 50th step with head blocking on, the code concatenates seven full-size
   `[1,H,T,T]` copies and throws them away. Drop `ctx.collect` from the condition.

Post-fix 8K chunked lands around **28–32 GB**, and 16K teacher-forced (E3, forward-only, so no
class-(ii) set and the `lse` loop doesn't run) around **50 GB** — which it does **not** reach today,
because pass 2 alone is 33 GB there at 1 % and 330 GB at 10 %.

### Two bugs found on the way

- **`train.build_model` raises `NameError: akw`.** `akw = dict(cfg.arm_kwargs)` is bound only in the
  `else` branch, but `patch_model(...)` runs unconditionally, so `build_model(cfg, phase_a=True)`
  with `phase_a_patched=False` crashes. The `if` branch's intent (leave the model unpatched for
  Phase A) is defeated anyway — `patch_model` patches regardless. `phase_a_patched=True` hides it,
  and `phase_a_kernel_check` exists to measure exactly the case that cannot currently run.
- **Head blocking is real, tested to 1e-12 in fp64 on values *and* gradients
  (`test_head_blocking_is_arithmetically_identical`, `test_head_blocking_gradients_agree`), and
  wired to nothing.** No `--head_block` in `run_s2.sh`; `TrainConfig.head_block = None`; and
  `marsea_chunked_attention` never reads `norm.head_block`, so **the chunked path cannot head-block
  at all** — which is the path S2 runs.

  Worth knowing what it could be worth. I checked every reduction in the normaliser, both τ heads,
  the relation head, the ν handoff, `column_stats` and `row_summary`: **nothing contracts the head
  dimension anywhere.** The forward is embarrassingly parallel over Q-heads, so the ceiling is
  `H / head_block` = up to 12× on the T²-shaped terms, not the 1.9× the test docstring measures.
  The gap is items 1–3 of "what it does not reduce": the output `torch.cat`, the wrongly-gated
  diagnostics concat (fix 4 above), and the backbone's own full-size `S`, `masked_fill` and value
  matmul outside the normaliser. Three things to handle at assembly: `dK_kv` must be summed over
  each group of g = 6 Q-heads in the backward (`repeat_kv`'s expand-backward), `L_max` in the
  chunked path is a global max over the head-flattened index, and two diagnostic scalars (`rho`,
  `row_trigger_violation`) plus the invariant verdict pool heads — all trivially recombinable.

### On the decision you were asked for

**I would not spend 4K, and 2 layers isn't the lever it looks like.**

*4K training is the most expensive option scientifically.* D-27 commits E3 to L = 16K held constant
across the n-sweep and E2 to 8K, so 4K training is off-length for **both** evaluation grids. The
damage is specific and lands on the quantity the paper leans hardest on: `TauK` reads
`column_stats` — max, second max, mean, **std**, gap, z-score — every one of which is a statistic
whose distribution shifts with the number of visible entries, and its bias is calibrated at
init to `softplus⁻¹(median_j(1/std_j) − τ_min)` *at training length*. A 4K-trained `TauK` meeting
16K columns meets systematically smaller `std`, i.e. a systematically larger `1/std` target, off
the calibration point. **The interval-hit rate is what that endangers** — App. H's "sharpest single
test of the theory" and E7's only genuinely empirical number.

*2 patched layers saves 2–5 GB, not half* (§2.3). And if you do take it, the layer set is **not**
free: D-9a puts the column test at the head S0 finds — (14, 3) — and the row test at the detector's
(l*, h*) = (19, 3), and both must sit in `PATCHED_LAYERS`. So 2 layers is forced to be **{14, 19}**.
Picking "the top 2 by retrieval score" could give {19, 22} and silently lose the coreference
column, which would make E7's column half unmeasurable — destroying the thing D-9a was decided to
enable.

*So: measure first, then engineer.* Run `preflight.sh` on the H100 — it already measures exactly
this (three `profile_memory.py` runs at the real lengths, gate at 72 GB) and removes the 16×
extrapolation lever arm. My prediction is that 8K chunked already fits at low `frac_over` and the
four fixes give the headroom to survive a healthy one.

**One bookkeeping point if a fallback is taken anyway.** Spec §18's pre-committed rule is
*"> 1.2 s/sequence at 8K ⇒ two patched layers or 4K training"* — it is registered against **step
time**, and `time_step.py` implements it that way (`rule_triggered = per_seq > 1.2 and L >= 8192`).
Invoking its lever set for a **memory** constraint is defensible, but it should be recorded as
such, in the same way §50 recorded the S0 refinement: the rule was registered against a different
trigger, and a reader who checks will notice.
