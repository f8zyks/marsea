# MarSea on Nebius — runbook for S0 and S2

*2026-09-11. Written against spec v4.9 (D-9a split `T_j`, D-31 sink exclusion, D-27 grids), the training procedure, and
the read-through of 2026-09-11. Every command below exists in `scripts/`; nothing here is a plan for code still to be
written.*

The programme is staged so that the two cheap stages can invalidate the expensive one before it is entered: **S0 is the
gate on everything** (spec §14), and the preflight is the gate on S0.

---

## 0. Provision (record §44, D-26)

| | |
|---|---|
| stage 1 | 1 × H100 80 GB, ~4 days: image, tests, data generation, detector, **S0**, S1 (end-to-end at 4K, step time at 8K) |
| stage 2 | 1 × 8 × H100 80 GB, ~2 days wall: the S2 training queue, then the evaluation queue |
| region | one European region for both VMs **and** the shared filesystem; request the 8×H100 quota on day one |
| storage | shared filesystem ~500 GB: RULER sets, tokenised QA, detector scores, Phase-A checkpoints, run checkpoints, parquet logs |
| fallback | RunPod Secure Cloud, H100 **SXM** (not PCIe: the patched layers are memory-bound) |

```bash
git clone <this repo> marsea && cd marsea
bash scripts/setup_nebius.sh          # venv, pinned wheels, RULER + the --gold_depth patch, essays, backbone, HotpotQA
# MuSiQue is a manual download (Google Drive zip): put musique_ans_v1.0_{train,dev}.jsonl in data/musique/
bash scripts/gen_data.sh              # every RULER grid, CPU-bound, ~2-4 h
```

`gen_data.sh` generates the **training pool first** and attempts every grid even if one fails, re-reporting the
failures at the end with a non-zero exit. It used to run under `set -e` with the training pool last, so a single
infeasible config aborted the script before S2 had anything to train on — and the feasibility guard did reject a legal
config (E3 at n = 128). The guard is now measured rather than guessed (`ruler.needle_budget`), and `patch_ruler.py`
makes RULER's own retry loop raise instead of spinning forever when the needles genuinely do not fit.

---

## 1. Preflight — before anything is trained

```bash
bash scripts/preflight.sh 2>&1 | tee runs/preflight.log
```

Seven checks, ~40 minutes on one H100. **Every one is a gate**: the script exits non-zero the moment one fails, and the
queue behind it must not start. (Until the d5bd980 round the memory line printed "peak < 72 GB" and compared nothing,
`profile_memory` swallowed OOM and exited 0, `time_step` printed the step-time rule and exited 0, and `verify_all` had
no verdict at all.)

1. **Environment and contract** — versions, and `assert_transformers_contract()` on `eager_attention_forward`'s signature
   (the patch point; it has changed twice upstream).
2. **Acceptance tests T0–T16** with `MARSEA_DEBUG=1`, so every normaliser call in every test also asserts INV-1…INV-9
   and SAN-1. **Gate: all green.**
3. **The numerical harness** (`scripts/verify_all.py`). **Gate: 0 violations.**
4. **Memory at the real lengths** — the S2 configuration decision, below. Five profiles, each gated at 72 GB:
   chunked training (2/4 layers, 2K–8K); **dense training at 8K**, whose verdict `run_s2.sh` reads before it launches
   the three dense E9 arms (it used to be "NOT a gate", while the uniform-quota ablation trained there on an
   extrapolated 27 GB); chunked teacher-forced evaluation at 8K/16K with **both** D-9a heads kept (`--sites 2`); and
   dense teacher-forced evaluation at 8K, which five of the seven eval job types run; and a dense arm on MarSea's
   paired relation at 8K/16K with **two models resident** (`--paired`), which is what B0/B1/B2/B4 do for most of the
   eval queue. The dense-8K *training* probe is the one gate whose failure is **recorded, not fatal**: under `set -e`
   it used to abort preflight before 4b, 5 and 6; `run_s2.sh` enforces its verdict at startup instead.
   **4b. The unit cap on this device** (`check_unit_cap.py`): the cap's binding decision is taken from the *relation's*
   mass excess, `Σ_{E_i.}(Atil − A_sm) > 1e-6` in fp64, and never from the softmax's fp32 row total — so it is
   independent of the softmax kernel by construction. This step proves that on the target device (razor-edge rows
   through the real dense, chunked and decode paths, forced empty and live at ρ = 0.05 and 0.5: no row bound without a
   relation, no binding row gained mass or lifted a zero, binding rows landed on their softmax mass, the cap's own θ > 0
   on every binding row on all three paths) and records the kernel's own row-sum error, the observed
   `min |excess − CAP_TOL|` margin and the dense/chunked excess difference. It also **bisects the θ-guard's window** —
   the fp32 prefix scan's error, which is a device property too (1.2e-7 on the dev CPU, 1.1e-6 at 8K / 2.0e-6 at 16K on
   the dev GPU, i.e. above `CAP_TOL`) — and gates it against the tolerance INV-3 allows a row (`8·log2(n_k)·ε`, 1.2e-5 at
   8K); a row inside the window keeps less than the window above its softmax mass. Passes on this box's CPU (row sums
   off by 1.2e-4 at 16K) and GPU alike. Quote `theta_guard_window` from `runs/unit_cap_device.json` with the kernel error.
   **If this gate fails** (`window > tol_cap`, exit 3, the message prints the number): set
   `MARSEA_TOL_CAP=<4 × the measured window>` in the environment of preflight *and* of `run_s2.sh` / `run_evalsuite.sh`
   / `run_s0.sh`, re-run preflight, and quote the value in App. F's device statistics — `unit_cap_record` writes it into
   every README and eval table. The `8·log2(n_k)·ε` default is not a derived law (a sequential fp32 scan can reach
   `n_k·ε/4`); it is a number the measured window has to clear, so overriding it by measurement keeps the property
   *checked* rather than assumed. At the dev GPU's 2.0e-6 window the override would be 8e-6, still under the default.
   The override is itself bounded, or refused: it is parsed once at startup (a typo names the variable and stops the
   run before a GPU is touched), it may not exceed `L·ε/4` (no fp32 scan errs by more — `MARSEA_TOL_CAP=1` used to make
   INV-3a admit any row silently), and 4b refuses it when it is more than 8× the window measured *on this device* (a
   stale value after a reschedule onto a different GPU). `unit_cap_record` writes the value and the ceiling in force.
5. **Step time at 8K** — the pre-committed rule of record §44: **> 1.2 s per sequence ⇒ two patched layers or train at 4K.**
   `time_step.py` now *fails* when the rule triggers (`--no_gate` is the explicit opt-out). The dense path — the three
   dense E9 arms and five of seven eval job types — is timed too (`runs/step_time_dense.json`, recorded, not gated).
   If the dense probe or the paired probe fails, preflight still exits 0 and lists what it recorded (it exited 1 when
   only the dense gate failed — review 9bacef9 C-1); a non-zero exit is a real gate failure.
6. **The detector** — regenerated unless the check *positively* confirms **this** spec version's stamp (`spec_version`, `git`, `created`,
   `backbone`), the old file being moved aside as `detector.json.superseded`. It then fails the preflight if no head
   clears the threshold, which is App. H's fallback case rather than a run.

### What the memory measurement decides

**Re-measured 2026-09-12 after the efficiency fixes below; the earlier "8K does not fit" no longer holds.** Fitted as
`activations = b·T + q·T² + (n−1)·r·T²` by differencing 1/2/4 patched layers (`runs/memory_model.json`), with
`--head_block 2` and the relation calibrated to ρ₀ — the configuration `run_s2.sh` actually runs.

| | before the fixes | after (head-blocked) |
|---|---|---|
| first patched layer at 8K | 124 GB (dense) / 94 (chunked) | **10 GB / 12 GB** |
| each additional patched layer at 8K | 22 GB / 32 GB | **~0** — checkpointing bounds the peak to one layer at a time |
| backbone (linear in T) | 0.9 GB / 1K tokens | 2.1 GB / 1K tokens |

Predicted training peak, batch 1, four patched layers, `--head_block 2`:

| | 4K | 8K | 16K |
|---|---|---|---|
| chunked | 14 GB | **32 GB** | 83 GB |
| dense | 13 GB | **27 GB** | 63 GB |

**So 8K training fits an 80 GB H100 with room, and neither pre-committed fallback is needed.** Two caveats worth
keeping: the quadratic term is now small enough that extrapolating it from T ≤ 1536 carries proportionally more
uncertainty, and 16K sits near the gate. `preflight.sh` measures both at the real lengths before the queue starts.

**Why it changed — four fixes, none of which touches the mechanism.**

0. **The sparse correction is accumulated in blocks.** `O += (A − a1)·V` was built as one `[nnz, d]` tensor — 128
   floats per relation entry, so ~10 GB at 8K and 5 % coverage, the same shape of mistake as (1) and on the path S2
   trains. Found in the d5bd980 round. Pass 2's inner loop also cost the size of the whole problem per iteration;
   measured after that fix at T = 2048 with 24.8 % of rows rebuilt, the peak is 2.40 GB at `row_block` 1024 and
   1.95 GB at 256 — flat in the knob.
1. **Pass-2 densification no longer gathers.** It rebuilt the cap-binding rows by gathering `k_rep_full[b_r, h_r]` and
   `v_rep_full[b_r, h_r]` — `[n_over, n_k, d]` each, i.e. `d = 128` times the score tensor beside them — so its cost
   scaled with `frac_over`: ~8 GB at 8K when 1 % of rows bind and **~82 GB at the 10 % the training procedure §11 calls
   healthy**, arriving first at 16K where E3 lives. Rows are now grouped by `(b, h)` (plain matmuls against `[n_k, d]`)
   and processed in bounded blocks; measured, the peak is now flat in `frac_over` (1.99 GB at both 21.5 % and 25.2 %).
2. **One sort, not two.** `sorted_prefix_stat` was called on the identical input twice per fan-out solve — once inside
   `SparsemaxMasked.forward` and again for the diagnostics. The statistics are threaded out of the Function instead.
3. **The differentiable `lse` loop is checkpointed.** Each chunk's `exp()` saved its output, retaining a full dense
   `[B, H, T, T]` (3.2 GB at 8K, 12.9 GB at 16K) — the chunked path's most avoidable quadratic.
4. **`want_dense_diag` no longer fires on logging steps.** It concatenated seven full-size `[1, H, T, T]` copies every
   50th step and threw them away.

**Head blocking** (`--head_block k`, default 2 in `run_s2.sh`) is what converts those into a peak reduction: every
program is per Q-head (spec §6.2), so the heads split exactly, on the dense **and** the chunked path, each block
separately checkpointed so only one block's `[B, k, T, T]` set is live. "Exactly" to fp64 rounding, not bitwise:
blocking changes the GEMM shapes and, for the per-head (E9) parameters, the gradient reduction order — measured at
4e-15 on values and 4e-11 on gradients in fp64, so expect ~1e-7 relative drift in fp32.

**Why ordinary distributed training still would not have helped.** The budget is for **one sequence** (B = 1: no
padding, no packing). DDP splits a batch that does not exist here; FSDP/ZeRO shard parameters and optimiser state, which
total **2.9 GB measured** — nothing against an activation bill. Only sequence parallelism attacks the right term, and
MarSea is the awkward case: the fan-out program is global over the query axis (`c̄_j = Σ_i A^sm_ij`, and the sparsemax
runs over the whole column `E_.j`), so every column solve would need its column gathered across devices. Spec §18 rules
it out in any case: "eight independent single-GPU processes; no distributed training" — the 8 GPUs run the 17 arms.

**The unit cap decides on the relation, not on the row total — and projects onto the row's own softmax mass.** In exact arithmetic `Σ_j Atil_ij = Σ_j A_sm_ij + excess_i`
with `excess_i = Σ_{j∈E_i.}(Atil_ij − A_sm_ij)` and `Σ_j A_sm_ij = 1`, so "the cap binds" ⟺ `excess_i > 0`. The code
tests exactly that (`normalizer.relation_excess`, fp64-accumulated over the relation's entries, `CAP_TOL = 1e-6`).
Three review rounds had it testing `Σ_j Atil > 1 + slack` instead, and each slack constant was right on one kernel:
an fp32 softmax row's total is off one by an amount that is a property of the *kernel* — ~1e-6 on CUDA, `n_k·eps/(2L)`
on a lane-wise x86 kernel (6e-5 at 8K, 1.2e-4 at 16K on the dev box; 1e-3 for a scalar kernel) on a razor-edge row.
The excess never touches a row total, and with the entries cast to fp64 before subtracting it is *exact* for the fp32
tensors. A binding row is projected onto `Σ_j A_sm_ij` summed in fp64 (one, in exact arithmetic) rather than the literal
1, and the projection refuses to bind when its own arithmetic finds nothing above the target — so a cap never adds mass
or lifts a Stage-1 zero on any kernel (review 7814665 C found it could, on a CPU razor row whose fp32 total sat below
one). An empty relation has excess exactly 0.0, so T0 and sanity 5(a) hold bitwise.
`torch.softmax` stays on the dense/decode paths (T0's identity with the stock backbone), and its row sums are what
the device makes them — reported by SAN-1 with a kernel-agnostic bound, `n_k·eps/2`, read by nothing else. Memory
figures predating e982f83's `nnz_block` of 2²⁶/d need re-measuring, which step 4 does.

**If a fallback is ever taken anyway**, two things to record. Spec §18's rule is registered against **step time**
(`> 1.2 s/sequence at 8K`), so invoking its lever set for a *memory* constraint should be written down as such. And two
patched layers is not free: D-9a puts the column test at the head S0 finds and the row test at the detector's
`(l*, h*)`, so both must be in `PATCHED_LAYERS` — "the top two by retrieval score" could silently drop the coreference
column and make E7's column half unmeasurable.

---

## 2. The detector, then S0 — the gate on everything

```bash
bash scripts/run_s2.sh 8               # Phase A for seeds 0-2, then STOPS (exit 3): S0 has not been run
bash scripts/run_s0.sh                 # sweeps the Phase-A weights, writes the licence into runs/detector.json
bash scripts/run_s2.sh 8               # again: Phase A is verified and reused, Phase B starts
```

S0 needs the Phase-A weights, and only `run_s2.sh` produces them — so the documented order used to exit 1 at S0, and the
natural recovery (run the training script) trained the whole grid with S0 never run, after which every column was
reported as measurable by default. Now `run_s2.sh` stops after Phase A until the licence exists, and `run_eval.py`
treats a detector without one as "nothing measurable" (review e982f83 I).

S0 answers two questions and writes both into `runs/detector.json`:

**(a) Does any `T_j` construction route?** Every (layer, head), span-tolerant matching, the attention sink excluded from
the row maximum, licensed at **≥ 80 % of examples** — and licensed **per kind**, because D-9a split `T_j`:

* `coref` — the key phrase's first mention, targets = the later mentions, `m_j = m + 1` on MV-NIAH;
* `value` — each needle value's first token, target = the row that emits it.

The detector selects *retrieval* heads (answer position → value token), which is construction `value`. Coreference is the
*induction* pattern and lives in different, usually earlier layers with a one-position offset, so the sweep looks
everywhere rather than only at `(l*, h*)`. On the development box (base weights, 8K, 40 examples) `value` was licensed at
60/336 heads and `coref` at 21, best `(14, 5)` and `(14, 3)` — but that was the *base* model; **S0 must be read on the
Phase-A weights**, which is what `run_s0.sh` does.

**(b) Is there a replication pair?** `M(a) > m_j` at `a ∈ {0.5, 1.0}` on the `(l*, h*)` score columns — the condition
under which Cor. `capacity` has bite on real data.

Since the d5bd980 round the licence also records the **site** it was granted at, and `run_eval.py` passes that back in
as `--coref_site`: a pass then keeps two `(layer, head)` sites and scores each `T_j` kind where it lives. Before this,
"licensed anywhere" licensed a measurement that was still taken at `(l*, h*)`, so the gate failed open. Each column in
the rows carries its `site`, and the table reports `site_by_kind` beside `interval_hit_rate_by_kind`.

`run_eval.py` passes **both** licensed sites back in (`--coref_site`, `--value_site`), and a site may name a head of
l\*'s own layer: the pass then keeps two heads of that layer. Every column in the rows carries the `(layer, head)` it
was measured at, and `site_by_kind` appears in the table.

Two consequences for `PATCHED_LAYERS`. The detector now applies its own threshold as a **rejection** (Sec. 6.7's "a head
retrieves if score > 0.1" was stored and printed but never applied) and excludes the attention sink from the row
argmax — so **`runs/detector.json` must be regenerated on the H100**; the one in the repo predates both. And when the
induction layer joins, it now **replaces the weakest retrieval layer** rather than making five, because every consumer —
the memory model, `run_s2.sh`'s budget, the E9 layer control — sizes for four.

**If no construction is licensed**, spec §12.3 applies and nothing is rescued by hand: E7's column half is reported as
*not measurable on this backbone*, the row side carries E7, and the interval-hit rate is reported on the synthetic
columns and the harness. `run_eval.py` reads that licence out of `detector.json` and drops exactly the unlicensed kinds,
so the tables cannot silently carry numbers the gate refused.

---

## 3. S1 — one end-to-end run before committing the grid

```bash
.venv/bin/python scripts/run_train.py --arm marsea --seed 0 --L 4096 --total_steps 700 --phase_a_steps 500 \
    --mode chunked --detector runs/detector.json --ruler_train "data/ruler/TRAIN_*" \
    --musique data/musique/musique_ans_v1.0_train.jsonl --hotpot_n 20000 \
    --eval_ruler "data/ruler/QUICK_*/validation.jsonl" --out runs/s1
```

Read the first hour against the training procedure §11 ("what a healthy run looks like"):

* Phase-B init: `gap_5a` **exactly 0** (E forced empty reproduces the Phase-A path) and `rel_gap_5b < 5 %`; coverage
  calibrated to 0.050 per layer. The run **stops itself** if either fails.
* steps 500–600: loss within 5 % of Phase A's final; coverage drifting from 5 % (either way is fine, 0 % is not);
  `τ_i` median in 0.8–1.3; `k*` mostly 1–3; cap binding on ≲ 10 % of rows.
* the E8 block appears in `train_log.jsonl` every 50 steps with **per-head** ρ (never pooled), the `|E_.j|`/`|E_i.|`
  histograms, `var_j τ_j`, the `k*` histogram, the regime quantiles, the zero-mass-row fraction, per-group gradient
  norms, non-finite counts and `b0`.

---

## 4. S2 — the training grid

```bash
bash scripts/run_s2.sh 8               # MODE=chunked L=8192 STEPS=2500 EVAL_EVERY=250 by default
```

Before any GPU is touched it refuses to start unless every training source is present: the RULER pool, the MuSiQue
training file (a manual download nothing else verifies) and HotpotQA — `build_training_sources` raises too, where it
used to skip MuSiQue silently and print-and-continue on HotpotQA. It also refuses at startup — not after ~36 h — if
dense 8K training did not pass preflight's memory gate, unless `SKIP_DENSE_E9=1` drops the uniform-quota and K_ret arms
knowingly. The quick evaluation runs every 250 steps, the training procedure's cadence (measured, the saving at 500 was
under 1.5 % of the grid). Every checkpoint carries a detector stamp (layers, `(l*, h*)`), which evaluation checks, and
Phase B refuses a Phase-A file saved under a different `PATCHED_LAYERS` — which S0 can legitimately cause. Then either
delete `runs/phaseA_seed*.pt` and re-run (~6 GPU-h), or set `ALLOW_PHASE_A_LAYER_CHANGE=1`, which is recorded in the run
README. Re-running preflight after S0 will not overwrite a detector carrying a licence (`PREFLIGHT_FORCE_DETECTOR=1`).

Phase A runs **once per seed** (`--phase_a_only`, the full 500 steps, verified on load) and **every** arm — including
every E9 ablation — forks that one file through `--phase_a_ckpt`. Phase A runs the **patched** SoftmaxNorm layers by
default, i.e. exactly the forward Phase B forks into; `--phase_a_unpatched` is available and measures and records the
kernel delta if the memory ever demands it.

D-24's reduced programme: 3 seeds on MarSea/B0/B2/B3, 1 seed on B1/B4, and three E9 ablations. `EXTENDED_E9=1` adds the
paper's further field-argument arms (no-ν, global τ_j, per-head heads, rank 4/64, deployed hierarchical `K_ret`); those
are S4, not D-24, so they are off by default.

Failure handling is pre-committed and automatic: a NaN abandons the step, restores the last checkpoint with the identical
data order, and logs `(step, micro, layer)`; a second at the same step halves the head learning rate once; a third stops
that arm × seed and reports it as a failure. Restarts are idempotent — the data cursor, RNG states, optimiser and
scheduler all ride in the checkpoint.

---

## 5. Evaluation

```bash
bash scripts/run_evalsuite.sh 8
```

Per arm × seed × job, resumable (a finished cell leaves a `.done` marker), at most one job per GPU, and it refuses to
start without an S0 licence, without its evaluation data, without the paired MarSea checkpoints the dense arms score
against, or if preflight's two-model `--paired` memory probe failed (`SKIP_PAIRED_GATE=1` overrides). An arm with no
checkpoint is listed in `runs/eval/SKIPPED_ARMS.txt` and the queue exits 2 (`ALLOW_MISSING_ARMS=1` accepts an
incomplete grid) instead of dropping it silently. Every job writes `runs/eval/<tag>.parquet` and `<tag>_table.json` under its own tag
(`marsea_s1_E3`, `e9_per_head_E2`, ...) with a provenance block; nothing shares a file. E3 runs at 16K on the chunked
path with its inert-padding control, **n = 400** spread evenly over the five n-levels (80 per point; E3pad 80);
E3depth — a control — runs for MarSea and B0 at seed 0 only. The dense arms take `--paired_marsea_ckpt` on every
RULER **and** QA job, so their headline precision is on MarSea's relation (`m_j/|E_.j|`, the identity of Prop.
`fullsupportceiling`) beside the full-column number; the relation is rebuilt per example, at every measurement site.
E9 arms are evaluated on the path they trained on. B5 runs last, teacher-forced only (D-29), and its task-level cell
reads "n/a by design".

E6 runs inside the queue when `data/iheval/iheval.jsonl` exists; **no script acquires IHEval**, so without it the queue
prints "E6 NOT RUN" — a decision for the paper, not a silent loss (see `PAPER_EDITS_e982f83.md`).

The tables for the paper:

```bash
.venv/bin/python scripts/collect.py --eval_dir runs/eval    # -> runs/eval/collected.{json,md}
```

pools seeds (mean, std, n) only when their provenance agrees, and builds the E8 block per patched layer and E3's
relation-coverage rate `|E_.j|/n` per stratum.

Every table prints its **pre-committed decision rule** beside the numbers it governs — E2's falsification is a hit rate
*falling* with m (the comparator's precision gap shrinking with m is arithmetic, not failure); E3's is the load-bearing
one, where flat precision on δ_j > 0 columns beside a falling fraction of such columns is the theory's own prediction and
flat precision on *all* columns would mean the relation, not the program, is doing the selection. Both outcomes are
reportable and the code prints the rule so a run's README cannot omit it.

---

## 6. What to watch, and the pre-registered readings

From the E8 block, reported either way, never acted on (training procedure §8.1):

| reading | meaning |
|---|---|
| coverage < 0.5 % on every patched layer | the arm **became B0** |
| `\|E_.j\| = 1` mass > 80 % | it **trained out of Thm. 1** (App. E cost (a)) |
| across-key `var(τ_j)` → 0 | constant temperature |
| `k*` approaching `K_ret` | the hierarchy is truncating — **only meaningful when `K_ret` is deployed**; it is `None` for the exact flat solve and the flag then reads `None` rather than firing |

---

## 7. Cost

Record §44's estimate was reduced programme ≈ 380–500 GPU-h ≈ \$1,500–1,950 at list; full ≈ 700–800 GPU-h ≈
\$3,000–3,500. **The e982f83 review estimated the queue as written at ≈1,000 GPU-h** (evaluation, never measured, was
the larger half), against `tab:compute`'s ≈336. Its two cuts are in — E3depth to MarSea/B0 at seed 0 (≈150 GPU-h),
`--n 400` on the 16K jobs (≈250) — which it puts at ≈610 GPU-h, ~4.5 days on 8×H100; the quick-eval cadence saves a
further ~10 % of training. `tab:compute` still needs the paper-side update. On-demand for S2; preemptible only for E9 re-runs. Stop stage-1 VMs when idle. Daily `rsync` off-platform
(egress is free).

---

## 8. Known-open items

* **Paper edits from the e982f83 review (§J 18–21)** — drafted with line numbers in `PAPER_EDITS_e982f83.md`: the
  unit cap's implemented rule (now kernel-independent; the vacuous `≤ 1 + δ` sentence is replaced), the claims no job produces (Llama cross-family, emission-policy and hard-variant
  ablations, the τᴷ–τQ tying, E7's measured PR trace, efficiency vs coverage, "2–16K"), the E3 padding wording and
  B1/B4's one seed, and the **E6 decision**.

* **Paper-side (Luke's):** §47's offer — remove `prop:twostep`(ii) and the "derived, not imposed" anchor sentences,
  restating (i) as quota conservation with the column total as a remark. The spec demoted these to SAN-1 in v4.4; the
  paper still carries them as a numbered claim.
* **E9 breadth:** D-24 commits to three ablations; the paper's E9 list is longer. `EXTENDED_E9=1` runs the rest, but they
  are not in the committed budget.
* **`hierarchical_topk` is now deployable** (`K_ret` on the normaliser) rather than dead code, so E9's `K_ret` ablation
  has something to ablate — but the deployed default remains the exact flat solve, which is what every number in S2 uses.
