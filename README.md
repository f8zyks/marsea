# MarSea — implementation and experiment code

**MarSea** (Multivariate Relations by Selective Exclusion in Attention) replaces the softmax line of a subset of
attention layers of a borrowed pretrained decoder with a two-stage, two-step normaliser that inherits standard
attention's mass and re-shapes only a learned pairwise *relation* of (query, key) pairs, producing exact zeros.
This repository implements `documents/records/marsea_implementation_spec.md` (v4.6), paper Sec. 2 / 5 / App. E / H
(`documents/marsea_iclr.tex`) and the training loop of `documents/records/marsea_training_procedure.md`.

## Layout

```
marsea/
  primitives.py   Sec. 3   sorted_prefix_stat (Lem. 1), sparsemax (custom autograd), entmax bisection, proj_eq, proj_le
  relation.py     Sec. 4   pairwise relation E = 1[<U k_j, V q_i>/sqrt(r) + b0 > 0] & vis; straight-through gate; b0 calibration
  heads.py        Sec. 5.2 column_stats, row_summary, TauK, TauQ (LN + f(x) = sign log1p|x| on scalar features)
  normalizer.py   Sec. 5   marsea_normalize dense reference; Diagnostics; State (nu hand-off, not detached)
  invariants.py   Sec. 2   INV-1..INV-9 + SAN-1 (asserted on every forward under MARSEA_DEBUG=1)
  baselines.py    Sec. 8   B0 SoftmaxNorm, B1 SoftmaxOneNorm, B2 MESHNorm, B3 KeyOnlyTauNorm, B4 RowEntmaxNorm, B5 MatchedSparsityNorm
  chunked.py      Sec. 6.6 exact chunked implementation (T11), per-chunk gradient checkpointing
  causal.py       Sec. 7/6.4 hierarchical top-K (prop:tournament), frozen-prefix decode cache, seal-at-boundary
  backbone.py     Sec. 6   MarSeaAttention (wraps Qwen2/Llama attention), vis from the 2-D pad mask, MarSeaContext
  detector.py     Sec. 6.7 retrieval-head detector (Wu et al. 2024)
  fidelity.py     Sec. 12  T_j / K_i quantities, intervals (SHARP endpoints), hit rate, P/R, distractor split, E8 block
  data/ruler.py   Sec. 13.1 RULER driver (patched niah.py with --gold_depth), prompt = input + answer_prefix, ground truth
  data/qa.py      MuSiQue / HotpotQA loaders, hop-ordered prompts, T_j by bridge-string coreference (D-9), metrics
  data/iheval.py  IHEval (E6) loader and the accuracy-vs-tier slope
  data/mix.py     the 1:1:1 training mix, B = 1, no padding, no packing, checkpointed cursor
  train.py        Phase A / Phase B (calibration, sanity 5(a)/(b), G1/G2/G3, one cosine, NaN policy)
  evaluate.py     teacher-forced and generation evaluation, S0 routing check, B5 two-pass, aggregation, decision rules
tests/            T0–T16 of spec Sec. 11 as pytest (T0/T16 need the backbone in the HF cache and a GPU)
scripts/          run_detector.py, gen_ruler.py, run_s0.py, run_train.py, run_eval.py, e1_probe.py, fig_pr_curve.py, verify_all.py
third_party/      RULER (git clone) + ruler_gold_depth.patch (applied by scripts/patch_ruler.py)
```

## Setup

On a fresh GPU pod (RunPod or Nebius -- the script's name is historical, nothing in it is provider-specific):

```bash
mkdir -p runs && bash scripts/setup_runpod.sh 2>&1 | tee runs/setup.log
source pod_env.sh          # the Hugging Face cache location it chose (on the network volume); in every shell
```

It checks the machine first (bash >= 5.1, a CUDA >= 12.8 driver, the GPUs' product names, tmux/flock/rsync), builds
`.venv` with the pinned wheels (torch 2.11.0 cu128), puts the backbone and HotpotQA caches under `/workspace/hf_cache`
when the repo lives under `/workspace`, verifies the uploaded data (126 RULER sets with the D0 length provenance, both
MuSiQue files), and clones RULER with the `--gold_depth` patch only when generation is needed (`NEED_RULER=1`, or no
data present).  By hand, the same steps are:

```bash
python3 -m venv .venv && .venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
git clone --depth 1 https://github.com/NVIDIA/RULER.git third_party/RULER && .venv/bin/python scripts/patch_ruler.py
(cd third_party/RULER/scripts/data/synthetic/json && ../../../../../../.venv/bin/python download_paulgraham_essay.py)
```

## Acceptance tests (spec Sec. 11)

```bash
.venv/bin/python -m pytest tests/ -q -s
```

| test | what | file |
|---|---|---|
| T0 | patched SoftmaxNorm == unpatched eager twin (bit-identical logits, greedy tokens); MarSea with E = ∅ likewise; frozen-prefix decode with E = ∅ token-identical | test_t00_backbone.py |
| T1, T2, T13 | Lem. 1 (support = top-k*, G monotone, psi), Thm. 2 both sharp endpoints (also on a scope), gradcheck of sparsemax / proj_le | test_t01_primitives.py |
| T3, T6, T9, T10, T12, T14, T16 | INV-1..9 + SAN-1 on 400 random cases; harness Prop. A loop to 1e-12; zero permanence on the real eq:rowprog path; dense-arm ceiling on both domains; the L-shaped PR trace; tau→0 uniform; target = Atil detector; tau_i = 1 identity | test_t03_twostep.py |
| T7, T8 | psi monotone / exclusion permanent at sealed tau (live head logged); tournament == flat when K ≥ k*, under-retention prefix; decode_step == dense re-solve on the emitted row | test_t07_causal.py |
| T11 | chunked == dense: fp64 1e-9, fp32 relative 1e-5, on every tensor and diagnostic; gradients agree | test_t11_chunked.py |
| T15 | NaN sweep (causal square, right-padded, empty columns, zero-quota rows), gradient into every module, nu gradient across layers, FD of tau_j / tau_i / logits | test_t15_nan.py |
| instrumentation | intervals vs the harness's planted columns (sharp endpoint), P/R regimes, relation-recall misses, distractor split, two-domain dense precision, E8 block, aggregation, task metrics, the 1:1:1 mix's determinism | test_fidelity_data.py |

`MARSEA_DEBUG=1` asserts INV-1..9 and SAN-1 inside every patched forward.

## Build order (spec Sec. 14) and how to run

```bash
# 1-3, 5, 7, 8  primitives, normaliser, relation, chunked, causal, baselines: the tests above
# 4  T0 on Qwen2.5-1.5B at 2K with a PROVISIONAL layer set: tests/test_t00_backbone.py
# 6  retrieval-head detector -> runs/detector.json (PATCHED_LAYERS, (l*, h*))
.venv/bin/python scripts/run_detector.py --n 200 --L 4096
# 9  RULER generation with the backbone tokenizer; S0 routing check (the GATE)
.venv/bin/python scripts/gen_ruler.py --grid E2      # also E3, E3depth, E4, train, eval_quick
.venv/bin/python scripts/run_s0.py --set "data/ruler/E2_L8192_K8_V4_Q1_d0.5_s0"
# 10 Phase A/B training, one seed at 4K first, then the S2 grid at 8K
.venv/bin/python scripts/run_train.py --arm marsea --seed 0 --L 4096 --total_steps 2500 \
    --ruler_train "data/ruler/TRAIN_*" --musique data/musique/musique_ans_v1.0_train.jsonl --hotpot_n 20000 \
    --eval_ruler data/ruler/QUICK_*/validation.jsonl
.venv/bin/python scripts/run_eval.py --arm marsea --ckpt runs/marsea_seed0/final.pt --set "data/ruler/E2_*_s0" --experiment E2
```

Arms: `marsea`, `B0`, `B1`, `B2`, `B3`, `B4` train under the identical recipe; `B5` is evaluation-only
(`run_eval.py --arm B5 --b5_marsea_ckpt ...`, teacher-forced only, D-29).  E9 knobs are `MarSeaNormalizer`
kwargs passed as `--arm_kwargs` JSON: `tau_i_pinned`, `quota_mode="uniform"`, `key_only_relation`, `tau_j_global`,
`no_nu`, `r`, `gate="hard_concrete"`, `per_head=H`.

## Code review 2026-09-09 — what changed

All A/B/C items of `code_review_2026-09-09.md` are fixed; the D-31 rule is implemented:

* **D-31 (spec Sec. 4.1, paper Sec. 2.3).** Position 0 -- the attention sink of a BOS-less decoder -- is never in a
  relation: `RelationHead.forward` masks the logits of the sink column and the sink row (`E[:, 0] = E[0, :] = False`,
  chunk- and cache-aware through `key_offset` / `n_k_total`); the b0 calibration excludes those pairs; E8 logs
  `frac_rows_zero_mass`.  Tests: `test_d31_sink_excluded`.
* **A1/A2** `--phase_a_only` for the S2 pre-job; the dry-run guard never touches Phase A; an incomplete Phase A is never
  saved and a loaded one is verified (step == 500, phase "A"); `--phase_a_ckpt` makes every E9 arm fork the shared file.
* **A3** B4/B5 (no Stage-1 tensors) are scored by `_stageless_attention_level` on the trace / paired relation.
* **A4/B12** `keep_dense_layers` is honoured on the chunked path; the interval statistics read fp32 scores recomputed from
  q, k at l* (never the bf16 matmul, never attention values -- a missing score tensor now raises).
* **A5/A6** The hit is defined only where Thm. 2 speaks: `m_j >= 1`, `|E_.j| >= 2`, `delta_j > 0`, and not the degenerate
  `[0, inf)` interval; excluded columns are counted by reason (`hit_excluded_counts`); `frac_columns_delta_pos` stays separate.
* **A7/A8** `--paired_marsea_ckpt` builds MarSea's `E[l*, h*]` per example for the dense arms' relation-domain precision;
  records carry `n = num_needle_k` and `depth` from the set's `config.json` (`load_set`).
* **A9** The gate is the sweep (`run_s0.py` -> `run_s0_sweep.py`): per-EXAMPLE fractions, licence on the full `T_j`
  (`coref_full`), the replication statistic `M(a)` at (l*, h*) for `a in {0.5, 1.0}`, and a `column_measurable` flag in
  `detector.json` that `run_eval.py` honours (column tables read "not measurable" when S0 failed).
* **A10** `proj_le` uses the right-derivative at 0 (`where(v >= 0, v, 0)`), so the ST signal of site (2) reaches site (3)
  on cap-slack rows; dense and chunked gradients agree.
* **B1-B4, B8, B13** `last.pt` is written right after the Phase-B init; step checkpoints sort numerically; the README is
  appended on resume; sanity 5(a) (Phase-A path == forced-empty to 1e-4) and 5(b) are both asserted, B2 gets a record;
  per-group gradient norms, non-finite counts and b0 are logged every 50 steps; E8 has per-head `rho`, the
  `tau_i Rtil_i / cbar_i` quantiles and the membership-vs-delta correlation; per-head E9 biases land in G3; the
  hard-concrete temperature anneals 1.0 -> 0.2 over Phase B; `DECISION_RULES` carries E8/E9.
* **B5/B6/B7/B9** decode uses the unit cap's `CAP_TOL`; the chunked key-only path scales by `sqrt(r)`; the fp64 invariant
  tolerance is chosen before the casts; `coverage_residual` uses the real `n_q`; both recall readings and the `m_j = 0`
  count are aggregated.
* **B11** closed by the split: `m_j = m + 1` on the coreference column, answer positions on the value columns.
* **B14** the patched generator draws the query before placement, so `gold_depth=None` output is not bit-identical to
  stock RULER at the same seed; every set is generated by the patched generator.
* **C** `past_key_value` (4.53-4.55) accepted as well as `past_key_values`; `torch_dtype` fallback in `load_backbone`;
  Phase-B calibration runs on the dense path whatever `--mode` is.

## Read-through 2026-09-11 — what changed

`marsea_readthrough_2026-09-11.md`'s items F-1..F-8 and the Sec. 3 coverage gaps.  `RUNBOOK_nebius.md` is the run plan.

* **F-1 (split `T_j`)** was already implemented (D-9a): the coreference column keeps the later mentions (`m_j = m + 1` on
  MV-NIAH) and the answer positions attach to the value columns; `T_j_kind` labels each.  `tests/test_tj_split.py`.
* **F-2 (E8 was empty in every log).** The model's forward pre-hook clears `ctx.diags` on every forward, so micro-batches
  1..15 wiped what micro 0 collected and `e8_summary` saw `{}` -- E8 as an experiment, and the three pre-registered
  degeneracy readings with it.  `train_step` now snapshots the payload after micro 0; verified end to end in a real run.
* **F-3 (chunked ignored `quota_mode="uniform"`).** The uniform quota is a reduction over all key chunks, which the
  single-pass structure cannot do, so the chunked path would have reported the INHERITED quota under the uniform label.
  It now asserts, and that E9 arm runs dense (see the runbook for its length and control).
* **F-4 (HotpotQA joint metrics).** Predicted passage slots were compared against `range(len(gold_slots))`, so the joint
  cell read 0 for every arm even on a perfect prediction.  Both sides are now prompt-order slots; measured on real data,
  support F1 goes 0.00 -> 0.22/0.40/0.40 on the first three dev examples.
* **F-5 (`replication_M`).** Counted the attention sink toward `M(a)`, biasing the S0 replication gate permissively --
  the one direction that matters, since `M > m_j` is what licenses Cor. capacity on real data.
* **F-6 (two b0 bisections).** `phase_b_init` re-implemented the bisection without the D-31 sink exclusion, so realised
  coverage sat below rho_0 and the README's number was not the run's.  Both callers now share `candidate_logits` and
  `bisect_b0`; a smoke run calibrates to 0.050000.
* **F-7** `SPEC_VERSION` 4.6 -> 4.9 (it is stamped into every checkpoint and run README).
* **F-8 (retained diagnostics).** `teacher_forced_pass` kept nine `[1, H, T, T]` fp32 tensors (~29 GB per example at 8K)
  and B5's pass 1 kept them for all four patched layers while reading only `E`/`supp_rel`.  Both now sub-select: measured
  373 MB -> 31 MB retained at T = 1024 (~24 GB -> 2 GB at 8K) with **identical reported quantities**.
* **Coverage gaps:** T4, T5 and T6's remaining halves now run on the tensor path (`tests/test_t04_t05_scope.py`) -- T5 is
  the one acceptance test that states the paper's headline (degree = m_j, P_j = 1 as `|E_.j|` grows) in code;
  `hierarchical_topk` is deployable via `K_ret` rather than dead, and the E8 truncation flag reads `None` instead of
  firing when no hierarchy ran; E6 has a driver (`scripts/run_e6.py`); Phase A now runs the **patched** SoftmaxNorm path
  by default (the unpatched SDPA alternative measures and records its kernel delta rather than asserting it).

**Found while profiling, not in the read-through:** the chunked path is called straight from the patched layer and was
**not** wrapped in `autocast(enabled=False)`, so under training or evaluation autocast every matmul and einsum in it ran
in bf16 -- the dtype that loses the exact zeros the mechanism is (H1).  It is guarded now, in the entry point and again
inside the per-chunk helper (`torch.utils.checkpoint` restores the ambient autocast state when it recomputes), with a
test that pins the result to the path's own nondeterminism.

## Independent memory review 2026-09-12 — what changed

`marsea_memory_budget_2026-09-12.md` verified the three profiling fixes, found two bugs and four memory items, and
recomputed the budget independently.  All are addressed.

**Two bugs, both real.**
* `build_model` raised `NameError: akw` whenever `phase_a_patched=False`, and `patch_model` ran unconditionally so the
  unpatched-Phase-A branch was defeated anyway -- an indentation error I introduced when wiring `head_block`.  Both fixed;
  the branch is now exercised.
* **Head blocking was wired to nothing**: absent from `run_s2.sh`, and `marsea_chunked_attention` never read
  `norm.head_block`, so the path S2 runs could not head-block at all.  It is now implemented on the chunked path
  (bit-identical in fp64, `test_chunked_head_blocking_is_identical`) and on by default in the queue (`HEAD_BLOCK=2`).

**Four memory fixes** (details and the before/after table in `RUNBOOK_nebius.md`):
1. **pass-2 densification** gathered `[n_over, n_k, d]` key and value tensors, so its cost scaled with the cap-binding
   rate: ~8 GB at 8K at 1 % and **~82 GB at the 10 % the training procedure calls healthy**.  Rows are now grouped by
   `(b, h)` into plain matmuls and processed in bounded blocks; the peak is now flat in `frac_over` (measured).
2. the duplicate `sorted_prefix_stat` -- called on the identical input twice per fan-out solve -- is gone, threaded out
   of `SparsemaxMasked` instead;
3. the differentiable `lse` loop is checkpointed (it retained a full dense `[B,H,T,T]`);
4. `want_dense_diag` no longer fires on logging steps.

**Result:** the per-patched-layer term falls from 124 GB to ~10 GB at 8K, the marginal per additional layer to ~0
(checkpointing bounds the peak to one layer at a time, as the review pointed out), and **8K training with four patched
layers now fits an 80 GB H100 at ~27-32 GB** -- so neither pre-committed fallback (4K, or two patched layers) is needed.
`runs/memory_model.json` carries the refitted model; `scripts/preflight.sh` re-measures at the real lengths.

**One more hole the end-to-end run exposed, of the same class as F-2:** the chunked path never materialises `E`, so the
E8 summary could not run on it -- meaning **E8 would have been empty for the whole of S2, which trains chunked**.  It is
now computed from the sparse relation (`e8_summary_from_sparse`), merged across head blocks with the head index shifted
out of block-local space, and `test_e8_agrees_between_the_dense_and_chunked_paths` pins the two summaries together.

**Three test gaps closed**, all of which the review identified: the chunked *backward* under autocast (the guard's whole
purpose, previously unexercised -- and it turned up that calling `.backward()` *inside* the autocast block degrades the
programs' gradients to ~3e-2 relative, so `train_step` now asserts the safe ordering); direct fp64 coverage of INV-3's
tolerance; and the row-block invariance of pass 2.  INV-3's tolerance is now the cap's slack **plus** the summation
error rather than the max of the two, and INV-5 imports `CAP_TOL` instead of hard-coding it.

## Independent review at d5bd980 (`marsea_review_d5bd980.md`) — what changed

Three readers, seven code paths, provenance verified against the tarball's checksums. It confirmed the eight claimed
fixes (one only half-applied), found two regressions the fixes themselves introduced, and re-listed the data-pipeline
items. Everything in its §H fix order is addressed.

**The blocker.** `ruler.py`'s feasibility guard rejected E3 at n = 128 — 512 needles, ~12.1K tokens of a 16K budget,
which spec §13.1 sizes as legal — and under `set -e` that `ValueError` aborted `gen_data.sh` **before `--grid train`
ran**, so the training pool never generated and S2 could not start. The constant was wrong in both directions: a needle
sentence measures 22 tokens, not 16, and niah.py's essay haystack is a list of *words*, so its floor is ~613 tokens
rather than a quarter of `L`. `needle_budget()` replaces the heuristic, `patch_ruler.py` (v2) makes RULER's retry loop
raise instead of spinning forever, and `gen_data.sh` generates the training pool first and attempts every grid.

**Two regressions the previous round introduced.**
* Stripping `ctx.collect` from the `want_dense_diag` gate put `E` out of reach on the dense head-blocked path, so
  **E8 was skipped with no fallback** — no coverage, no `|E_.j|` or `k*` histogram, and `coverage_live` null in every
  run. `E` now has its own gate, and a case that computes nothing records `e8_missing` rather than staying silent.
* `--head_block` in `run_s2.sh`'s `COMMON` reached the `per_head = 12` E9 arm, where the per-head parameters are
  indexed in global head space: a crash, and behind it pass 2's `relation.V[h]` would have used the **wrong head's**
  `V` on every cap-binding row. `head_offset` is now threaded through both paths, and a test at `H = 6`,
  `per_head = 6`, `head_block ∈ {1,2,3}` pins them identical (4e-15 on values, 4e-11 on gradients in fp64).

**Two mechanisms that were not the mechanism the paper describes.**
* Frozen-prefix decode was **unreachable under `--mode chunked`** — the chunked branch returns before the dense cache
  seeding — so every decode step recomputed `tau_j` live from a one-row `column_stats` instead of the value sealed at
  prefill, in the generation mode of E3. `FrozenPrefixCache.init_from_sparse` builds the same state from the sparse
  store (per-column top-`K_ret` by sorting, never a `[B,H,n_k,n_q]` topk), and a test shows the two caches agree.
* The chunked path silently ignored `K_ret`, and `run_s2.sh` queued the tournament arm chunked — **that arm would have
  measured the exact flat solve it exists to ablate against.** It is refused, and the arm moved to the dense group.

**D-9a was only half done.** `T_j_kind` entered `evaluate.py` as a *tag*: both kinds were measured at the detector's
retrieval `(l*, h*)` while the tables split by kind, so the tables looked split when they were not. A pass can now keep
two `(layer, head)` sites, the coreference column is scored where S0 licensed it, every column records its site, and
the coverage residual is reported per kind. The S0 gate records the site its licence refers to — `usable_anywhere`
previously licensed a measurement that was then taken somewhere else.

**`MARSEA_DEBUG=1` did nothing on the chunked path.** `check_invariants` was called from the dense normaliser only, and
`invariant_report` reads `d.Atil.dtype` first, so it *raised* on a chunked Diagnostics rather than degrading.
`invariant_report_sparse` checks INV-1/3/5/6/7/8/9 and finiteness on the sparse store; a test corrupts `A` and
`A_rowsum` to prove it is not inert.

**Arms that were never evaluated, gates that never gated.** `run_evalsuite.sh` skipped B1, B4 and every E9 arm (E9 is
D-24's load-bearing test, and its checkpoint path could not match the loop); `aggregate_ruler` dropped every E5 task
metric, so that table reported `nan`; `preflight.sh` printed "peak < 72 GB" and compared nothing; `profile_memory.py`
swallowed OOM and exited 0, never measured head blocking, and printed an R² that is 1.0 by construction on two points;
`time_step.py` printed the D-23 rule and exited 0; `verify_all.py` had no verdict at all. All fixed, and
`profile_memory` now differences two layer counts for the marginal per-patched-layer cost instead of charging the
backbone's activations to the patched layers.

**Also found while there** (not in the review): the sparse correction `O += (A − a1)·V` was built as one `[nnz, d]`
tensor — 128 floats per relation entry, ~10 GB at 8K and 5 % coverage, the same shape of mistake as pass 2's old
gathers, on the path S2 trains. It is accumulated in checkpointed blocks. Pass 2's inner loop also cost the size of the
whole problem per iteration; measured after the fix at T = 2048 with 24.8 % of rows rebuilt, the peak is 2.40 GB at
`row_block` 1024 and 1.95 GB at 256 — flat in the knob.

Two items are **runs, not code**: `runs/detector.json` predates the sink/threshold fix and must be regenerated, and
`preflight.sh` must re-measure memory on the H100 before the queue starts.

## Independent review at 30372ae (`marsea_review_30372ae.md`) — what changed

The review ran the suite and the harness itself this round, and verified the previous round's fixes by measurement
(head-offset composition over `Hkv × per_head × key_only × hb` at 8.9e-16; the NaN fixes reproduced on the old tree
and absent on the new; the feasibility guard recomputed by hand). Its §F order is what this round followed.

**B-1 — a coreference site in l\*'s own layer.** `sites` was keyed by layer, so a second head in the same layer
overwrote the first: one head was kept, the coreference columns were measured at h\* and *labelled* h′. Sites are now
(layer, head) pairs and a layer can keep a **list** of heads, through `ctx.keep_dense_head`, the backbone's slicing,
the chunked dense rebuild and `_kept_head`. Verified end to end: a pass with `--coref_site 19 7 --value_site 19 3`
reports `sites [[19,3],[19,7]]` and per-column `site` tags that differ within the one layer.

**B-2 — `truncation_events` disagreed between the two prefills**, so a reported D-28/E8 reading was incomparable
between `--mode dense` and `--mode chunked`. The dense side now counts the same event the decode path counts.

**B-3 — the sparse `frac_rows_zero_mass` was a coin flip in fp32.** It is the D-31 diagnostic, and it reads a
reconstruction whose residual at maximal cancellation — which *is* the zero-mass case — is ~3.6e-7, against an exact
`<= 0`. It now uses `4·eps·sqrt(n_k)`, reports that tolerance, and reports the ambiguous band beside it.

**C — the tolerance was indexed on the wrong axis.** `tol_sum` scaled with `n_q` for every invariant, while INV-3,
INV-7 and SAN-1 sum a **row**. Both are computed now; INV-1 keeps `n_q` (it genuinely sums a column) and INV-9, being
elementwise, takes neither. The sparse report also stops overstating itself: INV-3 is split into a real check and
`INV-3-recon` (a self-consistency check on the reconstruction, which cannot catch a broken cap); everything the
sparse store cannot see is listed in `_not_checked`; an empty report now **raises** instead of reading as a pass; and
the decode path runs the dense report on its single row, where `MARSEA_DEBUG=1` had been inert.

**And the mechanism-side half of C, which only a real run could find.** An end-to-end 2K smoke failed sanity 5(a).
`CAP_TOL` is a fixed 1e-6, but the quantity it covers is the fp error of a row sum of `n_k` terms: a real softmax row
of 1938 fp32 entries sums to `1 + 9.5e-7`. So from about T = 2000 the cap binds on rows **nothing pushed over one
unit**, rescales them (`a1 != Atil`, against INV-5), and the forced-empty path stops reproducing Phase A — the run
refuses to start. Measured across three runs of the same smoke: gap 4.6e-05, then 4.2e-04 (past the gate), then 0.0.
A knife-edge at 2K and well past it at 8K, where every S2 job lives. `cap_slack(n_k, dtype)` replaces the constant at
every site that binds or checks the cap; fp64 keeps the 1e-6 floor, so the bit-exactness tests are untouched. The
smoke now reports `gap_5a` exactly 0.0.

**§D — what the queue actually runs.** Neither queue script could observe a failed job (a bare `wait` returns 0, so
`set -e` never fired and Phase B ran after every arm had crashed): both now wait per PID and exit non-zero. The
`per_head` E9 arm — whose composition with head blocking is this round's fix — is back in the default chunked queue,
`K_ret` in the default dense group, and the dense arms train at 8K so they are comparable with the baselines and with
their own 8K evaluation. E3depth is evaluated. E6's `--out` named a file where `run_e6` creates a directory.
Preflight no longer gates *training* on an extrapolated 16K training peak nothing incurs, and no longer preserves the
stale `detector.json` it is supposed to replace — `DetectorResult` now carries a spec/git/timestamp stamp, its loader
tolerates the keys S0 writes into the same file, and `(l*, h*)` comes from the heads that clear the threshold.
`verify_all` reports **1,058,450 evaluations in 63 checks** — the number the paper quotes — rather than "62 checks",
and its structural rows can now fail the gate.

**§E** was swept in the same pass: `e8_missing` reaches the log; the sparse-correction blocks are 256 MB rather than
16 MB (64× fewer launches at the same peak); the chunked `extra` tensors follow the attributes into `head_sub` space;
E8 is computed before any narrowing; `MENTION_CACHE` is bounded; the word-boundary guard covers non-`\w` needles
(`c++` no longer matches inside `c+++`); `load_module()` is gone; `gen_ruler` continues per **config**, not per grid;
and `profile_memory --sites 2` can measure the two-site evaluation configuration.

## Independent review at e982f83 (`marsea_review_e982f83.md`) — what changed

The review verified the 30372ae fixes by measurement (all 17 per-head tensors at `dense_head=[3,7]` against a full-H
reference to 0.0 in fp64; the `verify_all` denominators against the draft's two tables) and then found that the tree
could not launch: two of its blockers were in the queue scripts, not the mechanism. Its §J order is what this round
followed.

**B-1 — 56 evaluations on 8 GPUs.** The eval queue's barrier was tested once per *arm* while each arm launched seven
jobs; 7 is coprime with 8, so it fired once in the whole loop. It now lives in the launcher, one test per launch, and
the GPU is the job's slot in the current batch. A simulated queue (`tests/test_review_e982f83.py`, a fake python and
`flock`) measures at most 8 concurrent jobs and sees an injected failure reported.

**B-2 — every seed and every E9 arm wrote one file.** `run_eval.py --tag` names every output; seed, arm, set, checkpoint
and git go into every row and a `provenance` block (detector hash, sites, cap slack, input file hashes, S0 status) into
every table; writes are atomic. **B-3** — the paired relation is built per example as it is scored
(`PairedRelation`), not as a ~268 GB host dict, and the 16K jobs run `--n 400`, spread evenly over their configs
(`balanced_take` — `[:n]` would have kept n ∈ {8, 16} and dropped the rest of the sweep). **B-4** — dense 8K training
is profiled and gated in preflight, the verdict is written into the profile, and `run_s2.sh` refuses the dense E9 arms
without it.

**C — B-1's class, three more times.** C-1: the head-blocked merge *overwrote* each narrowed block's fields while
stamping the full head list, so two kept heads in different blocks gave `A.shape[1] == 1`, value columns read from the
coreference head, and an `IndexError` — reachable on the default queue (`HEAD_BLOCK=2`, E3 chunked). The blocks now
concatenate and are permuted into `dense_head` order; a CPU test drives a real (tiny, random) Qwen2 forward with sites
`(2,1)`/`(2,6)` at `hb=2` and fails on e982f83. C-2: the paired relation is keyed by `(layer, head)`. C-3:
`_kept_head` raises for a head the layer did not keep. C-4: the chunked path no longer narrows the sparse store; the
decode cache is seeded from full-H state (`decode_seed`) and the backbone narrows only the *collected* copy.

**D — `cap_slack`, re-derived.** The review confirmed the 30372ae discovery and corrected its justification: the
row-sum error is the softmax kernel's *values* (worst at 2–3 nats, growing like log n_k), not the pairwise reduction,
so `8·eps·sqrt(n_k)` was ~25× looser than the worst row ever measured. It is now
`min(1e-3, max(1e-6, 8·eps·log2 n_k))` — 1.24e-5 at 8K, 1.34e-5 at 16K — with the ceiling making a bf16 slack
structurally impossible. Measuring it on CUDA (`scripts/measure_cap_slack.py`, now a preflight step) found a second
source the review had not: the chunked path's `exp(S − lse)` rounded lse at the *scores'* magnitude, 7.6e-6 off one,
flat in n_k. It is now computed about the row max, `exp((S − m) − r)`: 6e-7. Worst CUDA excess over both kernels is
1.8e-6 at 2K and 7.2e-7 at 16K (headroom 5.9–19×). Two fp32 tests pin both directions: no forced-empty row binds on the
dense, chunked or decode path across an entropy sweep at 8K and 16K (fails under a fixed 1e-6), and a genuine 3e-5
excess at 8K is still capped (fails under the sqrt form). On CUDA with random scores (ρ ∈ {0.01, 0.05, 0.1} at 8K, {0.01, 0.05}
at 16K) the rows whose genuine excess is left uncapped are ≤ 0.006 %, against the review's 0.8 % under the sqrt form. Every run README and eval table records the
slack in force. The smoke's `gap_5a` is 0.0.

**E, F.** `truncation_events` is split into `eviction_events` (cache loss, D-28) and `near_truncation_events`
(k\* ≥ 0.9 K_ret, E8); the dense zero-mass key uses the sparse path's tolerance. The sparse INV-3 was a tautology — it read
`rowsum` exactly where `cap_binds` said `rowsum ≤ 1 + slack` — and now checks `rowsum − Σ_E a1 + Σ_E A` from the store.

**G, H, I — the programme.** `scripts/collect.py` pools seeds (refusing mismatched provenance), builds the E8 block per
layer and E3's `|E_.j|/n`. E3depth runs for MarSea and B0 at seed 0; the quick eval runs every 500 steps. S0 is enforced:
`run_s2.sh` stops after Phase A until S0 has written its licence, and `run_eval.py`/`run_evalsuite.sh` refuse to
report columns without one. Checkpoints carry a detector stamp that evaluation checks; the training mixture fails closed;
preflight regenerates the detector on doubt; E5 dense arms use the paired relation; E9 is evaluated chunked;
`merge_e8_summaries` weights by rows and no longer turns a list into `True`; `git_hash` marks a dirty tree.

**Not code** — the paper edits of §J 18–21 are drafted, with line numbers, in `PAPER_EDITS_e982f83.md`, and E6 needs a
decision (IHEval is not acquired by any script).

## Independent review at e16a843 (`marsea_review_e16a843.md`) — what changed

The review executed the fixes (queues simulated at 1/3/8 GPUs, `collect.py` on synthetic parquets through the real
`aggregate_ruler`, C-1 over 42 head-block configurations) and found one blocking defect, which it attributed to its
own previous recommendation.

**`cap_slack` is too small on the CPU kernel — reproduced, and left as an open decision.** The worst case is a
*razor-edge* row, a spike ~16.6 nats over a flat tail just under half an ulp of the row max, which the previous
random-row search never reached. There `torch.softmax`'s CPU kernel drops the tail in each accumulation lane and errs
by `n_k·eps/(2L)`: on the dev box 6.0e-5 at 8K and 1.2e-4 at 16K, against a slack of 1.24e-5 / 1.34e-5. On CUDA the same
construction errs by ≤ 1.2e-6 (11× headroom at 16K), and the about-max form the chunked path uses errs by ~2e-6 on both
devices. The review's preferred fix, the about-max form on the dense and decode paths too, was implemented and
**reverted**. It makes T0 fail: the patched SoftmaxNorm arm moved 1.37e-2 in relative logits from the stock backbone,
because bf16 amplifies a ~1e-7 change in the weights. T0 and sanity 5(a) rest on that identity. What is in the tree
instead:
`measure_cap_slack.py` searches the razor edge and gates on what the mechanism actually runs (it fails on this CPU,
passes on this GPU — preflight on the H100 decides); a razor-edge unit test that passes on CUDA and is a *strict* xfail
on CPU; and a `cap_slack` docstring that states the problem. `softmax_about_max` now lives in `normalizer.py`.

**Preflight and queue.** Preflight's dense-8K *training* probe records its gate failure instead of aborting the three
unrelated gates after it; `run_s2.sh` enforces that verdict at startup (`SKIP_DENSE_E9=1` to drop those arms). The eval
queue checks its data before launching, adds `e9_reference_E2` (MarSea evaluated chunked, the E9 arms' matched
comparator), and is work-conserving — a finished job frees its GPU at once (`wait -n`), simulated with no same-GPU
overlap. `profile_memory --paired` measures a dense arm with the paired MarSea model resident; the gate's verdict was a
numpy bool that crashed the JSON write. Preflight will not overwrite a detector carrying an S0 licence.

**`collect.py`** carries every metric (the whitelist dropped E7's before/after recall, `support_EM`, all sample sizes and
the per-kind splits), records provenance conflicts and still writes everything else, reads pooled regime triples, and
gives E8 a seed dimension. **Training**: the quick eval is back at every 250 steps; Phase B refuses a Phase-A file saved
under different patched layers unless `--allow_phase_a_layer_change`. **Smaller**: T16 is its own test and also checks
`E` and `τ_j`; `save_rows` is atomic; E8 quantiles survive a float64 default dtype.

## The unit cap made kernel-independent (2026-09-14, after review e16a843)

Three review rounds moved a constant: the unit cap tested `Σ_j Atil_ij > 1 + slack`, and a real fp32 softmax row does
not sum to one. How far off it is is a property of the *kernel*, not of fp32: ~1e-6 on CUDA (pairwise reduction);
`n_k·eps/(2L)` on a lane-wise x86 kernel, where a razor-edge row — one spike, a flat tail just under half an ulp of it —
loses 1/L of its tail from the denominator (6.0e-5 at 8K and 1.2e-4 at 16K on the dev box's CPU); up to `n_k·eps/2 ≈
1e-3` for a scalar kernel. `1e-6`, `8·eps·√n_k` and `8·eps·log₂n_k` were each right on one device. The programme runs on
Nebius machines whose CPUs we do not know, and a constant chosen on one kernel is no answer.

The fix is the criterion, not the constant. In exact arithmetic `Σ_j Atil_ij = Σ_j A_sm_ij + excess_i` with
`excess_i = Σ_{j∈E_i.}(Atil_ij − A_sm_ij)` and `Σ_j A_sm_ij = 1`, so **the cap binds iff `excess_i > 0`** — a quantity
computed from the relation's entries only, which never reads a softmax row total. `normalizer.relation_excess`
accumulates it in fp64 over the relation's entries (nnz-sized; a full fp64 row sum would copy the `[B,H,T,T]` tensor)
and the cap binds above `CAP_TOL = 1e-6`, the original spec constant. The excess's only error is the elementwise fp32
rounding of the relation's entries, bounded by ~1.5·eps·(Σ_E Atil + Σ_E A_sm) ≤ 3.6e-7 at the decision boundary on any
device and measured ≤ 3.0e-7 on CPU and CUDA at 8K/16K — 3× under `CAP_TOL`. An empty relation has excess *exactly*
0.0, so the forced-empty path is bitwise standard attention whatever the kernel does, which is what T0 and sanity
5(a) rest on. Dense, chunked (from the sparse store: `Σ (Atil_s − A_sm_s)` per row) and decode all take the same
decision; `proj_le_masked` takes the caller's `binds` for the unit cap and still tests the sum for the quota cap.

What follows from it. `torch.softmax` stays on the dense and decode paths — the about-max form that would make its row
sums kernel-proof moves T0's logits by 1.4e-2 in bf16 — and its row sums are read by nothing in the mechanism: SAN-1
reports them against the kernel-agnostic bound `n_k·eps/2`, a statement about the device. INV-3 is restated against the
row's own softmax mass, `Σ_j A_ij ≤ max(1, Σ_j A_sm_ij) + 1e-6 + tol`, in fp64; INV-5 recomputes the binding set
independently from the full row in fp64, and INV-5b checks that a binding row was projected onto one unit. The
chunked path's `A_rowsum` no longer clamps a non-binding row's total at one. `scripts/check_unit_cap.py` (preflight
4b) drives razor-edge rows through the real normaliser on the target device — forced empty on all three paths, live at
ρ₀, and a 2e-6 excess that must bind — and records the kernel's row-sum error; it passes on this box's CPU, where every
previous constant failed, and on its GPU. The `cap_slack` function, its ceiling and formula are gone.

## Review at 7814665 (`marsea_review_7814665.md`) — the cap must not add mass

The review confirmed the kernel-independent criterion and found the one thing it had left open: with the binding set
decided by the relation, `proj_le(Atil, 1)` on a row whose fp32 total sits *below* one (a kernel deficit larger than
the relation's excess — a razor-edge row on a lane-wise CPU) solved the equality problem *upward*: θ < 0, Stage-1 zeros
resurrected in `a1`, and since `a1`'s off-relation entries are `A`, the output moved with the device. Reproduced here by
construction (θ = −7.2e-9, 8 zeros lifted), and live on this CPU: 4 of 22 binding razor rows had an fp32 total under one.

Two changes close it (`normalizer.unit_cap`). The target is the row's own softmax mass, `Σ_j A_sm_ij` summed in fp64 —
one in exact arithmetic, and the same fp64 pass that computes the excess (`row_masses`: key-chunked, cast-first, so the
excess of the fp32 tensors is *exact*, ρ-independent in memory and free of the host sync `torch.nonzero` forced). And
`proj_le_masked` refuses to bind when its *own* fp32 arithmetic finds nothing above the target — the projection's
internal prefix sum is a third kernel-dependent total, so neither the fp64 target nor the fp32 `sw − excess` the review
proposed guarantees θ > 0 on their own. Together: a cap never adds mass, never lifts a zero, `a1 ≤ Atil` elementwise,
and a binding row lands on its softmax mass to the projection's accuracy (≤ 1.1e-7 on CPU, ≤ 1.8e-6 on CUDA) — on any
kernel. The target is a constant of the exact program and is detached. INV-3 loses its `max(1, ·)`:
`Σ_j A_ij ≤ Σ_j A^sm_ij + 1e-6 + tol`, in fp64; INV-5b (emitted always) checks that binding rows went *down* onto their
softmax mass; the sparse `INV-5-decision` is renamed `INV-5-plumbing` for what it is; `INV-3-recon` bounds by the fp64
softmax mass. `check_unit_cap.py` asserts all of it on the target device and records the real safety margin
(`min |excess − CAP_TOL|`) and the dense/chunked excess difference (the two softmax formulas differ by up to ~6e-6 at
ρ = 0.5, so the two paths may legitimately decide differently on a row within that of `CAP_TOL`).

Also this round, each verified before it was changed: `collect.py` crashed on E6's `slope` scalar and wrote nothing
(F-1); the Phase-A layer guard fired inside `--phase_a_only`, i.e. on the documented runbook (F-2 — now below the return,
with one message from `run_s2.sh`'s startup gate and `README.json` written before the return); preflight's `--paired`
probe was a hard gate nothing read (F-3 — recorded, enforced by `run_evalsuite.sh`); the eval queue dropped arms with no
checkpoint silently and exited 0 (F-4 — listed, `runs/eval/SKIPPED_ARMS.txt`, exit 2 unless `ALLOW_MISSING_ARMS=1`; the
paired MarSea checkpoints join the data gate); `collect.py` now carries `rho_head` per head, `source`, the
count-weighted `pass2_rebuilt_row_frac`, and positional `per_seed` lists; `wait -n -p` is guarded and bash ≥ 5.1
asserted. Not taken: the reviewer's masked `sum(dtype=float64)` — on CUDA it copies the tensor to fp64 (+1.5 GB at 8K,
head_block 2); the key-chunked form has the same properties at 0.19 GB.

## Review at 9bacef9 (`marsea_review_9bacef9.md`) — the cap is closed; the guard's window measured

The review confirmed the two-part fix on all three paths (99/84/303 binding rows; no row gained mass, no zero lifted)
and made the point that matters for the schedule: **on this kernel the §C bug was latent** — `torch.softmax` here puts
every razor row's mass *above* one, so the constant-1 target never scaled a row up in practice, binding sets are
identical between 7814665 and 9bacef9 and `A` moved ≤ 1.2e-7. The fix is device safety, not a numbers correction;
nothing needs re-running. Every claim below was verified before it was changed.

**The one open question (§D), answered.** The θ ≤ 0 guard has a window: a row whose excess exceeds `CAP_TOL` by less
than the fp32 prefix scan's error can be left uncapped. Bisected here: **1.2e-7 on this CPU** (torch's `cumsum`
accumulates in higher precision — the reviewer's 8.7e-8), but **1.06e-6 at 8K and 1.95e-6 at 16K on this GPU — above
`CAP_TOL`** — and a strictly sequential fp32 scan could reach `n_k·ε/4`. So the *binding set* is device-dependent for
excesses inside the window, while the *row* is not harmed: a row inside it keeps less than the window above its softmax
mass. `check_unit_cap.py` now bisects the window on the target device (`theta_guard_window` in
`runs/unit_cap_device.json`) and gates it against the tolerance INV-3 allows a row (`invariants.cap_tolerance`,
`8·log2(n_k)·ε` = 1.2e-5 at 8K); the sharpness test goes through `unit_cap` (excess rule *and* guard), at
`4·max(CAP_TOL, window)`; the live gate runs at ρ = 0.05 and 0.5; and decode's §C properties are gated with a column quota
already in the cache (with an empty cache `Atil == A_sm` on the relation and the earlier "decode ok" was vacuous, 0
binding rows — now 41). The clean statement is `Σ_j A_ij ≤ Σ_j A^sm_ij + max(CAP_TOL, window)`, both terms reported.

**Two corrections.** B-1: the reviewer's masked `sum(dtype=float64)` was wrong on CUDA (confirmed by them, +1.50 GB),
but the shipped masked chunked form held four fp64 temporaries (0.375 GiB on CUDA at 8K/hb2, 0.50 on CPU), not the one
the docstring claimed. `row_masses` now differences two fp64 sums per chunk — exact because `Atil == A_sm` bitwise off
`E` (both gates are hard forward), never reads `E`, **0.125 GiB and 16 ms against 32**, output bitwise equal on every
case measured; the fp64 cancellation is ~2e-12 at 16K, six orders below `CAP_TOL`. B-2: the "6.1e-6 dense/chunked
disagreement" was `sm_mass` (the kernel row-sum difference: 1.4e-5 at 2K, 6.0e-5 at 8K on this CPU), not the excess,
which differs by **≤ 4e-8** (≤ 1.7e-7 at 16K, ρ = 0.5). Docstrings corrected; the path-agreement window is now
`2·excess_diff + window`, not a literal 1e-5.

**The three launch blockers**, each reproduced first: C-1 `preflight.sh` exited 1 exactly when only the dense gate had
failed (`[ -n "$PAIRED_NOTE" ] && echo` as the branch's last command; `if/fi`, explicit `exit 0`, and the tail is now
tested for all four outcomes). C-2 the E9 loop dropped untrained arms silently and a glob cannot see a directory never
created: the queue now inventories every arm it will look for — the main grid, B5's B0 weights, the E9 reference and an
explicit `E9_ARMS` list matching `run_s2.sh` (`EXTENDED_E9` included) — **before the first job**, writes
`runs/eval/SKIPPED_ARMS.txt` and refuses to start unless `ALLOW_MISSING_ARMS=1`, which names the gap at the top and
again at the end (before the failed-job verdict, so both are seen). C-3 `collect.py` padded per-seed lists *after*
appending, so a key first scored at seed 2 landed at index 0 labelled seed 0: padding now precedes the append; E8's
`per_seed` is positional over its seeds too; the coverage row gained a seed dimension.

**Invariants.** INV-3 tightened where that is legitimate: `INV-3a` bounds *Stage 1's* output `Σ a1 ≤ Σ A_sm + CAP_TOL +
tol_cap` (fp64 sums of fp32 entries: a 1e-5 excess trips it; the old bound needed ~6e-4 at 16K), while INV-3 on `A`
keeps Stage 2's fp32 allowance `tol_sum` — the reviewer's fp64-level tightening of INV-3 itself is *wrong*: T6 at
`τ_i = 50` overshoots by ~7e-6 on a 10-key row, INV-7's term. `INV-5c` asserts the cap's own dual (`Diagnostics.cap_theta`,
recorded on all three paths; `theta` is Stage 2's and non-negative by construction, so `check_unit_cap.py:117` and the
razor test were vacuous) is positive on every binding row and ≤ 0 on every over-`CAP_TOL` row left alone; the sparse
`INV-5-plumbing` is two-sided with it. The stored `sm_mass` is protected by `INV-3-mass` (`|rowsum − sm_mass − excess|`
against the kernel bound; ×1.1 trips it). The sparse INV-3 no longer reads the fp32 `rowsum` against the fp64 `sm_mass`
— that difference is the kernel's razor error, 1.2e-4 at 16K on this CPU, *above* the 1.14e-4 it allowed (a latent
false trip) — but the store's fp64 sums alone. `invariant_report` runs under `@torch.no_grad` and holds no whole-row
fp64 copy (`sum(dtype=float64)` on a [B,H,T,T] tensor is 1.0 GiB on CUDA at 8K/hb2; the old line 89 was 3.0 GiB).

**Smaller.** `unit_cap_record`'s rule string names the target and the guard; `collect.py` carries `rho_col`, `rho_row`,
`zero_mass_tol` (the tolerance that defines `frac_rows_zero_mass`), `pass2_rows`, keeps the count-weighted pass-2
fraction as the mean, reads E6's strict and contains tables, and `source` is set on dense blocks; `profile_memory.py`
builds the paired model inside a try (an OOM there writes the JSON with a failed gate and exits 3, which preflight
tolerates); `run_evalsuite.sh` treats an unreadable paired JSON as a refusal, `run_s2.sh` reads the detector guarded,
both queues reject `NGPU` ≤ 0 or non-integer; preflight step 5 also times the dense path (recorded, `--no_gate`).
Not done: the fourth-form excess *is* the shipped one now; E6's decision and the paper edits remain the author's.

## Review at e6ca44f (`marsea_review_e6ca44f.md`) — one harness, a defined failure branch

The review confirmed every fix of the previous round on their CPU (window 8.8e-8 at 8K / 1.1e-7 at 16K there; the
CUDA figures stand on this box's measurement), reproduced the INV-3 refutation at physical scale (the Stage-2 overshoot
is ~8e-7 and scales with the row's *mass*, not its length), and found one new thing, verified here before acting:

* **Two `verify_all.py`** (D). `scripts/verify_all.py` (779 lines) carried the machine-readable verdict and the exit code
  preflight gates on; `documents/verify_all.py` (762) carried the reviewer's re-target of Prop. A(iii) to the row's own
  softmax mass and the matching label. Neither had both, and the copy that runs printed the label the paper no longer
  uses. The reviewer's merged file was diffed against both (exactly the code copy plus the five softmax-mass hunks),
  adopted as `scripts/verify_all.py` (0 violations over 1,058,450 evaluations, exit 0, `A(iii) … own softmax mass 0 /
  80741`), and **`documents/verify_all.py` is now a symlink to it**; the transient merged copy was removed. One file.
* **The window gate's failure branch** (E). `cap_tolerance`'s `8·log2(n_k)·ε` is not a derived law — a sequential fp32
  scan can reach `n_k·ε/4`, 36× it at 16K; it is a number the measured window has to clear, and the gate is what makes
  that safe. If it fails on the target device: `MARSEA_TOL_CAP := 4 × window` (the gate prints the number), re-run,
  quote it in App. F; `cap_tolerance` reads the override (never below the default), `unit_cap_record` writes the value in
  force into every README and eval table, and `runs/unit_cap_device.json` records it. In the runbook under 4b.
* **Manifest completeness** (H). `handoff.sh` now hashes every file the tarball ships (two PDFs under `scripts/` were
  unlisted).
* **Skip counts** (H). On a CPU with the tokenizer cache warm, 8 tests skip, all GPU-gated (six T0 backbone tests, two
  bf16-autocast chunked tests); a cold cache adds the six `test_tj_split.py` tests, so 14 skips on a fresh container is
  expected. No skip is a regression; everything else must pass.

Not done: `PAPER_EDITS_e982f83.md` is behind the current tex (the reviewer has patched the paper directly and has the
file); anything still owed there is a patch against the current `.tex`, which is the author's. The working-tree
question (§G) was left as instructed.

## Review at 4f754ed (`marsea_review_4f754ed.md`) — the override gets a ceiling

Everything from e6ca44f verified by the reviewer (the merged harness sha256-identical to theirs, the symlink, the
manifest, the failure branch). One new finding, reproduced here before acting: `MARSEA_TOL_CAP` took
`max(default, override)` with no upper bound, so `MARSEA_TOL_CAP=1` made INV-3a and INV-5b admit any row for a whole
queue while recording the number faithfully, and a malformed value raised inside an invariant check mid-run with no
mention of the variable. The same shape as `cap_slack`: a tolerance nothing physical bounds, at the one place a human
still types one in. Now (`invariants.validate_tol_cap_override`): parsed once at import, naming the variable; refused
above `L·ε/4` at the run's length (no fp32 prefix scan errs by more) — checked at startup by `unit_cap_record`, so a
bad value stops the run before a GPU is touched; and refused by `check_unit_cap.py` when it exceeds 8× the window
measured on *this* device, so a stale override cannot make the gate self-certifying after a reschedule. The intended
path (4× a 2.0e-6 window = 8e-6) clears both bars. `unit_cap_record` records the ceiling beside the value.

Also noted from the review: the bundle attached last round was the previous one (`e6ca44f`), caught by their
provenance check; the `4f754ed` bundle was on disk. And the paper's supplementary build must dereference the
`documents/verify_all.py` symlink (`cp -L` / `tar -h`).

## Review at 7e05ad6 (`marsea_review_7e05ad6.md`) — the eval path records and validates at its length

Both override bars verified by the reviewer by running them; the bundle matched the commit for the first time in two
rounds. Minor items taken: `run_eval.py` called `unit_cap_record()` with no length, so every eval table recorded `None`
for the tolerance and the ceiling and the override was never validated on that path — it now passes the job's length
(the longest tokenised prompt + gold, or the `_L<n>` of a RULER set name), so the eval tables carry the same three
numbers the training READMEs do and an over-ceiling override refuses the job at startup. Three review tests read files
`handoff.sh` deliberately leaves out of the bundle (`RUNBOOK_nebius.md`, `documents/verify_all.py`); they now skip with
a reason when the path is absent rather than fail against the tarball. The redundant NaN test in the override parse is
gone. Not taken (by design, per the review): the bundle stays code-only.

## Ops-playbook review (2026-09-17) — the training launcher, atomic checkpoints, recorded resumes

Found while writing the code-level operations playbook (`documents/runbook/marsea_ops_playbook.md`) and confirmed by
its review: `run_s2.sh`'s round-robin counter was never reset after a wave, so Phase A's three launches shifted every
Phase-B wave — on 8 GPUs the 17 chunked jobs ran as 5 / 8 / 4 and the three dense E9 arms as a fourth wave (reproduced
by simulating `launch()`/`barrier()`). `barrier()` now resets the counter and the dense arms are launched first inside
the same round-robin: 20 jobs in 8 / 8 / 4, one arm-duration less of wall clock. `test_ops_playbook_review.py` runs the
real script with recording stand-ins and asserts the GPU sequence. Also: `save_checkpoint` writes under a temporary name
and renames, so a pod lost during the 250-step `last.pt` overwrite no longer leaves an unloadable file; a resume that
cannot restore its optimizer state now records `optimizer state not restored on resume` in the README (it was a stdout
line only, and such an arm is a different recipe from its siblings). Round 2 of that review: the checkpoint write is
also *durable* — `fsync` of the data before the rename and of the directory after it, since `os.replace` alone is atomic
against process death but a node can die with the rename published and the blocks unwritten (a network volume is not
guaranteed ext4 `data=ordered`); a behavioural test kills `torch.save` midway and checks the old `last.pt` still loads.
`SKIP_DENSE_E9=1` now drops the dense arms whatever the memory verdict (it was consulted only after a failed one).

## Review at acda7b6 (setup script) — the product-string trap, `runs/` as output, the data gate

Each verified before acting. The SXM check was an allow-list on the product string and NVIDIA does not put "SXM" in it
(an H200 SXM reports plain `NVIDIA H200`), so it warned on the correct hardware and would have been ignored when a real
`NVL` part came: it is now a deny-list on `NVL|PCIe`, plus a memory floor (`MIN_GPU_MIB`, 140 000) and a visible-count
check (`EXPECT_NGPU`, 8). Five `runs/*.json` were tracked, `detector.json` among them, and S0 writes into that file: from
S0 on every README, checkpoint and eval table would have carried `<hash>+dirty`. `runs/` is output now, wholesale, and
a test asserts `git ls-files runs/` is empty. `RUNBOOK_nebius.md` is back in the tracked root (the two gate-4b doc-code
tests were inert with it under the ignored `documents/`, and the pod had no runbook). The setup script's data check
exits 1 on any set without the D0 length provenance (never correct to proceed) and, under `REQUIRE_DATA=1`, on an
incomplete upload; the HF cache rule is "same filesystem as the repo" rather than a `/workspace` name test (silent on
any other mount point); `runs/backbone.json` records the backbone snapshot's commit sha for the appendix. The
`SKIP_DENSE_E9` drop is announced once and uses an `if`.

## Engineering findings recorded during implementation (not decisions; spec Sec. 17)

* **Unit-cap binding decision.** `proj_le` for the unit cap binds iff the relation's excess `Σ_{E_i.}(Atil − A_sm)`
  exceeds `CAP_TOL = 1e-6` (exact in fp64), never on the fp32 row total, whose error is a kernel property; a binding row
  is projected onto its own fp64 softmax mass, and the projection refuses to bind when its arithmetic finds nothing to
  remove, so the cap never adds mass (see the two sections above). This makes INV-5 and INV-9 exact and T0's `E = ∅` comparison bit-identical
  in bf16.  The quota cap in fan-in step 2 tests its sum with a zero tolerance (T16's `tau_i = 1` identity needs it).
* **INV-1 tolerance.** The relation's total is a sum of up to `n_q` fp32 products; it is asserted at `1e-5` absolute-or-
  relative at `n_q <= 64`, scaling with `sqrt(n_q/64)`, and at `1e-12` in fp64 (round-2 ambiguity #30).
* **`std` at a one-query column.** `sqrt` has an infinite derivative at 0, so `column_stats` computes it on a safe
  argument and returns exactly 0 with a zero subgradient (H11/H12); the unguarded form NaNs the backward of every causal batch.
* **Tournament root retention.** `hierarchical_topk` retains `K_ret` per block AND `K_ret` at the root; under-retention
  (`K_ret < k*`) then yields the top-`K_ret` prefix (harness Prop. 22), and exactness for `K_ret >= k*` is unchanged.
* **Chunked path and the ST gate.** The straight-through gradient is dense on visible pairs at all four sites; the
  chunked path accumulates the off-relation part of sites (3)/(4) per key chunk and recomputes it densely for the
  (few) rows the unit cap binds.  Gradients agree with the dense path to 1e-8 relative in fp64.
* **Chunked == dense at the model level, up to one bf16 ulp.** With the bf16 backbone the two paths produce attention
  outputs that differ in ~1 % of elements by exactly one bf16 ulp (different fp32 accumulation order); the network then
  amplifies that through the following layers (mean relative hidden-state difference ~1 % by layer 11, greedy argmax
  unchanged on the 2K probe).  T11's fp32/fp64 tensor-level equality is the meaningful statement; the bf16 drift is
  a kernel property shared with SDPA-vs-eager.
* **`vis` from the 2-D mask.** transformers 5.16 hands SDPA layers `None` or a boolean mask; `vis` is built inside the
  patched layer (spec Sec. 6.1 [v4.1]).  Left padding is not an error: pad query rows are fully invisible and get `A = 0`.
* **T16 (padding-length invariance)** is asserted in fp32 with eager unpatched layers: bf16 GEMM tiling differs with
  the padded shape and drifts to ~1e-2 relative through 28 layers, which is a kernel property, not a mask leak.
* **transformers pin.** 5.16.1 (>= 4.53, D-30); `assert_transformers_contract()` checks `eager_attention_forward`'s signature.
* **The split `T_j` (spec v4.7, 2026-09-09).** The coreference column of the key phrase's first mention has the later
  mentions as targets, `m_j = m + 1` on MV-NIAH (the other `m - 1` needles, the question, the answer prefix); the answer
  positions belong to the value columns (`T_j_kind`: `coref` / `value`; QA analogue: bridge-string later mentions /
  the gold answer's first mention).  S0 licenses each kind separately (`column_measurable = {coref, value}` in
  `detector.json`) and the tables report `interval_hit_rate_by_kind`.  Regression test: `tests/test_tj_split.py`.
* **Uniform-quota ablation (E9)**: every active column of a (b, h) slice receives the slice's in-relation mass divided by
  its number of active columns (the in-relation total is preserved).  Stated here because the spec names the arm
  without a formula.
