# MarSea — decisions register

*One line per standing decision, with the master-record section that holds the reasoning. The
register is the index; the record is the argument. Newest first within each group.*

> ⭐ **The record is one file again.** `claude/marsea_master_record.md` holds **§0–§60**, merged
> 2026-09-15 (§60.4); the live end is **§60**. `claude/marsea_master_record_part2.md` is a
> superseded pointer — do not write to it. Every section number below resolves against the one
> record.

## Implementation and review process (2026-09-13)

| # | Decision | Record |
|---|---|---|
| D-34 | **Review method.** Three readers with **different mandates** (fix verification + regression hunt; adversarial numerics + mechanism fidelity; the experiment programme end to end) — never the same mandate three times. Reviewers **execute**: torch in the review container, findings measured not inferred. Once two verified trees exist, the review is a **diff**, not a re-read. | §56.4, §27 |
| D-33 | **Every review document is delivered three ways** — downloadable file, project doc, and (where it carries a decision) a record section. A document that exists in one place does not exist. | §56.5 |
| D-32 | ⭐⭐ **Provenance protocol, mandatory before a line is read.** (1) `git rev-parse HEAD`; (2) `git status --porcelain` **must be empty** — an uncommitted tree is not reviewable; (3) `sha256sum` manifest over every `*.py`/`*.sh` under `marsea/`, `scripts/`, `tests/`; (4) the reviewer verifies all three first. Automated as `scripts/handoff.sh` → `HANDOFF.txt` + tarball. | §56.3 |

## Mechanism — numerical contract (2026-09-15)

| # | Decision | Record |
|---|---|---|
| D-37 | ⭐⭐ **The unit cap binds on the RELATION'S MASS EXCESS, and a binding row is projected onto the row's own softmax mass.** `excess_i = Σ_{j∈E_i.}(Atil_ij − A_sm_ij) > CAP_TOL = 1e-6`, accumulated in fp64 from entries cast **before** subtracting; the target is `sm_mass_i = Σ_j A_sm_ij` (fp64), never the literal 1; and `proj_le_masked` additionally refuses to bind when its own arithmetic finds `θ ≤ 0`. Because `Atil == A_sm` bitwise off `E`, the excess equals `Σ_j Atil_ij − Σ_j A_sm_ij` exactly and **reads no row total**, so the decision is independent of the softmax reduction kernel. An empty relation gives excess exactly `0.0`, so T0 and sanity 5(a) hold bitwise on any device. `cap_slack`, its ceiling and its formula are **deleted**. | §59 |
| D-36 | ⚠️ **Two forward-pass changes rode in with D-37 and are kept.** (a) The chunked path computes `exp((S−m)−r)` (`softmax_about_max`), not `exp(S−lse)` — flat at ~1e-6 in `n_k` against `½·ulp(|lse|)`. The dense and decode paths keep `torch.softmax` for T0 bit-compatibility, so the two differ by ~1e-6 relative and **the appendix must say which formula produced the chunked numbers**. (b) The cap's projection target. Chunked numbers are **not bit-comparable** across e982f83 → 9bacef9. | §59.4 |
| ~~D-35~~ | ⚠️ **SUPERSEDED by D-37.** `cap_slack(n_k, dtype)` — a size- and dtype-dependent binding *slack* — was the third of three constants, each right on exactly one kernel. The law is `δ_max = n_k·eps/(2L)` with `L` the reduction's lane count (~1e-6 CUDA, 6e-5 at 16 lanes, 1.2e-4 at 8, up to 1e-3 scalar), so **no constant is correct on an unknown CPU**. Kept as the index of a dead end; the run README still records the rule in force. | §59.1 |

## Compute and schedule (2026-09-05)

| # | Decision | Record |
|---|---|---|
| D-26 | **Provider: Nebius AI Cloud**, two stages — 1–2 H100 VMs for setup/S0/S1/data prep (~4 days), one 8×H100 VM for S2 + minimal S4 (~2 days). CoreWeave rejected (8-GPU node only, $6.16/GPU-h); RunPod is the fallback account (Secure, H100 SXM). | §44 |
| D-25 | Budget: reduced programme ≈ $1,500–1,950 at list price; full ≈ $3,000–3,500. On-demand for S2; preemptible only for E9 reruns. ⚠️ **The queue as written implies ≈1,000 H100-h, a 3× overrun** — see Open. | §44, §58.5 |
| D-24 | Programme that fits 25 Sep: train at 8K, evaluate E3 at 16K (chunked path); Phase B 2,000 steps; 3 seeds on MarSea/B0/B2/B3, 1 seed on B1/B4; E9 = {τ_i pinned at 1, inherited vs uniform quota, pairwise vs key-alone relation}. | §44 |
| D-23 | Operating rules: European region for VMs + shared FS; quota request day one; checkpoint every 250 steps; idempotent restart; queue of 8 single-GPU processes; daily off-platform copy; S1 step-time rule (> 1.2 s/seq → 2 patched layers or 4K). | §44 |
| D-9a | **Confirmed 2026-09-10.** `T_j` is SPLIT: the coreference column's `T_j` = later mentions only (other needles, question, answer prefix), `m_j = m + 1` on MV-NIAH; the answer positions belong to the value columns (`m_j` = emitting tokens, typically 1). Not `2m + 1`: S0 showed answer positions route to values, not to the first mention. Column test at the head S0 finds (e.g. (14, 3)); row test at the detector's `(l*, h*)`. ⚠️ **The two sites may share a layer** — the source of the B-1 defect class (§58.4). | §53, §58.4, spec §12.3, paper App. H E7 |
| D-31 | **Confirmed 2026-09-09.** Position 0 (the sink) is excluded from every relation: `E[:,0] = E[0,:] = False` via a mask on the logits; its row and column stay at standard attention; `b0` calibration excludes those pairs. Paper Sec. 2.3. | §52, spec §4.1 |
| D-30 | **Confirmed 2026-09-08.** `transformers >= 4.53`; exact release pinned at implementation time; `eager_attention_forward` signature asserted at import. | spec §17.5 |
| D-29 | **Confirmed 2026-09-08 ("for now").** B5 (matched-sparsity softmax) is teacher-forced only; its task-level cell is "n/a by design". | spec §8, §17.4 |
| D-28 | **Confirmed 2026-09-08.** Decode-time procedure = spec §6.4 as written (frozen-prefix; per-key re-solve on the grown relation at sealed `τ_j`; incremental `\bar c_j`; only row `t` emitted; past rows never revised). MarSea's task-level numbers and cost come from this path. | spec §6.4, §17.3 |
| D-27 | **Confirmed 2026-09-08.** E2 at K = 8 keys, L = 8K; E3 at L = 16K held constant across the n-sweep (commits S2 to the chunked path of spec §6.6). | spec §13.1, §17.2 |
| D-22 | Backbone: Qwen2.5-1.5B (base) primary, Llama-3.2-3B cross-family; patch = the softmax line of `eager_attention_forward` in the layers holding the backbone's retrieval heads. ⚠️ **No Llama arm exists in the queue** — see Open. | §39.1, §58.6 |

## Mechanism (standing; do not relitigate)

| # | Decision | Record |
|---|---|---|
| D-21 | `τ_i = 1` is the identity on step 1's row; E8 logs `τ_i` about 1. (Reverses audit-5's "not an identity".) | §43 |
| D-20 | Step 1 is the **row** softmax (`A^{sm}`, standard attention); the column stage reads its column sums; no column softmax anywhere except as the α→1 limit of step 2 on the relation. | §41 |
| D-19 | No column abstention component. `m_j ∈ {0,1}` columns are row-side tests; `m_j = 1` is the `τ_j ≥ 1/δ_j` regime of Thm. 2; distractors are removed by the composition (Stage-1 concentration + row cap); E8 reports the three-way split. | §40 |
| D-18 | `τ_j → 0` is the uniform re-shaping, NOT standard attention; only `𝓔 = ∅` is. Warm start through a near-empty relation (ρ₀ = 5 %), never through τ. | §39.2 |
| D-17 | Softmax runs exactly once, at Stage 1 step 1; no normalisation of any kind after it. `\bar c_j`, `\bar c_i` read attention weights, never raw scores. | §38.1 |
| D-16 | Name: MarSea (Multivariate Relations by Selective Exclusion in Attention). | §37 |
| D-15 | Two-step selective form is the mechanism: fan-out capacity inherited (never predicted), step 2 re-shapes the relation slice; fan-in step 1 = unit cap, step 2 = cap `≤ \bar c_i` (not equality); no gates — the relation is the only membership object; empty relation = standard attention entrywise. | §35–36 |
| D-14 | The claim is a bound on **degree**, not mass; `Cor. capacity`'s capacity half disclaimed; E1 measures degree. | §36 |
| D-13 | Notation `\bar c_j`, `\bar c_i`; explicit relation/complement layout in eqs. 3 and 7. | §38 |
| D-12 | The relation is pairwise (`u_φ(k_j)·v_φ(q_i) > 0`, with `b0` as a constant coordinate), arrival-sealed; never reads the query set as a whole. | §36, spec §4 |
| D-11 | Causal/prefix form is native design, never an obstruction. | earlier |
| D-10 | Precision is the centre; recall is protected. | earlier |

## Paper and experiments

| # | Decision | Record |
|---|---|---|
| D-9 | **Confirmed 2026-09-07.** `T_j` for the column tests = the coreference-by-string-match construction; the S0 routing check on B0 licenses it; relation-recall misses logged separately; option (e) code def–use chains stays in reserve. Paper: App. H "E7, the column ground truth". **Superseded in part by D-9a's split.** | §40, §46, §53, spec §12.3 |
| D-8 | E3 (n-sweep) is load-bearing; E2 tests Thm. 2 via the hit rate stratified by m. | §36 |
| D-7 | MESH = capacity-without-shape (both marginals learned, dense); key-alone τ^K = the field-argument test; MESH's two-marginal problem is infeasible under a causal mask — stated as the baseline's limitation. | §39.3, §42 |
| D-6 | ICLR 2027 venue; TMLR draft is the fuller fallback; 9-page limit deferred until experiments land. | earlier |
| D-5 | Contribution (i) = the 2×2 object (capacity × shape on both margins); MarSea fills the shape cells and inherits capacity. | §34 |
| D-4 | Verification count not led with; dagger on deterministic rows; per-row units. | §32 (audit 4, decision 2B) |
| D-3 | Contribution sharpened to the field argument; MESH named as prior art for learned marginals. | §32 (audit 4, decision 1B) |
| D-2 | Do not rewrite motivating examples unilaterally. | earlier |
| D-1 | `joca_ws` dropped. | earlier |

---

## Open (Luke's) — ⚠️ ordered by what blocks 25 Sep. Abstract due **18 Sep**.

### Blocks the queue launch — three one-line edits, all verified by execution (§59.6)

| Item | Where |
|---|---|
| ⚠️⚠️ `preflight.sh:136-141` — the `if`-branch's last command is `[ -n "$PAIRED_NOTE" ] && echo`, which returns 1 when empty, so **the first command in the runbook exits 1 exactly when the dense-8K gate fails** — the condition it has just declared non-fatal. `; true`. | §59.6 |
| ⚠️⚠️ `run_evalsuite.sh:159-167` — the E9 loop has no `SKIPPED_ARMS+=` and cannot see an arm whose directory was never created. On the documented `SKIP_DENSE_E9=1` recovery, **three of six E9 arms vanish with exit 0 and an empty SKIPPED_ARMS.txt**, and `collect.py` then prints a three-arm E9 table saying nothing. E9 is D-24's load-bearing test for contribution (i). Compute the skipped list at the **top**, beside the data gate. | §59.6 |
| ⚠️ `collect.py:236-240` — the pad appends *after* seed `si`, so a key first scored at seed 2 lands at index 0 and is then labelled `seeds=[0,1,2]`. Fix the left-padding, or drop `seeds=seeds` (short-and-unlabelled is honest; full-and-mislabelled is not). | §59.6 |
| Then: `preflight.sh` on the H100 → `run_s2.sh` (stops after Phase A) → `run_s0.sh` → `run_s2.sh` → `run_evalsuite.sh` → `collect.py`. | §59.6 |

### The one open question on the mechanism

| Item | Where |
|---|---|
| ⚠️ **The `θ ≤ 0` guard's suppression window has never been measured on a GPU**, and `check_unit_cap.py` cannot detect it — its live-decision assert was weakened to one-sided. On CPU the window is 8.7e-8 (torch's `cumsum` carries a higher-precision accumulator); on a device whose prefix scan is a genuine fp32 reduction it becomes ≈ `n_k·eps/4` ≈ **2.4e-4 at 8K, 240× CAP_TOL**. Bisect it on the device and gate it. One function. | §59.3 |

### Paper — the cap edits, ✅ APPLIED 2026-09-15 (§60)

| Item | Where |
|---|---|
| ✅ **All eight applied**, by `patch_iclr_cap.py` (all eight `old` strings matched exactly once). `prop:twostep`(iii) now reads `Σ_j A_ij ≤ Σ_j A^sm_ij`; the finite-precision remark and `eq:excess` are in §3.2; §2.2, the proof, `tab:verification`, App. D and App. E follow; **Patch 7's device statistic is in App. F**. Build: 42 pp, 0 err / 0 undef / 0 multiply-def / 0 overfull, every edit checked in the rendered PDF. `verify_all.py`'s `Prop. A(iii)` row re-targeted to the row's own softmax mass — re-run, exactly one line of output differs (the label). | §60.2, paper-edits doc |
| ⚠️ **Before this could be applied, the two copies of the draft had to be reconciled**: `~/Dev/marsea` was two decisions behind the project (missing D-31's sink sentence and D-9a's E7 rewrite), and the eight cap patches matched the *stale* copy cleanly. Forward-ported by `portforward_d9a_d31.py`; both passages verified byte-identical to the project's own text. **The project and the repo now hold the same bytes.** | §60.1 |
| Which softmax formula produced the chunked numbers (D-36), and the `cap` rule in force (already in the run README). | §59.4 |

### Paper — forced by `--n 400`, and new

| Item | Where |
|---|---|
| E3's **per-stratum exact-set accuracy** carries a ±0.14 interval at 80 examples/stratum/seed — report it **pooled over n**. | review e16a843 §H |
| **"Flat" becomes an explicit equivalence band** (±0.021 to ±0.057 on total drift across the four doublings). Pre-commit to it. | review e16a843 §H |
| The **depth panel is one seed and one baseline** (B0 only) — caption it, no error bar. | review e16a843 §H |
| MarSea vs B3 and vs B2 are **under-powered at three seeds** — say so rather than report a silent null. | review e16a843 §H |

### Paper — claims nothing in the queue produces (**not edited unilaterally**, D-2)

| Item | Where |
|---|---|
| Cut or qualify: Llama cross-family (D-22); "we ablate both emission policies"; the hard-variant ablation; the τᴷ–τQ tying; E9 over layer set and block size; E7's measured PR trace; efficiency-vs-coverage; "2–16K"; the E3 padding control's "every point"; B1/B4 at one seed. | paper-edits doc |
| ⚠️ **E6 — the oldest open decision in the project.** Nothing downloads IHEval; the queue prints "NOT RUN". It is the only experiment testing the *ranked* half of the motivating scene. Acquire it this week or state in the paper that it was not run. | §58.6 |
| Remove `prop:twostep`(ii) and the "derived, not imposed" anchor sentences (offered §47, still unanswered). | §47 |
| Audit-5 C5, C6; the TMLR consequences pass. | §39.5 |

### Code, during the queue

| Item | Where |
|---|---|
| `check_unit_cap.py`: add the θ-window bisection (above); record `max \|relation_excess − exact fp64\|` and `min \|excess − CAP_TOL\|`; exercise ρ=0.5 and the decode path on a *live* relation. | §59.3 |
| INV-3 and INV-5b keep the fp32-era `tol_sum` on quantities that are now fp64 sums — a `+3e-4` corruption of `A` does not trip INV-3. Tighten to `rs_sm + CAP_TOL + O(n_k·2⁻⁵²)`: ~500–1000× sensitivity, free. | §59.6 |
| `invariant_report` is not `@torch.no_grad` and builds a whole-row fp64 copy with graph (~2 GiB at 8K/hb2) — and **preflight step 2 runs `MARSEA_DEBUG=1` as a hard gate**. | §59.6 |
| `INV-5-plumbing` is one-sided; the stored `sm_mass` is protected by nothing on the sparse path; `diag.theta` is Stage-2's dual, not the cap's, so the θ asserts are vacuous — record `extra["cap_theta"]`. | §59.6 |
| `train.py:211` `unit_cap_record`'s rule string still describes the pre-D-37 rule, and it is written into every README and eval table. | §59.6 |
| `collect.py` still drops `rho_col`, `rho_row`, `zero_mass_tol` (the last defines `frac_rows_zero_mass`), `_pass2_rows`; `source` is `None` on dense blocks; E8's own `per_seed` is unpadded; the coverage row has no seed dimension; E6's `exact_match_strict` is never read. | §59.6 |
| `run_evalsuite.sh` catches only `FileNotFoundError` on the paired JSON; `run_s2.sh:90` reads the detector unguarded; `NGPU` of 0 or a non-integer hangs both queues. `profile_memory.py` builds the second model outside any `try`, so an OOM there still aborts 4b/5/6. | §59.6 |
| ⭐ **The parity harness** — one configuration hitting every rare branch, asserting every E8 key, every cache field and every column record across dense and sparse. Still not built; it is the structural defence behind every instrumentation defect of §57–§59. | §57, §58.8 |
| Still unmeasured for a fifth round: **evaluation throughput**. It is the largest term in the cost model. | §59.6 |
