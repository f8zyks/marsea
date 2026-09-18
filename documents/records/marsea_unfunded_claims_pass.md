# MarSea — the unfunded-claims pass on `marsea_iclr.tex`

*2026-09-16. Method: every sentence in the draft that promises a measurement, checked against
(a) the D-24 reduced programme actually queued in `run_s2.sh` / `run_evalsuite.sh`, and (b) whether
the quantity it names is computed anywhere in the code. **Not a re-reading of
`PAPER_EDITS_e982f83.md`'s list** — that list was built from the draft alone; this one was built
from the draft against the repo, which is why it is longer.*

**Copy checked:** `~/Dev/marsea/documents/marsea_iclr.tex`, 2,963 lines, mtime 2026-09-15 18:55Z.
§0.3b identity probes all pass — D-31 sink sentence ✓, `\emph{coreference column}` ✓,
`\emph{value column}` ✓, `\label{eq:excess}` ✓, the θ-window amendment ✓ (lines 695–699, 2520–2527).
The project copy carries the same markers. Line numbers below are this file's.

⚠️ **Nothing has been edited.** This is the pass, not the patch. Once you sign off on the wordings
I will produce a `patch_iclr_unfunded.py` in the usual form (asserts every match up front, `.bak`,
post-check, refuses double-apply).

---

## ⚠️ Amendment, 2026-09-16 evening — one item closed, one added

**Closed.** The draft's claim that E3 runs at **16K** was briefly false: the generator was producing
11,598 tokens for a 16,384 request (see the run plan §1.5). Fixed by `patch_ruler_length.py`; the
regenerated sets measure 96.4–98.8 % of `L`, so the claim is true again and needs no edit.

**Added, and already applied.** The same investigation left a **4-point gold-depth drift that is
monotone in `n`** (0.470 → 0.510). That is now **pre-registered in the E3 controls paragraph**
(`patch_iclr_e3prereg.py`), together with the measured length parity and a commitment to bound the
drift by E3depth's own slope rather than argue it away. ⭐ Note what kind of item this is: not an
unfunded claim but the opposite — a *funded* disclosure added before the data was seen. ⚠️ Its
numbers come from `--n 20, seed 0` and must be re-measured on the full grid before submission.

**Unchanged.** Item 7 below — the unfalsifiable "flat" in the E3 paragraph — is still open and is
still the one I would do next.

---

## The count, and why it grew

| | |
|---|---|
| previously tracked | ~11 unfunded promises |
| **now** | **23**, in four classes |
| of which *new* | **9 found by checking the code rather than the prose** — a sentence can name a quantity that reads plausibly and simply is not computed anywhere |
| of which *newly* unfunded | **6**, created this morning by closing E6 |

⭐ **The single most important finding is item A-6.** The paper calls the field-argument ablation
"contribution (i)'s load-bearing test", and the specific arm it names third —
`τ^K` with and without `ν_j` — is `no_nu`, which sits inside `EXTENDED_E9` and **D-24 turns
`EXTENDED_E9` off**. The load-bearing ablation of the paper's own novel object is not in the queue.

---

## Class A — promises with no arm and no code (7). **Cut or re-aim.**

### A-1 · line 541 · §2.5, causal form

> "We ablate both emission policies."

**Unfunded.** `MarSeaContext` has no emission-policy knob (`normalizer.py` §206–228 lists
`no_nu, key_only_relation, tau_i_pinned, tau_j_global, quota_mode, gate, per_head, hidden, K_ret,
block_size, head_block` — no emission policy), and no arm in `run_s2.sh` varies one.

**Proposed:** delete the sentence. The paragraph's argument — that growth is edge-removing and the
frozen-prefix error is one-sided — is a theorem (Prop. `prefix`) and stands without it.

### A-2 · lines 1551–52 · App. limitations, same claim restated

> "the two emission policies are a real choice … and we report both rather than claim one dominates."

**Unfunded**, same reason. ⚠️ This one is worse than A-1 because it explicitly promises a reported
comparison.

**Proposed:** → "…and we do not claim one dominates; which to prefer is a deployment question we do
not settle here."

### A-3 · line 1538 · App. limitations, hard vs soft truncation

> "a hard variant preserves them at the cost of differentiability at the threshold, and we ablate both."

**Unfunded.** No hard-truncation variant exists in `normalizer.py`.

**Proposed:** → "…at the cost of differentiability at the threshold; we deploy the soft form and do
not evaluate the hard one."

### A-4 · lines 1486–87 · App. limitations, τ^K–τ^Q tying

> "E9 ablates the tying of `τ^K` to `τ^Q` in self-attention, where it is definable, and we predict it costs; the cost is what makes the separation mechanism rather than bookkeeping."

**Unfunded.** No tying knob, no arm. ⚠️ And the last clause stakes the *separation's* status on a
measurement that will not exist.

**Proposed:** → "Tying `τ^K` to `τ^Q` is definable in self-attention and we expect it to cost, but we
do not evaluate it; the separation is argued from Thm. 1 and from the two heads' different targets,
not from an ablation."

### A-5 · line 1057 · §5, the E9 summary

> "E9 ablates every design decision, **including each head's field argument and the tying of `τ^K` to `τ^Q`**"

Field argument: **partly funded** (see A-6). Tying: **unfunded** (A-4). "Every design decision":
**false** under D-24.

**Proposed:** → "E9 ablates the design decisions the theory rests on — each head's field argument,
the relation's parameterization, the retention constant `K`, and the inherited capacity against a
predicted one"

### A-6 ⭐⭐ · lines 2903–2912 · App. E9, the full list

> "**E9** ablates every design decision in turn — the relation's parameterization (its rank `r`, pairwise against key-alone, and which layers carry it), `K`, block size and emission policy, `τ_i` learned against pinned at 1 …, `τ_j` learned against fixed or uniform, and the inherited capacity … **The load-bearing ablation of this group strips the field argument from each head** — `τ^K(k_j)` against `τ^K(k_j,s_{·j},ν_j)`, `e_ij` from `u_φ(k_j)` alone against the pair, and **`τ^K` with and without `ν_j`**"

Checked item by item against the D-24 queue:

| promised | knob | in D-24? |
|---|---|---|
| pairwise vs key-alone relation | `key_only_relation` | ✅ `e9_key_only_relation` |
| `τ_i` learned vs pinned at 1 | `tau_i_pinned` | ✅ `e9_tau_i_pinned` |
| inherited vs uniform capacity | `quota_mode="uniform"` | ✅ `e9_uniform_quota` + matched control |
| `K` (retention) | `K_ret` | ✅ `e9_hierarchical_K64` |
| per-head parameters | `per_head` | ✅ `e9_per_head` (not promised in this list, but runs) |
| `τ^K(k_j)` vs `τ^K(k_j,s,ν)` | `key_only_tau` | ✅ — **this is baseline B3**, 3 seeds |
| **`τ^K` with and without `ν_j`** | `no_nu` | ❌ **`EXTENDED_E9`, off under D-24** |
| `τ_j` learned vs fixed/uniform | `tau_j_global` | ❌ `EXTENDED_E9`, off |
| relation rank `r` | — | ❌ no arm |
| which layers carry the relation | — | ❌ no arm |
| block size | `block_size` | ❌ no arm |
| emission policy | — | ❌ no knob (A-1) |

**Proposed:** replace the list with what runs, and move the field-argument sentence to name only its
two funded members:

> "**E9** ablates the decisions the claims rest on: the relation read pairwise against from the key
> alone, `τ_i` learned against pinned at 1 (which reduces Stage 2 to a plain unit cap), the
> inherited capacity of (2) against a uniform one — the ablation that puts a predicted budget back,
> and is reported as such — the retention constant `K`, and per-head against shared parameters.
> **The load-bearing ablation of this group strips the field argument from each head**: `τ^K(k_j)`
> against `τ^K(k_j,s_{·j},ν_j)` (reported as baseline B3), and `e_ij` from `u_φ(k_j)` alone against
> the pair. Rank, layer placement, block size and the emission policy are not ablated here."

⭐ **Or fund it instead.** `no_nu` costs **one arm × one seed ≈ 5 GPU-h**, forks the shared Phase-A
checkpoint like every other E9 arm, and is already implemented and tested. If the budget recompute
(§2.5 of the run plan) comes in under ~0.7 s/seq, adding `--arm_kwargs {"no_nu":true}` to the D-24
list is the cheapest way to keep the sentence you most want to keep. **This is the one place in the
pass where I would spend compute rather than cut text.**

### A-7 · line 2842 · App. reporting rules, efficiency

> "Efficiency is reported as parameters, FLOPs, peak memory and wall-clock latency *as a function of the measured relation coverage*"

**Unfunded, entirely.** No FLOP counter, no latency instrumentation, no parameter count anywhere in
`marsea/` or `scripts/`. `time_step.py` measures training step time and `profile_memory.py` peak
memory — neither is stratified by coverage, and neither is a FLOP or latency report.

**Proposed:** → "We report peak memory and training step time on the deployed configuration
(App. F); a full efficiency study stratified by relation coverage is not attempted here, and the
cost of materializing a relation's columns is proportional to that coverage by Sec. 2.3."

---

## Class B — invalidated by closing E6 this morning (6). **All must change.**

### B-1 · line 1055 (approx.) · §5, the E4–E9 paragraph
> "E6 tests the *ranked* half on IHEval, reporting accuracy against conflicting-tier count as a slope, not a pooled mean."

**Proposed:** delete the clause; renumber the sentence to run E4, E5, then E7.

### B-2 · §5, Datasets paragraph
> "and **IHEval** … supplies what evidence selection cannot, a *ranked* order over conflicting instruction tiers."

**Proposed:** delete IHEval from the sentence; the paragraph then names two datasets, which matches
what runs.

### B-3 · App. E4/E5/E6 paragraph — the whole **E6** block (prediction + falsification)

**Proposed:** replace with one honest sentence:

> "**E6** was to test the ranked half of Sec. 1 on IHEval, stratifying by the number of conflicting
> instruction tiers. We do not report it: IHEval places one conflicting instruction at one level of
> a fixed four-level hierarchy rather than sweeping a depth, so the stratification the design calls
> for is not available in the data. The ranked half of the motivation is therefore **not tested in
> this paper**."

### B-4 · App. reporting rules
> "IHEval is scored as task accuracy, and the depth stratification E6 reports is taken on its multi-turn rule-following task…"

**Proposed:** delete the sentence.

### B-5 · App. "Why these three datasets and not others"
> "RULER, MuSiQue and **IHEval** were selected because each supplies something the others cannot: … and a ranked order over conflicting sources respectively."

**Proposed:** → "RULER and MuSiQue were selected because each supplies something the other cannot: a
controlled `m` at constant difficulty, and a real task with annotated gold evidence." (Keep the
IHEval rationale in a single trailing sentence explaining why the third instrument was dropped, so
a reviewer sees the reasoning rather than a silent gap.)

### B-6 ⭐ · §1 and the abstract — **the framing decision, and it is yours, not mine**

The ranked half now has **no experiment behind it anywhere**. Three places lean on it: the traffic
intersection scene (§1, "One scene, both margins"); "which instruction outranks which when they
conflict" (§1); and "Competition may be **ranked** … or **graded**" (§1, definitional).

Two defensible positions:

- **(i) Keep it as motivation, mark it untested.** One clause in §5: "The ranked case is motivation
  only; every measurement in this paper is of the graded case." Honest, cheap, and a reviewer who
  reads carefully will still note the gap.
- **(ii) Re-weight the framing to the graded case** and demote ranked to a remark. More work, but it
  removes the mismatch between what the intro promises and what §5 delivers.

I lean **(i)** at three days out — (ii) touches the abstract, and abstract surgery this week is how
sentences get broken.

---

## Class C — funded, but weaker than the sentence implies (7). **Qualify, don't cut.**

### C-1 · line 2869–72 · E7's predicted PR trace
> "**E7 also reports the measured precision–recall trace against the predicted one** … sweeping `τ_j` on held-out columns should reproduce Fig. 2 column by column"

The trace *is* reproduced — in the **synthetic harness** (`verify_all.py`, `fig_pr_curve.py`,
Table 5's three trace rows). There is **no `τ_j` sweep on real data** in `run_eval.py`. The sentence
sits in the E7 paragraph, so it reads as a claim about trained columns.

**Proposed:** add four words — "…so sweeping `τ_j` on held-out columns should reproduce Fig. 2 column
by column; **we verify this in the harness of App. F and do not sweep a trained head's temperature**."

### C-2 · lines 2896–98 · E8's membership–`δ_j` correlation
> "and the correlation between each membership decision and the realized separation `δ_j` of the column it entered, which is how we learn whether the relation keys on the structure the theory says matters"

**Not computed.** `fidelity.py`'s key set is `rho, rho_head, frac_rows_zero_mass, regime_q10_50_90,
hist_Ecol, hist_Erow, frac_Ecol_singleton, var_tau_j, var_tau_i, tau_j_median, tau_i_median,
hist_kstar, frac_rows_over_unit, truncation_flag, row_trigger_rate, col_induced_rate,
row_trigger_violation, kstar_by_Ecol`. No `δ_j` correlate.

⭐ **Cheap to fund** — `δ_j` is already computed per column for the interval-hit rate, and `E` is in
hand, so this is one scalar per batch. **Recommend funding rather than cutting**, and if it is not
funded by D5, cut the clause.

### C-3 · line 1534 · E7 "reports attainability"
> "and E7 reports attainability rather than only the sign of the margin."

No attainability computation exists. But `fidelity.py:101` already records the **signed distance to
the nearer endpoint** when the interval is missed, which is the operative content.

**Proposed:** → "and E7 reports the signed distance from the realized `τ_j` to the nearer endpoint,
not only whether the interval was hit."

### C-4 · line 924 · the cross-family check
> "Qwen2.5-1.5B (Apache 2.0) primary, **Llama-3.2-3B as a cross-family check**"

`backbone.py` supports `SUPPORTED_FAMILIES = ("qwen2", "llama")` — the code is real. **No Llama run
exists in the D-24 grid**, and adding one is a second full programme, not an arm.

**Proposed:** → "Qwen2.5-1.5B (Apache 2.0); the implementation also supports Llama-3.2-3B and we
report a cross-family check **only if** the budget permits." ⚠️ Or, cleaner and what I would do:
drop Llama from the sentence entirely and state single-backbone scope in Limitations. A conditional
promise still reads as a promise to a reviewer.

### C-5 · line 968 · "Six [baselines] … retuned under a matched schedule"
True, but under D-24 **B1 and B4 get one seed** while MarSea/B0/B2/B3 get three.

**Proposed:** footnote — "B1 and B4 are run at a single seed; their comparisons are reported without
a variance estimate and are not used to support any claim."

### C-6 · line 2755 · matched-sparsity baseline
> "and softmax sparsified to MarSea's realized support size"

B5 is **evaluation-time and teacher-forced only** (D-29). The paper never says so.

**Proposed:** add — "…(evaluated teacher-forced only, since it needs MarSea's recorded support; its
task-level cell reads *n/a by design*)."

### C-7 · line 1060 and Table 6 · the compute figure
> "E1–E9 together are staged cheapest-first at **≈336** H100-hours"

D-24 reduces this to **≈150**, and the S4 row's 80 h included E6.

**Proposed:** update the number in both places; drop E6 from the S4 row; re-total. ⚠️ **Do this last,
after the budget recompute (§2.5 of the run plan), not now** — the 150 figure is itself unmeasured.

---

## Class D — checked and genuinely funded. **Leave alone.**

So you know what the pass covered rather than only what it flagged:

| claim | funded by |
|---|---|
| padding-length invariance (line 2748) | `test_t00_backbone.py::test_t16` — runs it with a live relation |
| interval-hit rate, and by kind | `evaluate.py:565–566` |
| relative interval width `m_jδ_j/W_j` | `fidelity.py:106` (`rel_width`) |
| the three-way distractor split (lines 346, 2798) | `fidelity.py:144–149`, incl. `excluded_argmax_elsewhere` |
| MESH arm's column-marginal residual | `baselines.py:9, 88` — logged, as the text says |
| `τ_i R̃_i/c̄_i` distribution about 1 (line 1523) | `regime_q10_50_90` |
| `k*` stratified by `|E_·j|` (line 1214) | `kstar_by_Ecol` |
| row-trigger ≥ column-induced rate | `row_trigger_rate`, `col_induced_rate`, `row_trigger_violation` |
| E3's coverage rate `|E_·j|/n` (line 908) | `hist_Ecol` |
| the kernel's `δ` and the θ-window as device statistics | `check_unit_cap.py`, App. F |
| "Every claim verifies at zero violations" | `verify_all.py` — 0 / 1,058,450 over 63 checks |
| all six baselines exist | `baselines.py:202` `ARMS = ("marsea","B0","B1","B2","B3","B4","B5")` |

---

## What I would do, in order

1. **Class B first** (6 edits). They are mechanical, they are wrong *today*, and E6's removal is
   already recorded everywhere else. ~1 hour.
2. **A-1 through A-5 and A-7** (6 edits). Deletions and re-aims, no judgement calls left in them.
   ~1 hour.
3. **A-6** — decide: cut the list to what runs, **or** add `no_nu` to the D-24 grid for ~5 GPU-h.
   ⭐ If the step-time measurement leaves any slack, fund it.
4. **C-2** — decide: add the `δ_j` correlate to `fidelity.py` (~10 lines), or cut the clause on D8.
5. **B-6** — the framing call. Yours.
6. **C-7** — last, after §2.5's budget recompute.
7. One more thing the pass does not cover and §1.6 of the run plan does: **the equivalence band for
   "flat"** has to be written into the E3 paragraph before any data exists. The draft currently says
   "a flat precision on the `δ_j>0` columns … is the theory's own prediction" without saying what
   counts as flat. ⚠️ That is not an unfunded claim — it is an **unfalsifiable** one, which is worse.

Line 1067 — "**Apart from E1, Sec. 5 has not been run.**" — stays true until D8 and then must be
replaced with what did run. Leave it until last; it is the honest state today.
