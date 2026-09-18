# MarSea — the experiment programme: E1–E9, S0–S4, and which claim rests on which

*2026-09-15. Written to decide the scope question ten days out from the paper deadline. Sources:
`marsea_iclr.tex` §5 and App. H, `tab:compute`, record §24.2, §36.2, §40, §44 (D-24), §53.*

---

## 1. The stages, and what each costs

| stage | contents | gate | pivot if the gate fails | H100-h |
|---|---|---|---|---|
| **S0** | E1 re-run on **trained-backbone** score columns; the S0 head sweep that licenses E7's column constructions | a replication pair exists: `M > m_j` at a stated `a` | Cor. `capacity` does not bite on real columns — the separation argument loses its bridge to data | **1** |
| **S1** | mechanism correctness end to end: identities, hierarchical-vs-flat agreement, relation logging, a real training run | identities to tolerance; relation coverage non-zero | implementation defect; nothing downstream means anything | **15** |
| **S2** | **E3** (the *n*-sweep, both controls) + **E2** + **E4**, on RULER | precision flat in `n` on `δ_j>0` columns; hit rate not falling in `m` | report the training-effect claim and rescope the paper | **60** |
| **S3** | **E5** (MuSiQue, HotpotQA) + **E7** (fidelity) + **E8** (diagnostics) | — | — | **68** |
| **S4** | ~~**E6** (IHEval depth)~~ **NOT RUN (2026-09-16)** + **E9** (ablations) | — | — | **80** |
| | subtotal 224; ×1.5 debugging allowance | | | **≈336** |

D-24's reduced programme cuts this to ≈150 H100-h of actual work — 3 seeds on MarSea/B0/B2/B3,
1 on B1/B4, E9 down to three ablations — but **8–10 days end to end**, because the binding
constraint is the serial chain (data → Phase A → S0 → Phase B → eval queue → collect), not the
GPU-hours. 150 H100-h across 8 GPUs is under a day of pure compute.

---

## 2. The experiments in full

### E1 — the replication probe · **already run** (synthetic) · re-run at **S0** on real columns

**Panel (a).** True relation size held at `m_j = 4`; candidate set `|E_·j|` grown 4 → 256. MarSea's
fan-out degree is **4.00 at every size** (slope −0.0000); any full-support scheme reaches all
`|E_·j|` of them (slope +1.0000). Precision 1.000 against 1.000, 0.500, 0.250, 0.125, 0.062,
0.031, 0.016.

**Panel (b).** Six keys, true sizes 1–32 at `n_q = 63`. Under a per-key `τ_j` inside Thm. 2's
interval `ν_j` tracks `m_j` (1.00, 1.59, 3.20, 6.68, 13.95, 28.21); under one global `τ` it
saturates (1.34, 2.09, 3.66, 5.82, 8.11, 11.48) — a factor of 32 compressed into 8.6.

**Claims supported:** Cor. `capacity`'s **cardinality half** (a query-local scheme cannot bound
fan-out degree); Prop. `fullsupportceiling` (the comparator's precision is the *identity*
`m_j/|E_·j|`, not a tendency); Cor. `necessity` in training-free form (a global temperature cannot
serve two keys with disjoint intervals).

⚠️ **What it is not.** `E` and `τ_j` are set by hand and Stage 2 is inert. It is a **possibility**
result: it shows a column-coupled mechanism *can* express what query-local schemes provably cannot.
It says nothing about whether a *trained* `τᴷ` finds it. **That gap is the entire empirical burden
of the paper**, and S0/S2 are what close it.

---

### E2 — the *m*-sweep at fixed *n* · **S2** · a Thm. 2 test

RULER, `m ∈ {1,2,4,8,16}` at fixed `n` and fixed sequence length, 8K.

⚠️ **An earlier draft called this load-bearing and read a growing precision gap as evidence for
Thm. 1. That reading is arithmetically inverted** (§36.2): the comparator's precision is the
identity `m/n`, which *rises* with `m` at fixed `n`, so MarSea's gap `1 − m/n` *shrinks*. What
growing `m` actually stresses is the recovery interval, whose endpoints both scale like `1/m` — the
head must land a narrower absolute target.

**Reports:** MarSea's own precision, exact-set accuracy, and the **interval-hit rate stratified by
`m`**. **Falsification:** a hit rate falling with `m`.

**Claims supported:** Thm. 2 (the sharp recovery interval) on real score columns; indirectly
Cor. `necessity`, since a content-predicted `τ_j` is what makes a per-column target reachable.
The comparator arm carries no information here beyond confirming the identity.

---

### E3 — the *n*-sweep at fixed *m* · **S2** · ⭐ **LOAD-BEARING**

Hold `m`, sweep `n` over five levels up to a few hundred candidates, at a **fixed 16K length** for
every point (the control requires constant `L`; this is what puts the chunked path on the critical
path). This is Fig. 2(a) on real data.

**Predicted:** the comparator's degree grows as `n` and its precision falls as `m/n`
(Cor. `capacity` + Prop. `fullsupportceiling`); MarSea's degree stays at `k* = m` **on the columns
where Assumption 2 holds**. That qualifier is the test — App. E predicts the assumption fails on a
growing fraction of columns as `n` grows (63.6 %, 14.9 %, 1.2 % recovery at 32, 128, 512).

**Reports per `n`:** the fraction of active columns with `δ_j > 0`; the interval-hit rate on those;
MarSea's precision on those **against on all**; supporting-fact precision/recall and answer F1;
the measured relation-coverage rate `|E_·j|/n`.

**Two mandatory controls, and the result is worthless without both:** every point run twice against
inert padding of equal token count (Du et al. report 13.9–85 % degradation from length alone), and
gold at a fixed relative position with the position-marginal profile reported alongside
(Chowdhury's U-shape).

**Falsification, pre-committed both ways:** precision *flat* on the `δ_j>0` columns beside a falling
fraction of such columns is the theory's own prediction; precision flat on **all** columns would
mean the relation is doing the selection rather than the program.

**Claims supported:** the paper's **headline falsifiable prediction** — that evidence selection
degrades with candidate count for architectural reasons and MarSea's does not. Everything in §4
lands here.

---

### E4 — the single-target control, `m = 1` · **S2**

RULER S-NIAH. The regime **Prop. `scope`(i) explicitly concedes** to query-local schemes: selection
at a single target *is* available to row-sparsemax at large temperature.

**Prediction:** a gap near zero. ⭐ **A *large* gap is the interesting negative** — it would mean
the benefit is not coming from multivariate structure, and E2/E3's reading would need revision.

**Claims supported:** it protects the separation claim from the commonest reviewer objection —
"your gain is just a better-tuned sparse attention". Cheap, and it is the control that makes E3
mean what the paper says it means.

---

### E5 — externalisation · **S3**

MuSiQue-Answerable (`m ∈ {2,3,4}` by hop) and HotpotQA-distractor. Reports each benchmark's own
**precision-sensitive** criteria: supporting-fact EM/F1 and the *joint* metrics on HotpotQA,
support F1 beside answer F1 on MuSiQue.

⚠️ Its `m`-stratification is **confounded with difficulty by construction** (MuSiQue varies gold
count and reasoning depth together), so it **corroborates E2 and never replaces it**.

**Claims supported:** that the mechanism does something on a **real task**, not only on a
configurable synthetic one. ⭐ **This is the experiment whose absence hurts most in a reviewer's
eyes**, because the paper itself cites `yen2025helmet` for the point that synthetic recall does not
predict downstream performance. Without E5 the empirical case rests entirely on an instrument the
paper warns the reader not to trust alone.

---

### E6 — the ranked claim · **S4** · ❌ **NOT RUN — closed 2026-09-16**

IHEval, stratified by the number of conflicting instruction tiers, reporting accuracy **as a
function of depth — the slope, not a pooled mean**, on its multi-turn rule-following task.

**Prediction:** at depth one (two sources, one overriding) MarSea and a well-tuned query-local
baseline are close, a two-level decision being within a pointwise rule's reach; as depth grows the
query-local baselines degrade markedly and MarSea gently. **Falsification:** a flat gap across
depth withdraws the **ranked** framing for the weaker binary-exclusivity claim.

**Claims supported:** the **ranked** half of §1's motivation — the traffic intersection, preemption,
"an officer outranks a light". Without it the paper's opening scene is motivated but never tested,
and the intro's two-margin story rests on the graded half alone.

❌ **CLOSED 2026-09-16: NOT RUN, and not because it was hard to acquire.** IHEval's records carry
no tier count — and cannot, since the benchmark places **one** conflicting instruction at **one**
level of a *fixed* four-level hierarchy rather than sweeping a depth. **E6's x-axis does not exist
in the data.** The loader also mismatches on prompt fields, and `answer` is a dict of IFEval-style
verifiable constraints rather than a reference string, so the string-match scoring in `run_e6.py`
is ~0 for every arm. Re-aiming it at the *placement* axis with vendored IFEval checkers is about a
day, on a 1.5B base checkpoint that may floor out; declined three days before the abstract.

⭐ **Consequence for the paper, stated plainly:** the ranked half of §1's motivation now has **no
experiment behind it at all**. Either the opening scene is re-written to claim only what E3 and E7
support, or it stands as an explicitly untested motivation. It cannot stand as a promise.

---

### E7 — fidelity, both margins · **S3** · ⭐ carries the *sharpest* test

Wherever gold evidence is annotated, on the two column kinds D-9a separates (coreference columns at
the head S0 finds; value columns where the answer rows are scored):

- **all five quantities of Def. `fidelity`**, never a single aggregate, and **recall twice** —
  before and after Stage 2, since Prop. `stagecomp`(i) says recall cannot rise through Stage 2 so
  any deficit between the readings is Stage 2's;
- the distribution of the **relative interval width** `m_jδ_j/W_j` — what a hard column costs is
  width, not feasibility;
- the **measured precision–recall trace** against Fig. 1's predicted L;
- ⭐⭐ **the interval-hit rate**: the fraction of active columns whose realized `τ_j` lands in
  `[1/(W_j+m_jδ_j), 1/W_j)`, with the signed distance to the nearer endpoint when it does not.

⭐ **The paper calls the hit rate "the most direct test of Thm. 2 available on real data" and the
one genuinely empirical quantity of the pair** — the trace is an identity given Assumption 2 and
α = 2, so a departure indicts the assumption or the implementation, never the theorem. The hit rate
is the only number the theory does *not* already determine. **Falsification:** a hit rate at chance
with the misses unbiased in sign says the head is not using the column and the mechanism is running
on a temperature that happens to work.

**Claims supported:** Thm. 2 on real data; **contribution (i)** — that a head reading the
participant *together with its field* is necessary — because B3 (`τᴷ(k_j)` with the column
withheld) is scored **here**; Prop. `stagecomp`'s ratchet; Prop. `graded`'s rates.

---

### E8 — mechanism diagnostics · **S3**

Relation coverage on both margins **from epoch one**; the full `|E_·j|` and `|E_i·|` **distributions,
not their means**; realized `k*` stratified by `|E_·j|`; the one fan-in trigger (a violation is an
implementation bug, not a finding); the distribution of `τ_i R̃_i/c̄_i` about 1; the correlation
between each membership decision and the realized `δ_j`; the three-way split of each answer row's
distractor keys (excluded by Stage 1 / rejected by the row / surviving).

**Claims supported:** it is the **degeneracy detector**, and the paper is explicit that it has a
detector rather than a training scheme that avoids the degenerate optimum. Two collapses must be
told apart: an empty relation everywhere (= standard attention) and `τ_j → 0` on a non-empty
relation (the inherited quota spread uniformly — *not* standard attention). App. E cost (a)'s
`|E_·j| = 1` degeneracy is invisible to the collapse variances and visible only here.
**Falsification:** if `k*` on covered columns approaches the deployed `K`, the hierarchy is silently
truncating and **no E7 number is meaningful** until `K` is raised.

---

### E9 — ablations · **S4**

Everything in turn: the relation's parameterization (rank `r`, pairwise against key-alone, which
layers), `K`, block size, emission policy, `τ_i` learned against pinned at 1 (which reduces Stage 2
to a plain unit cap), `τ_j` learned against fixed or uniform, and the inherited capacity against a
uniform one (the arm that puts a predicted budget back).

⭐⭐ **The load-bearing ablation of the group strips the field argument from each head** —
`τᴷ(k_j)` against `τᴷ(k_j, s_·j, ν_j)`; `e_ij` from `u_φ(k_j)` alone against the pair; `τᴷ` with and
without `ν_j`. The paper's own words: *"a mechanism that claims to model a multivariate relation
should lose measurably when its parameters are made blind to the field, and if it does not, the
claim is decoration."*

**Claims supported:** **contribution (i)** — since §32.0 this *is* the novel object. After MESH was
found to learn both marginals, the paper no longer claims the learned column marginal; it claims
the marginal predicted from the participant **together with its field**. E9's field arm and B3 are
the only two tests of it.

D-24 reduces E9 to three arms: `τ_i` pinned, uniform quota, key-alone relation. ⚠️ Note that the
key-alone *relation* arm is not the same as the key-alone *`τᴷ`* arm — B3 is the latter, and it is
scored on E7.

---

## 3. The other direction: for each claim, what carries it

| claim | primary evidence | secondary | if that evidence is missing |
|---|---|---|---|
| **Thm. 1** — query-local schemes attain exactly a product set | S0's replication pair (`M > m_j`) | E1(a) | the theorem is true but its bite on real data is asserted |
| **Cor. `capacity`**, cardinality half | **E3** | E1(a) | the headline prediction is untested |
| **Prop. `fullsupportceiling`** — `P_j = m_j/n` exactly | E3 (it is an identity; the arm confirms it) | E1(a), E5 | low risk: it is arithmetic, not a measurement |
| **Thm. 2** — the sharp interval | ⭐ **E7's interval-hit rate** | E2 stratified by `m` | the theory's one empirical quantity is unmeasured |
| **Cor. `necessity`** — content-dependent `τ_j` | E7 hit rate vs **B3**; **E9's field arm** | E1(b), training-free | contribution (i) is untested |
| **contribution (i)** — participant *and* field | ⭐ **E9 field arm + B3 on E7** | E1(b) | ⚠️ the paper's novel object has no evidence |
| **shape vs capacity** (against MESH) | **B2** on E3's precision | Prop. `fullsupportceiling` | partly covered in scope B |
| **the ranked half** of the motivation | ❌ **E6 NOT RUN** (2026-09-16) | — | ⚠️ **nothing** — the intro's scene must be re-written or marked untested |
| **no degenerate collapse** | **E8** | — | the paper's own stated risk is unmonitored |
| **real-task transfer** | **E5** | — | the case rests on synthetic data the paper distrusts |

---

## 4. What each scope buys

| scope | stages | experiments | wall clock | what the paper can claim |
|---|---|---|---|---|
| **C** | none | E1 only | 0 | theory + a training-free possibility result |
| **B** | S0, S1, S2 | E2, E3, E4 | ~4–5 days | the **separation claim tested on data**, with both controls; Thm. 2 partly, via E2's hit rate |
| **B+** | B + E5 and E7 from S3 + the one E9 field arm | + E5, E7, E9(field) | ~6–7 days | the above **plus** a real task, the sharpest test of Thm. 2, and contribution (i) |
| **A** | S0–S4 | all | 8–10 days | what the draft currently promises |

⭐ **B+ is the interesting one**, and it is not much more than B: E7 rides on checkpoints S2 has
already produced (it is an evaluation pass over logged scores, not new training), and the E9 field
arm is one additional Phase-B run. E5 needs MuSiQue prepared but reuses the same arms. The reason
`tab:compute` prices S3 at 68 h is that it includes E8 across the full grid and E5 on both
datasets — a targeted subset is much cheaper than the stage.

---

## 5. The honest read on acceptance

A judgement, not a measurement. ICLR's base rate is roughly 30 %.

- **Scope C** (theory + E1): well below base rate at ICLR. The theory is genuinely strong — a
  separation theorem with an exact attainable-set characterisation, a recovery theorem with **sharp**
  endpoints, and a comparator whose precision is an identity rather than a tendency — but "no
  experiments" is close to disqualifying for a mechanism paper at a conference. This is the
  configuration the record's own hostile reviewer called *"a much stronger TMLR submission than an
  ICLR one"*.
- **Scope B**: roughly **base rate, 25–35 %**. You would have the headline prediction tested with
  proper controls, which is real. Against it: one synthetic benchmark, one backbone, and —
  decisively — **contribution (i) has no evidence at all**, because its two tests (E9's field arm,
  B3 scored on E7) both live outside B. A reviewer who reads the contributions and then looks for
  the experiment that tests the novel object will not find one.
- **Scope B+**: **45–60 %**, and this is the first scope where I would call the paper *complete
  rather than truncated*. Every numbered contribution has an experiment behind it, the empirical
  case is not resting on synthetic data alone, and the one quantity the theory does not determine is
  measured.
- **Scope A**: 50–65 %. The marginal gain over B+ was E6, E8 and the remaining ablations —
  genuinely valuable, but not what a reviewer's accept/reject hinges on. ⚠️ **With E6 closed as
  NOT RUN (2026-09-16), scope A and B+ now differ only by E8 and the ablations**, which narrows
  the gap between them and makes the text pass, not the extra arms, the thing separating them.

⚠️ **The text matters as much as the scope.** In B or B+ the draft's ~11 unfunded promises become
false statements in a submitted paper. Cutting or qualifying them is a day of careful work and it is
worth more than any single extra experiment: a reviewer who checks one promise and finds nothing
behind it discounts everything else.
