# MarSea — fifth adversarial audit: the two-step selective form

*2026-09-02. Draft audited: 39 pp with the two-step form as the mechanism. Method: two
independent agents — one implemented the mechanism cold from Sec. 2 + the spec and tested the
invariants; one read the experiments as a hostile ICLR reviewer — plus my own pass. Findings
verified by hand before listing.*

**Headline: the design as written two turns ago had two real bugs and a false claim, and most
of what follows is against work from this session.** Both bugs are fixed and verified; the
experiment-design findings are partly fixed and partly yours to decide (Tier C).

---

## Tier A — design bugs, fixed and verified

### A1 ⭐ The equality-constrained row step resurrects Stage-1 zeros

`eq:rowprog` as written had `Σ_{j∈𝓔ᵢ.} aⱼ = c'ᵢ`. For `τᵢ` small enough the target `c'ᵢ/τᵢ`
exceeds `Σ_𝓔 Ãᵢⱼ`, the dual `θᵢ` goes **negative**, and every coordinate of the relation —
including entries Stage 1 set to exactly zero — becomes positive. That contradicts
`prop:stagecomp(iv)` ("a Stage-1 false negative is unrecoverable"), which is load-bearing.
Measured: **314/600** random draws violate INV-6/INV-7. The cold implementer's minimal case:
`Ã = [0.719, 0.036, 0.000]`, `τᵢ = 0.5` → `A = [0.485, 0.144, 0.126]`, `θ = −0.25`.

⚠️ **My earlier check missed this** because it tested only the row sum and the complement, not
zero-permanence; and `verify_all.py`'s `stagecomp` block tested a *hand-drawn* non-negative floor,
never `eq:rowprog` itself. The spec's claim that "each invariant is already checked in
verify_all.py" was false for INV-6/7.

**Fix applied**: step 2 is a **cap**, `Σ_{𝓔ᵢ.} aⱼ ≤ c'ᵢ`. The projection onto an inequality set
has `θᵢ ≥ 0` for every `τᵢ > 0` (KKT multiplier of an inequality), so zeros are permanent and the
quota is a ceiling. This also restores the two-regime story on the row (slack below
`c'ᵢ/R̃ᵢ`, binds above) with the sub-unit-via-`τᵢ` property Luke chose `≤` to keep. The
alternative — keep the equality and constrain `τᵢ ≥ 1` — is coherent but removes the slack regime
entirely; I took the cap and the paper says why. **0/80,741 after the fix.**

### A2 ⭐ Column-softmax step 1 made "gates closed = standard attention" false

`eq:step1` had `Ā.ⱼ = cⱼ · softmax(s.ⱼ)` — a *column* softmax rescaled to the row-softmax column
mass. That preserves the column **total** but not the **entries**: `Āᵢⱼ ≠ Aˢᵐᵢⱼ`, max
entrywise gap 5.34 in the cold test, `Ā₁₁ = 1.216` for a single attention weight. Consequences,
all measured: with every relation empty the output differs from standard attention on
**600/600** draws; row supply before step 2 is *not* one (range 0.41–1.98); the step-1 cap
binds on 44% of rows with no relation at all and produces exact zeros; and Luke's own words
("the rest of the column is subject to standard attention without constraints") were false of
the formula.

**Fix applied**: `Ā.ⱼ := Aˢᵐ.ⱼ`. Every stated motivation — commensurate columns, inherited
capacity, `Σⱼcⱼ = n_q` derived — holds identically, and now also: gates-closed **is** standard
attention entrywise (max diff 1e-16), row supply is exactly one before step 2, and step 1 costs
nothing because the layer already computed it. **0/600 on all nine invariants after both fixes;
0/2,000 on the entrywise identity in the harness.**

### A3 "R̃ᵢ is exactly one unit's worth of supply" — my sentence, false

Written in the cleanup pass. Column totals are conserved, not row totals; after fan-out
concentration `R̃ᵢ` ranges widely. Rewritten: `R̃ᵢ` is now the *relation's* supply on the row,
the regime threshold is `c'ᵢ/R̃ᵢ`, and **`τᵢ = 1` is not an identity element** — the row's
identity is content-dependent, and E8 logs `τᵢR̃ᵢ/c'ᵢ` about 1.

### A4 Two incompatible fan-in programs in one paragraph

The old prose ("Substituting `u = a/τᵢ`… `Σ uⱼ ≤ 1/τᵢ`", "`Σⱼ Aᵢⱼ = min(τᵢR̃ᵢ, 1)`", "for
`τᵢ ≤ 1/R̃ᵢ` the cap is slack") survived my equation rewrite and contradicted it. The whole
subsection is rewritten around the cap form.

### A5 `v1_probe.py` panel (b) used **negative temperatures** on 3 of 72 columns

Where Assumption 2 failed on a drawn column, `lo = 1/(W + mδ)` was negative and the script used
`τ = −6.04` and `τ = −18.50`. That, not "targets not exactly tied," was the source of the
reported deficits. Fixed: such columns are re-clipped and counted; the script prints how many.
Panel (b) values moved slightly (1.59, 3.20, 28.21 vs 1.55, 3.07, 27.61) and the paper is updated.

---

## Tier B — consistency, fixed

- **Proofs were missing.** Sec. 3 says every claim is proved in App. F; `prop:twostep` and
  `prop:scopemono` had none. Both now proved.
- **The gates were redundant with the relation** and the query gate was *unreachable* — with `𝓔`
  as one object, `𝓔ᵢ. ≠ ∅` iff some key's relation contains `i`, so the "disjunction trigger" is
  one condition and `ω^Q` changes nothing (max |ΔA| = 0.0 over 400 draws). Gates removed;
  Sec. 2.3 is now "Conditional activation is the relation." Five components remain: two stages,
  relation, hierarchy, causal form.
- **`def:fidelity`'s row support was over all keys**, so `Pᵢ` was pinned by the dense
  off-relation entries (measured 0.451 as written vs 0.775 on the relation). Now on `𝓔ᵢ.`, and
  **relation recall** `|Tⱼ ∩ 𝓔.ⱼ|/mⱼ` is a named, separately reported quantity — a target the mask
  excludes is a recall miss before any program runs and must never be charged to a temperature.
- **S0's gate could never pass**: `Ma > cⱼ` with `cⱼ = Σᵢ Aˢᵐᵢⱼ ≥ Ma` always. Now `M > mⱼ` at a
  stated `a`.
- **App. H's E1 paragraph described the retired mass experiment** — Sinkformer, gate mean, `n_k=6`,
  `cⱼ=1`, none of which exist in the shipped script. Rewritten, and it now says plainly that
  MarSea's `τⱼ` is an **oracle** and both curves are identities: the figure is the shape of the
  guarantee, not a measurement of a head.
- Residual predicted-budget language at eight sites (`tab:stages` routing `νⱼ` into `cⱼ`; App. B's
  per-block normalization; App. D's "collapse in `cⱼ`"; Prop. 12's "budgets satisfying"; the
  "budget–cardinality residual"; the pigeonhole "over active keys"; `νⱼ` init).
- `νⱼ` initialised at 1, not 0 (0 is outside its stated range `[1, |𝓔.ⱼ|]`).
- Harness: 690,483 → **853,965** checks, 0 violations; both drafts' tables reconcile exactly.

---

## Tier C — experiment design; partly fixed, partly yours

### C1 ⭐ E2's decision rule was arithmetically inverted — fixed, but read it

"A gap that grows with `m` is evidence Thm. 1 is doing work." By Prop. 14 the comparator's
precision is `m/n`, which **rises** with `m` at fixed `n`; so MarSea's gap `1 − m/n` **shrinks**
with `m` if MarSea is at ceiling, and could only grow if MarSea *failed* at small `m`. The S2 gate
would have killed the programme on the theory's own success signature. What growing `m` actually
stresses is Thm. 2 — both interval endpoints scale like `1/m`. **Rewritten**: E2 is a Thm. 2 test
reporting the interval-hit rate stratified by `m`; **E3 is now load-bearing** and is Fig. 2(a) on
real data, reporting per `n` the fraction of columns with `δⱼ > 0` and precision on those vs all.

### C2 ⭐⭐ The column theory has no ground truth in multi-hop QA — **yours to decide**

The hostile reviewer's strongest finding, and I think it is right. In a causal decoder, "which
`m` of `n` passages support the answer" is the answer query's **row** — `Kᵢ`. The row step is
query-local (App. D says so), so **Thm. 1 does not speak to the headline metric**, and a
row-entmax with a learned temperature controls its own membership completely. On the column side,
gold gives `Kᵢ`, not `Tⱼ`; with one answer query, `mⱼ ∈ {0, 1}` for every passage key — `mⱼ = 1`
is the regime `prop:scope(i)` concedes, and at `mⱼ = 0` Thm. 2 is undefined and `Pⱼ = 0` is
forced (sparsemax always has ≥ 1 atom). **E7's interval-hit rate — "the sharpest single test of
the theory" — cannot be run on the data the paper chose.**

What exercises the column theory is a key that governs **many annotated queries**: coreference
(antecedent → mention set), a signal → the lanes it governs, a variable → its references. The
TMLR draft's vision setting has this; the ICLR draft dropped it. Options: (a) annotate `Tⱼ` at
the token level for QA and say how; (b) keep QA as the row test, say so, and add one typed task
for the column; (c) reframe the ICLR contribution around the row margin plus the synthetic column
results. I would not choose this for you — it decides what the paper is a paper about.

### C3 Inherited capacity means a key can never be silenced or amplified — stated now

A distractor key keeps its full softmax mass; a large `τⱼ` concentrates it onto fewer members
rather than shrinking it. The motivating story ("the highest-priority one renders the rest
irrelevant") requires losers to emit *less*; here losers lose reach, not mass, and suppression is
delegated to the row stage. Now stated in Sec. 2.1. An experiment that exposes it: per key class,
`ΣᵢÃᵢⱼ` vs `ΣᵢAˢᵐᵢⱼ` (identical by construction — say so), and the answer row's incoming
distractor mass before Stage 2.

### C4 B5/B6's disqualification argument is wrong on the row margin

"By Thm. 1 their support membership is uncontrolled" — true on columns, false on the row, where a
query-local top-k controls its membership fully. On row metrics the paper offers no prediction
separating MarSea from row-entmax with a learned temperature except *which keys Stage 1 let reach
the row*. Now stated in Sec. 2.2's honest notes; the baseline paragraph in App. H still needs the
matching correction and a direct measurement of the Stage-1-attributable row-support difference.

### C5 Aggregation unspecified

Nothing says which layers carry MarSea, which head defines the row `Sᵢ`, or how a passage's
support is formed from its token keys (union over tokens). E3 now points at App. H for this, but
App. H does not yet contain it. One paragraph, needed before any run.

### C6 "Constant task difficulty" in E2 (plausible)

RULER's own multi-key results degrade with needle count on the exact-set criterion. Needs a
citation or a per-`m` ceiling control.

---

## What survived an attempt to break it

Thm. 1 and Thm. 2 unchanged; `prop:scopemono` (monotonicity, containment, rescue) verified
0/6,000 and now proved; the fixed-relation regime (quota flat at slope −0.0001 while the column
grows +0.283); the degree bound as the exclusive set grows (0/10,000); `τᵢ` direction under
`a/τᵢ` (support 12 → 1, matching the prose); `Σⱼcⱼ = n_q` derived; all 39 bibliography entries;
every label the spec cites resolves.
