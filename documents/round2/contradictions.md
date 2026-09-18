# MarSea — contradictions across spec v4 / paper / master record / harness, and residual ambiguities

Sources: `spec_v4.md` (spec), `marsea_iclr.tex` (paper, line numbers from the file), `master_record.md`
(record), `verify_all.py` (harness), `reference/` (ref). Severity: **HIGH** = a literal reader
produces a wrong run or wrong headline number; **MED** = wrong on a reportable quantity or a
paper-level inconsistency a reviewer will find; **LOW** = cosmetic or bounded by tolerance.
"Followed" = what `pseudocode.md` implements.

## A. Contradictions, ranked

### C-1 [HIGH] Spec's E2 RULER config is self-contradictory
Spec §13.1 L743: `MV-NIAH (E2): niah_multivalue essay / words / numbers, 1 / m / 1, m in {1,2,4,8,16}, n = num_needle_k = 32`.
`num_needle_k` cannot be both 1 and 32. Stock RULER `niah.py` (fetched; L132-142, L186-188): every one
of `num_needle_k` keys carries `num_needle_v` values; `num_needle_q` keys are queried; the answer is the
queried keys' values. So "m needles among n candidates at fixed n" is `K = 32, V = m, Q = 1`.
**Followed:** `K=32, V=m, Q=1` for E2; `K=n, V=4, Q=1` for E3 (which is what the spec's E3 line already says).
Consequence flagged as R-2: at `m = 16, n = 32` that is 512 needle sentences (~7-8K tokens) — likely infeasible at `L = 8K`.

### C-2 [HIGH] "Needle depth is a generator parameter" — it is not
Spec §13.1 L749-750: "needle depth is a generator parameter; hold it at 0.5 and ... sweep {0.1,...,0.9}".
Paper L997-999 makes fixed gold position a *mandatory* E3 control. Stock `niah.py` L105/L157 draws each
needle's depth with `random.sample(DEPTHS, len(needles))`; there is no depth argument.
**Followed:** patch `niah.py` with `--gold_depth` (queried key's needles at depth d, distractors random); the
patch is part of the deliverable and must be shipped with the code.

### C-3 [HIGH] RULER jsonl `input` has the answer prefix stripped
Spec §13.1 L745: "Output jsonl fields: index, input, outputs (list of gold values), length". `niah.py`
L277-291 cuts `answer_prefix` ("The special magic numbers for X mentioned in the provided text are") out
of `input` and stores it as its own field (plus `token_position_answer`). Prompting with `input` alone changes
the task and misaligns every gold-position search.
**Followed:** prompt = `input + answer_prefix`; gold = `" " + ", ".join(outputs)`.

### C-4 [HIGH] B2 / MESH: four versions of the baseline
- Spec §8 (v4): `eps = 1`, MESH step = 4 normalised-gradient descents on plan entropy with 5-iter inner
  Sinkhorn, **straight-through** (not differentiated), 20-iter outer Sinkhorn **ending on the row scaling**,
  `m = n_real_q`, `h_a` on `LN(K_kv)`, log the column residual.
- Reference `marsea_cold.py::MESHNorm` (v3): `eps = 0.1·std` per slice, descent lr 0.1, gradient **through**
  the unrolled descent (`create_graph`), 30 iters ending on the **column** step, `m = n_q` incl. pad rows, no LN.
- Paper L1243-1245: "the strength of the entropy term is a fixed global hyperparameter"; L2692-2695 states the
  causal infeasibility, the row-ending and the residual (added in v4, record §42).
- `test_cold.py` L249-250 asserts both `row_sum == b` and `col_sum == a` at 1e-3 — the second is expected to
  fail under a causal mask (its label says so) and must not be ported as a pass/fail test.
**Followed:** spec v4 (pseudocode §6 B2). T9 must show B2's column precision at the ceiling.

### C-5 [HIGH-for-interpretation] "`tau_i = 1` is not an identity element" vs "pinned at 1 reduces Stage 2 to a plain unit cap"
Paper Sec. 2.2 L458-459: "Since `c̄_i ≤ R̃_i` always, `τ_i = 1` lies in the binding regime ... and is not an
identity element; the row's identity is `τ_i = c̄_i/R̃_i`"; spec §5.4 repeats it. Paper App. H L2830-2831:
"`τ_i` ... pinned at 1 (which reduces Stage 2 to a plain unit cap)"; notation table L1111: "identity at 1 on a
unit-supply row". **Verified numerically with the harness's own primitives** (26,151 random rows, 18,286 with the
cap binding): at `τ_i = 1`, `A[i,E] == a1[i,E]` to 4e-16 on every row. Reason: when the cap binds, `c̄_i = Σ_E [Ã−th]_+`
with `th` the cap's dual, and `proj_le(Ã_E, c̄_i)` has the same unique threshold, so `θ_i = th` and `u = a1_E`.
So `τ_i = 1` **is** the identity relative to step-1's output — `θ_i > 0` there, but it is the cap's own dual, and
step 2 changes nothing. The "identity" at `τ_i = c̄_i/R̃_i` is instead the boundary of the *slack* regime relative
to `Ã` (`A_E = τ_i Ã_E`, a rescaling, not an identity either).
**Followed:** spec's `TauQ` init at `τ_i = 1` (it is exactly "start at the step-1 row", the intended standard-attention
start); E8 logs both `τ_i R̃_i/c̄_i` and `τ_i` about 1; the E9 "pinned at 1" arm is read as "no step 2" (App. H is right,
Sec. 2.2's sentence should be softened — Luke's call, paper-level).

### C-6 [HIGH] `vis` derivation from the 4-D additive mask is version-fragile
Spec §6.1: "`attention_mask` reaching a patched layer must be the additive 4-D mask ... assert its minimum is
`finfo.min` or `-inf`. Derive `vis = mask > -1e30`." With `attn_implementation="sdpa"` at load (as the spec
requires), transformers 4.48-4.52 `_update_causal_mask` returns `None` for un-padded causal inputs, and ≥ 4.53's
mask interface hands SDPA layers a **boolean** mask or `None`; neither satisfies the assertion.
**Followed:** build `vis` inside the patched layer from the 2-D padding mask + causal KV offset (exact for right
padding and KV-cache decoding); assert consistency with whatever 4-D mask arrives. `S = QKᵀ·scale` then
`masked_fill(~vis, -inf)` replaces the additive mask (identical softmax; see C-26).

### C-7 [MED] Relation form: paper `1[<u,v> > 0]` vs spec `<u,v>/√r + b0`
Paper L491, L1079, L1725 have no bias and no `1/√r`; App. E L1721-1731 argues the pairwise form is *required*
for arrival-sealing. Spec §4.1 adds `b0` (trainable, calibrated to `ρ₀ = 0.05`) and `1/√r`, justified as a
constant coordinate appended to `u` (still a function of the pair). ref `ambiguities.md` #18 flagged the same.
**Followed:** spec. The paper should say so in one clause (paper-level).

### C-8 [MED] Spec cites a paper statement that does not exist
Spec §4.3: the biased straight-through estimator "is stated in the paper's limitations". The paper contains no
mention of straight-through, the estimator, its bias, LoRA, Phase A/B, or `ρ₀` (grep: 0 hits for
`straight|biased|estimator|LoRA|Phase`; L1718 says only "initialises the relation near-empty").
**Followed:** spec for code; recorded as a paper gap.

### C-9 [MED] Where `ν_j` comes from, and whether it carries gradient across layers
Paper L1511-1513: "taken from the previous block of the same key (or the previous layer in an encoder)";
L383-385: "enters the next block or the next layer's head". Spec §5.3: previous **layer** in the full-sequence
(training) form, previous block only in the streaming form. Gradient: spec §5 L315 writes `state.nu_next = nu`
(no detach); paper L381-382 chooses `ν_j` over `|supp|` precisely because it "carries a gradient"; the reference
detaches (`state.nu_next = nu.detach()`).
**Followed:** spec + paper — previous patched layer, **not detached** (gradient from layer `l+1`'s `TauK` flows
into layer `l`'s `p`). Cheap (p is already alive). If instability appears, detaching is the E9 fallback.

### C-10 [MED] On which domain is a dense arm's precision "pinned"?
Paper L862 `m/n`; L930-931, L993-994 `m_j/|E_.j|` for MESH/the comparator; L1591 "`m_j/|E_.j|` for a *masked*
comparator"; L2806 `(1, m_j/n_q)`; spec T9 "every dense arm ... at exactly `m_j/|E_.j|`". Dense arms have no
relation, so `|E_.j|` can only mean MarSea's relation from the paired run.
**Followed:** score dense arms on both domains — the full visible column (`m_j/n_vis_j`) and MarSea's recorded
relation (matched domain, as B5 does) — and report both; T9 checks the identity on each.

### C-11 [MED] The column ground truth `T_j` is "decided" in the spec and "open" in the record
Record §36.3: "E7's interval-hit rate cannot be run on the chosen data ... Not decided unilaterally"; §40:
"Luke has not yet committed to (d)"; paper L2732-2733: "where those come from is stated with E7" (it is not).
Spec §12.3 presents C-a/C-b/C-c gated by the S0 routing check as if settled, and §13.1 L753 refers to the bridge
strings "for §12.3" although §12.3 has no bridge-string construction (record §40's option (d)).
**Followed:** spec C-a/b/c behind S0. The bridge-string construction is listed in R-1 as needing Luke.

### C-12 [MED] App. E says the relation is not evaluated; Sec. 2.3 and the spec say it is the mechanism
Paper L1534-1535: selective exclusivity "is not part of the mechanism evaluated in Sec. 5"; L1744-1747: "the
generalization would then change nothing in E2 or E3"; L1554-1556: "the five heads ... become three, at the cost
of whatever parameterizes the mask". Sec. 2.3 L480-502, spec §4 and record §36.0(3) make the learned pairwise
relation with `ρ₀ = 5 %` coverage the deployed mechanism (there are no gates; five heads are already four
modules `U, V, TauK, TauQ`). **Followed:** spec / Sec. 2.3. App. E's framing is stale.

### C-13 [MED] Stale notation and regime statements inside the paper
- L1108 `Ã_ij = c_j p_ij` — should be `c̄_j p_ij` on `E_.j`, `A^sm_ij` off it (eq:program L313-320).
- L1111 `θ_i` "the dual of the unit cap" and L2552 the same — it is the dual of the *quota* (L431-432); its range is
  `[0, max_{j∈E_i.} Ã_ij)`, not `max_j`.
- L1112 "the cap binds iff `τ_i R̃_i > 1`" — the regime threshold is `c̄_i/R̃_i` (L449-450). Record §39.3 says this
  was corrected in App. D; the notation table still has it.
- L1473 `Σ_j A_ij = min(τ_i R̃_i, 1)` — false in the two-step form (`= Σ_{∉E} a1 + min(τ_i R̃_i, c̄_i)`); record
  §36.1 says this prose was removed — it survives here.
- L1068 (tab:stages caption) "scoped to the active set selected by the gate"; L1554 "the query gate" — no gates
  (record §36.0(3)).
**Followed:** Sec. 2 equations / spec. All paper-level.

### C-14 [MED] Predicted-budget vocabulary surviving in the paper
L1294 "`c_j` is predicted per key from the key and its column"; L1350-1351 "`Σ_i Ã_ij = c_j` is learned, not fixed
at one"; L1374-1375 "collapse in `c_j`"; L1509-1510 frozen-prefix "with capacity holding approximately";
L526-527 "sealed ... as `c_j^(b)` is" (spec §7 ERRATUM). All contradict eq:step1 (inherited) and spec §0's
"code that predicts `c_j` is implementing an earlier version". **Followed:** inherited.

### C-15 [MED] The record's "authoritative" mechanism section is superseded by its own later sections
Record §4 is headed "⚠️ Authoritative. Both drafts implement exactly what is below" and states `Ã_ij = c_j p_ij`,
a predicted `c_φ(k_j, γ, ν_j)`, gates `ω^K, ω^Q`, "five heads", "cap slack iff `τ_i R̃_i ≤ 1`", `Σ_j c_j = n_q` "over
the active keys". Superseded by §35.0 (two steps), §36.0 (row-softmax step 1; cap not equality; no gates), §41.
Also stale: §0.3 "five components ... conditional activation (gate)"; §0.6 / §8 "S2 ~38 / ~30 H100-h" (paper
tab:compute: 60) and "the decisive baseline — column-entmax at uniform `c_j`" (now the E9 "uniform quota" ablation,
spec §13); §35.0's equality `Σ_{E_i.} a_j = c'_i` and column-softmax step 1 (superseded by §36.0); §35.4(e)
"`τ_j → 0` alone returns standard attention" (retired by §39.2). **Followed:** §§36-42 only.

### C-16 [MED] Harness labels and one harness block are not what the spec cites
`verify_all.py` uses TMLR numbering (Prop. 14/17/18/19/20/21/22/23/24, Cor. 2/5, "Prop. A/B") not the ICLR labels
in spec §11; the mapping is by content (pseudocode §10). Spec §2 says INV-6/7 must be tested "against the real
`eq:rowprog` code path, never against `max(Atil − th, 0)` with a sampled `th`" — the harness's Prop. 20 block
(L278-305) and its one-mask twin (L494-524) do exactly the latter; the real-path check is Prop. A(v) L631/L636
(record §36.1 admits this). **Followed:** T6 ports Prop. A(v), not Prop. 20.

### C-17 [MED] Decode-time computation of the fan-OUT entry for a new row is unspecified
Spec §6.4: relation and `τ_j` for cached keys are frozen at prefill and "each new query row runs its own fan-in
step"; nothing says how `Ã[t, j]` (a *column* quantity) is obtained at step `t`. Harness L312-323 (prop:prefix(iii))
re-solves the column over `z[:t]` at every `t` and fixes the new entry. Seal-at-boundary (spec §7) computes
`c̄_j^(b)` "when a block CLOSES", which a decode block's own rows cannot wait for.
**Followed:** pseudocode §5.7 — per-key re-solve over the cached relation scores at frozen `τ_j`, taking only the
new entry; seal-at-boundary in decode = use the previous block's sealed values. Confirmed as R-3.

### C-18 [LOW] `eq:tauhead` omits `ν_j`
Paper L363 and notation L1107: `τ_j = τ^K(k_j, s_.j)`; L376-382, L225, L1077 include `ν_j`. **Followed:** with `ν_j`.

### C-19 [LOW] "`n` up to a few hundred" vs `n ≤ 128`
Paper L889, L2668 vs spec §13 E3 `n ∈ {8,…,128}`. **Followed:** spec (RULER's essay fill bounds `n·m` at 8K).

### C-20 [LOW] A "hard variant" of the row truncation is promised (L1494-1497 "we ablate both") and is in no E9 list
(spec §13, paper L2828-2837). **Not implemented**; listed for Luke.

### C-21 [LOW] "Cost is unchanged" / "cost proportional to relation coverage"
Paper L502, L2767-2769 vs spec §6.5: the dense reference materialises ~4 `[B,H,T,T]` fp32 tensors regardless of
coverage; only the chunked path's pass 2 scales with coverage. **Followed:** report efficiency on the chunked path,
say which path produced the number.

### C-22 [LOW] Module swap by `load_state_dict(strict=False)` (spec §6.1) vs wrapping the original module
Equivalent numerically; wrapping keeps the projection `nn.Linear` objects and their names, so peft's
`target_modules` and T0's weight equality are trivially preserved. **Followed:** wrap.

### C-23 [LOW] `ν` state "passed layer to layer through `kwargs`" (spec §5.3)
Newer transformers filter attention kwargs through typed dicts; a mutable per-forward context object set by a
pre-forward hook is robust. **Followed:** context object.

### C-24 [LOW] LR group for modules that exist only from Phase B
Spec §9 gives LR 1e-3 for new modules and one cosine over 3500 steps, silent on adding a param group at step 500.
**Followed:** `opt.add_param_group` at Phase-B start; the scheduler multiplier continues (no restart).

### C-25 [LOW] `c̄_i ≤ 1e-9`: harness keeps `a1_E`, spec returns `u = 0`
Harness L625 (`if ci > TOL`) leaves `a2[E] = a1[E]` (mass ≤ 1e-9); spec §3/§5 give `u = 0 ⇒ A_E = 0`. Differ by ≤ 1e-9
(ref ambiguity #29). **Followed:** spec.

### C-26 [LOW] "bit-identical" (spec §1.2, §8 B0) vs T0's `1e-5` bf16 tolerance
`masked_fill(-inf)` vs HF's additive `finfo.min` give identical fp32 softmax on every row with ≥ 1 visible key
(pad rows differ: uniform vs 0, never read). SDPA-vs-eager differences make bit-identity meaningless anyway.
**Followed:** T0's tolerance and exact greedy tokens.

### C-27 [LOW] S0's content
Paper tab:compute L2607: "E1 on trained-backbone scores; gate: a replication pair"; spec §12.3 adds the routing
check (mass fraction > 0.5 on ≥ 80 %). **Followed:** spec (superset).

### C-28 [LOW] Panel (b) numbers
Record §30.3 (1.55, 3.07, 27.61) vs paper L955 (1.59, 3.20, 28.21); §36.1 explains the fix (negative temperatures).
Paper is current.

### C-29 [LOW] "Answer positions" under teacher forcing
Spec §12.2 "the positions that emit each gold value" / "each gold answer token": the position that emits token
`t+1` is `t`; for multi-token values the spec does not say first-token or all. **Followed:** RULER — the position
emitting the first token of each value (one row per value → `m` rows); QA — every position emitting a gold token.

### C-30 [LOW] Phase A is arm-independent
Spec §9 lists Phase A under every arm; at fixed seed it is the same 500 LoRA steps for all seven. **Followed:** run
once per seed, fork the checkpoint (the spec's "same adapted weights" is then literal).

## B. Residual ambiguities after reading everything (ranked by risk)

- **R-1 [HIGH] Which `T_j` construction E2/E7's interval-hit rate uses, and which token is "the key".** Record
  §36.3/§40 leave it to Luke; spec C-b says "last token of the question / the needle key phrase" — the key phrase
  appears once in the question and `m` times in the needles (`niah.py` L83). Chosen: the question copy for C-b, the
  needle copies as C-a columns. E2's *falsification criterion* (hit rate falling with `m`) depends on this.
- **R-2 [HIGH] E2 feasibility at `L = 8K`.** `K = 32, V = 16` ⇒ 512 needle sentences; the essay haystack shrinks to
  nothing and the task changes character. Either `n` must drop for E2 (e.g. `n = 8`) or `L = 16K` for the large-`m`
  points. Needs a decision before generating the S2 grid.
- **R-3 [MED] Generation-mode semantics for MarSea/B3** (C-17): per-step column re-solve on cached relation scores
  (cost `O(|E_t.|·t)` per token without hierarchical top-K) vs something cheaper. Exact-set accuracy is a headline
  number, so this is not cosmetic.
- **R-4 [MED] B5 in generation mode.** The trace is per (example, row) from a teacher-forced pass; generated rows do
  not align with gold rows. B5 is well-defined for attention-level metrics only, unless a position-keyed trace is
  accepted as an approximation. Spec says "evaluation-time only" without saying which mode.
- **R-5 [MED] `δ_j` when `T_j ⊄ E_.j` or `E_.j \ T_j = ∅`.** Chosen: compute the interval on `T_j ∩ E_.j` (`m_j` becomes
  the recoverable count), `δ_j = +∞` (lower endpoint 0) when the relation holds no non-target; relation recall logged
  separately as the spec demands. Not stated anywhere.
- **R-6 [MED] `K_i` for RULER rows.** Gold source keys = value tokens only, or the whole needle sentence (key
  phrase + value)? Chosen: value tokens, with the key-phrase variant logged.
- **R-7 [MED] Dense-arm domain for E3's "comparator precision `== m/|E_.j|`" (C-10).** Both domains are reported;
  which one is the headline is a paper decision.
- **R-8 [LOW] Calibration batch.** "One warm-up batch" — composition (16 sequences at `L`, mixed 1:1:1?) and whether
  the `b0` quantile is joint over heads (it must be, `b0` is shared) are unstated. Chosen: the first Phase-B training
  batch, joint over heads and batch, per layer.
- **R-9 [LOW] MESH details not in the spec:** `‖grad‖₂` per `(b,h)` slice vs global, noise only on visible entries,
  `m` per batch element. Chosen as in pseudocode §6.
- **R-10 [LOW] IHEval on a base checkpoint:** the multi-turn rule-following format has no chat template; "task
  default" decoding and prompt layout are undefined for a base model. Affects E6 (S4) only.
- **R-11 [LOW] transformers pin.** Spec says ≥ 4.48 "verified against main"; the mask and kwargs interfaces changed
  at 4.53. The code must pin one version and assert the signature at import (spec §1.3) — the pin itself is unstated.
- **R-12 [LOW] `row_summary` and `R̃_i` use the hard `E`, not `g`** (no logit gradient through the summary). Spec
  silent; reference does this; kept.
- **R-13 [LOW] Ties in B5's top-`n_i` and in `sorted_prefix_stat`:** stable, lower index first (H3/H8). Kept.
