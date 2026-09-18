# MarSea — Master Project Record

> ⚠️ **Renamed 2026-09-02: JOCEA → MarSea** (*multivariate relations by selective exclusion in
> attention*). The name is applied retroactively throughout this record, so earlier sections read
> "MarSea" for work done under the old name. File and project-doc paths moved with it:
> `joca_iclr.*` → `marsea_iclr.*`, `joca.*` → `marsea.*`, `jocea_*` → `marsea_*`. A few references
> to superseded documents keep their original names (`joca_positioning_decisions.md`, `joca_ws`)
> because those documents were never renamed.

> ⚠️ **Reading order for implementers (added 2026-09-05).** The CURRENT design is §§35–43;
> the standing decisions are §36 (two-step selective form, cap, no gates), §38.1 (softmax once,
> at Stage 1 step 1), §39.2 (`τ_j → 0` is uniform, not standard attention), §40 (no abstention;
> `m_j ∈ {0,1}` columns are row-side), §41 (row softmax at step 1), §43 (`τ_i = 1` is the identity
> on step 1's row). **§§1–34 are history**: §4 below is headed "authoritative" but describes the
> superseded design (`Ã = c_j p`, a predicted budget `c_φ`, gates, five heads, column-entmax
> baseline, S2 = 38 h); where an early section and §§35–43 disagree, §§35–43 win, and the
> implementation contract is `claude/marsea_implementation_spec.md` (v4.1).

> ⭐ **This file is the whole record again, §1–§60 (merged 2026-09-15, §60.4).**
> `claude/marsea_master_record_part2.md` is superseded and must not be written to.
> The live end of the record is **§60**; the standing decisions are §0, §36, §38.1, §39.2,
> §40, §41, §43 and the decisions register.

**Authoritative state as of 2026-08-29** (superseded where §§35–43 say so). This is the document to read first. It supersedes
`claude/joca_positioning_decisions.md` (now a pointer) and absorbs
`claude/marsea_audit_2026-08-23.md` (kept for its itemized change log). Prior-session history:
`claude/thread_history_constrained_attention.md`.

---

## 0. START HERE — handoff for a new session

### 0.1 The project instructions are correct again

The project's own instructions say *"under drafting for 2027 ICLR research paper submission…
best chance of acceptance at ICLR"* and *"one or two paper drafts."* As of **2026-08-25 (§24)**
that is once more the operating reality: Luke has directed the work toward **ICLR 2027**, with
the TMLR submission **set aside, not cancelled**. There are now **two live drafts** from one
body of work. Earlier revisions of this record said the instructions were out of date; that
note is withdrawn.

### 0.2 Four standing decisions — do not relitigate

1. ⚠️ **Venue is ICLR 2027 (abstract 18 Sep 2026, paper 25 Sep 2026 AoE); TMLR is set aside,
   not cancelled.** Luke, 2026-08-25: *"Let's move toward ICLR paper submission (putting the
   TMLR paper submission aside for now)."* **The 9-page main-text limit is back in force** and
   is a hard constraint (10 pp camera-ready; references, appendices and the mandatory
   statements are excluded). `marsea.tex` — the 68 pp TMLR draft — is **not deleted and not
   deprecated** and remains the fallback if ICLR rejects. As of **2026-08-29 (§27)** the two
   drafts describe the **same mechanism and the same theory** again; do not port ICLR-motivated
   **compressions** backwards.
2. **`joca_ws` (the workshop paper) is dropped entirely.** Do not revive it, port fixes to it,
   or treat it as prior-art insurance.
3. ⚠️ **The causal/prefix form is NATIVE DESIGN, not an adaptation.** Both programs run over
   the visible prefix in a decoder (§3.6 of the paper, §21 here). **Never again raise causal
   decoding as an obstruction, a reformulation, or unimplemented work.** Luke has instructed
   this explicitly.
4. **Precision is the centre; recall is protected, not traded.** ⚠️ As of §27 the word *trade*
   has **two** correctly-scoped homes, not three — the third was the region where
   `m_jδ_j > W_j` fails, and that region is now empty (§12).

### 0.3 Current state

**Two drafts, one body of work.**

| Draft | File | Venue | Length | Scope |
|---|---|---|---|---|
| **ICLR 2027** (active) | `claude/marsea_iclr.tex` | ICLR 2027 | **9 pp main + 2 pp refs + 19 pp appendix = 30 pp** | reasoning only |
| TMLR (set aside, but **current**) | `marsea.tex` | TMLR | **80 pp** | vision **and** language/reasoning |

`marsea_iclr.tex` builds clean against the **official ICLR 2027 style files** (§24.5):
**0 errors, 0 undefined refs, 0 undefined citations, 0 overfull boxes, 0 multiply-defined
labels.** 30 pp total — **main text ends on p9**, references p10–11, appendices p12–30.
⚠️ **There are now two figures**: Fig. 1 is the new precision–recall trace (p7) and Fig. 2 the
E1 probe (p8) — the probe was Fig. 1 in every earlier record entry, and LaTeX renumbered it
because the new figure appears first. All references are by `\ref`, so nothing broke. 32 bibitems, all cited, none uncited; 16 numbered claims, 16 proofs,
**5 restatements** (two results became full statements in App. E — §27.4). ⭐ **App. E is new
as of §28** — selective exclusivity — and shifted Proofs to App. F, numerics to App. G,
protocol to App. H.
⚠️ Limitations is a `\paragraph`, not a `\section`, purely to hold the 9-page limit; the source
carries a comment to restore `\section` for the 10-page camera-ready (§24.13).

`marsea.tex` — **80 pp** (66 before Tiers A and B, 68 after, 77 after §28's new §5), builds clean:
**0 errors, 0 undefined refs, 0 overfull boxes, 0 multiply-defined labels.**
`marsea.bib` — 80 entries; `marsea.bbl` — 71 cited. `verify_all.py` — all claims at **0 violations**
over **456,621 checks** (§29.4), and ⚠️ **no block in it rejects any more** (§28.5).

⭐ **Independently reproduced on 2026-08-25** (§23) and again on **2026-08-29** (§27): before any
editing, `marsea_iclr.tex` and `verify_all.py` were transcribed into a clean directory and
reproduced every recorded invariant exactly — 24 pp, main text p9, Fig. 1 p8, 16 claims / 16
proofs / 7 restatements, 32 bibitems, and every `verify_all.py` count including the k\* regime
table (1·5·8 / 1·7·14 / 4·35·74) and Prop. 24(iii)'s row sums (0,1,2,0,2). **Do this before
editing; without it no later measurement means anything.**

**The mechanism has five components**, and §2 of the ICLR paper introduces all five:
Stage 1 (fan-out program) · Stage 2 (fan-in program) · conditional activation (gate) ·
hierarchical top-K selection · native causal/prefix form. **§4 below is the authoritative
statement of the mechanism.**

**ICLR paper structure:** §1 Intro (+ Related work) · §2 Method (2.1–2.5) · §3 Theory (3.1–3.5) ·
§4 Application · §5 Experiments E1–E9 · Limitations · statements · App. A Notation and components
· B Blocking and hierarchy · C Related work in full · D Limitations continued · E Proofs ·
F Numerical verification · G Experimental protocol.

**TMLR paper structure (`marsea.tex`, set aside):** §1 Intro · §2 Related Work · §3 Method
(3.1–3.7) · §4 Theory (4.1–4.8) · §5 Experimental Plan (vision, V1–V11) · §6 Language and
reasoning (6.1–6.7, V12–V18) · §7 Limitations · App. A Proofs · App. B Numerical verification.

### 0.3a ✅ CLOSED 2026-08-28, REOPENED AND RE-CLOSED 2026-08-29

**Closed 2026-08-28 (§25)** when `marsea.tex` was caught up to `marsea_iclr.tex` on the eight
mechanism divergences. **Reopened 2026-08-29 (§27)** when the Tier A theory corrections landed in
`marsea_iclr.tex` first, and **re-closed the same day** when they were ported into `marsea.tex`.

The table below is kept because it is the index of what changed and the fastest way to see
whether an old copy of either file is stale.

| | ported | recorded in |
|---|---|---|
| Fan-in cap fixed at 1, `β_φ` deleted | both | §24.12 |
| Row exclusivity `τ_i` inside the projection | both | §24.10, §24.12 |
| Row closed form `A_ij = τ_i[Ã_ij − θ_i]_+` | both | §24.10 |
| Field-reading heads on all four sites | both | §24.13 |
| Per-margin heads `τ^K_φ, τ^Q_φ, ω^K_φ, ω^Q_φ` | both | §26 |
| Fan-in trigger as a **disjunction** | both | §24.8 |
| `Σ_j c_j = n_q` as a normalization | both | §24.6 |
| ⭐ **Sharp endpoints in Thm. 2** | **both** | **§27.2** |
| ⭐ `cor:joint` consistency hypothesis | **both** | **§27.3** |
| ⭐ `τ_i·R̃_i` regime correction | **both** | **§27.3** |

⚠️ **The standing rule that produced this section, restated:** *a fix applied to one draft does
not certify the other, in either direction.* §25 ported forward and three fixes never came back
(§27.3); §27 fixed the ICLR draft first and had to port forward again. ⭐ **And as of §60.1 it is not only
about the two papers**: two copies of `marsea_iclr.tex` drifted the same way, and the patch script
for an unrelated fix applied cleanly to the stale one.

### 0.3b ⭐ Is this copy of `marsea_iclr.tex` current? Three probes (added 2026-09-15, §60)

Before editing, patching or quoting any copy of the ICLR draft, `grep` for these. **All three must
be present.** Each is text no pre-2026-09-10 copy contains, and their absence is what §60.1 caught:

| probe | decision | if missing |
|---|---|---|
| `One position is excluded from every relation by construction` | **D-31** (§52) — the sink is in no relation | run `portforward_d9a_d31.py` |
| `\emph{coreference column}` | **D-9a** (§53) — the `T_j` split | same |
| `\emph{value column}` | **D-9a** (§53) — the `T_j` split | same |

Two negative probes are as useful: a current copy contains **neither** `$m_j=2m$ moves with the
sweep` nor `together with the answer positions that emit its value`, both of which D-9a removed.

⚠️ **`patch_iclr_cap.py` now enforces this itself** and refuses to write on a copy missing any of
the three (§60.6); `patch_iclr_window.py` carries the same guard plus `eq:excess`. Any future patch
script against this draft should do the same — asserting that your `old` strings match is not the
same as asserting you are editing the right file.

⚠️⚠️ **And before editing ANY file in this project, check whether a second copy of it exists**
(§61.2): `verify_all.py` existed twice, under `scripts/` and under `documents/`, and two parties
edited the two copies on the same day without either being able to see the other's change. The
probes above guard the draft; nothing guards a file you did not know was duplicated except
`find . -name <basename>`, which costs nothing.

### 0.4 Files, and where they live

| File | Role |
|---|---|
| `marsea.tex` | The TMLR paper. Single source file. |
| `marsea.bib` / `marsea.bbl` | 80 references / 71 cited, formatted. |
| `verify_all.py` | Verification harness. **Must sit beside `marsea.tex`** — Appendix B `\lstinputlisting`s it. ⚠️ **Shared by both drafts**; changing it makes both tables stale at once. |
| `math_commands.tex`, `tmlr.sty`, `tmlr.bst` | Build dependencies. TMLR style files are the official ones from `JmlrOrg/tmlr-style-file`. |
| `v1_probe.pdf` | Figure 1, used by **both** drafts. ⚠️ **Watermarked reconstruction — must be regenerated.** |
| `claude/marsea_iclr.tex` | The ICLR 2027 paper. Single source file, 9 pp main text. |
| `claude/marsea_iclr.bbl` | 32 bibitems, exactly the cited set. Hand-maintained — there is no `.bib` for it. |
| `iclr2027_conference.sty` / `.bst` / `.tex` / `.bib` | **Official** ICLR 2027 style files, at the project root. |

⚠️ **The ICLR build files live under `claude/`, not at the root.** They still need
`math_commands.tex` (root) beside them at build time, and the real `v1_probe.pdf`.

**Context documents, in reading order:** this file (§0 first) →
`claude/marsea_iclr_audit_2026-08-28.md` (the adversarial audit: findings and triage) →
`claude/marsea_iclr_tierA_applied_2026-08-29.md` (what Tier A did) →
`marsea_audit_2026-08-23.md` (historical change log; ⚠️ describes **four** components, a `g_j`
top-gap gate and a 41 pp TMLR draft — all superseded) → `thread_history_constrained_attention.md`
(reconstruction of the lost first session) → `joca_positioning_decisions.md` (superseded pointer).
⚠️ `Cona-sp_Project_Synthesis.md` is Luke's own July upload and describes the **predecessor
mechanism, Cona-sp** — which MarSea's own separation theorem later proved sits *inside* the
query-local class. Read it as history, never as current design.

⚠️ **A retrieval fact worth knowing before planning a session** (learned 2026-08-25, refined
2026-08-29): project *docs* can be read back in full. **A doc large enough — `marsea.tex` at
286 KB — comes back as a path to a local file instead of inline, which means it can be edited
surgically with no transcription risk.** Smaller docs come back inline and must be re-emitted in
full to be changed, which is the §19 hazard. Project *file attachments* that are PDFs —
`v1_probe.pdf`, `marsea.pdf` — come back as **extracted text, not bytes**, so a session cannot
recover the actual figure or a compiled PDF from the project.

Build: `pdflatex → bibtex → pdflatex → pdflatex`, or `pdflatex ×3` when the `.bbl` is present
and citations are unchanged. Requires `lmodern` (Debian/Ubuntu: `apt install lmodern`) and,
for the ICLR draft, `wrapfig` and `tabularx` (the latter added 2026-08-30 so the notation table
fills the text block — see §28.10).

### 0.5 ⚠️ Process rules learned the hard way

- **Work in exactly one directory.** A stale `cp` silently reverted two edits on 2026-08-23 (§19).
- **Verify edits in the rendered PDF, not only in the `.tex`.**
- ⭐ **A correction is not done until it is propagated to every site that states the claim.**
  After changing any quantitative claim, `grep` the whole source for the old figure and for
  paraphrases of it, and check the count is zero.
- **Any number in the paper that came from a script must be reproducible by a script the
  paper ships.**
- ⭐ **A patch script must write before it can fail — or assert every match up front.** Learned
  twice on 2026-08-26 (§24.11, §24.13) and used correctly on 2026-08-29: every patch in §27
  asserted that each `old` string occurred **exactly once** before writing anything, and each
  batch was then grep-verified for the new text. Never trust the script's own stdout.
- ⭐⭐ **Run an adversarial reader over any mechanism change that touches a closed form.**
  Learned 2026-08-28 (§25.4): a port that passed a clean symbol sweep, a clean ref/label sweep,
  a clean rebuild and a read of the rendered PDF still carried **nine** invalidated theory
  statements.
- ⭐⭐ **Run it over every draft the change touches, in both directions** (§27.3).
- ⭐⭐ **When a proof bounds a quantity it could compute exactly, compute it, and check the
  resulting condition is sharp before anything downstream leans on it** (§27.2). One `≥` in
  Thm. 2's Step 2 propagated into a false corollary, a false half of a proposition, an overclaim
  in the abstract, and a mis-targeted experiment — and the numerics could not catch any of it,
  because the harness *rejection-sampled on the very condition the looseness made unnecessary*.
- ⭐ **When a page limit will not close, check which *page* each float landed on, and count
  text lines per page.** Learned 2026-08-28 (§26.3) and again 2026-08-29 (§27.5): roughly ten
  rounds of sentence-shaving moved the boundary by under two lines. **The decisive diagnostic is
  to build once with the figure deleted** — if the prose then fits, the entire overflow is the
  float and no amount of shaving will help.
- **Check float placement on any table that grows.** The notation table silently floated to
  the end of the document once.

### 0.6 The four things most worth doing next

1. **Regenerate `v1_probe.pdf`.** It is the ICLR paper's *only* figure and its only run
   experiment, and per §0.4 it is the one artifact the project cannot hand back to a future
   session. ⚠️ **It is now also the last thing standing between the build and a confirmed
   9-page boundary** — §27's build used a placeholder, calibrated against the recorded baseline
   but not the real figure. Luke set this aside on 2026-08-26 ("QUIT task #2"); it stays on the
   list because the paper cannot ship without it. ⭐ **A second reason has appeared**: E1's
   decisive baseline — column-entmax at uniform `c_j`, which the paper itself calls "the core
   comparison" — is **absent from Fig. 1**, and adding it is one more series in the same figure
   (audit §6.7).
2. **Run S0 and S2.** S0 costs one GPU-hour and can invalidate the central theorem's
   applicability; S2 (the RULER `m`-sweep, ~38 H100-h) is the decision point. With the deadline
   on 25 Sep, S0–S2 is the critical path.
3. ⭐ **Implement the heads as specified and check the E7 diagnostic early.** ⚠️ **The
   interval-hit rate must now be scored against the sharp interval
   `[1/(W_j+m_jδ_j), 1/W_j)`** — measured against the old sufficient interval, a head landing in
   the gap would be recorded as a miss while recovering exactly, and that gap is **27% of
   columns** on the planted-margin sweep (§27.2).
4. ⭐ **Decide the rest of Tier C.** Tiers A **and B** are applied to both drafts (§27).
   ⭐ **One Tier C item is now discharged** — audit §6.4, `δ_j` hiding `n` in Prop. leakage's
   comparison — by §28's selective-exclusivity section, which concedes the point explicitly and
   supplies the mechanism that removes it. **The rest is untouched and is Luke's call**:
   Prop. 2's near-vacuity in a decoder where `n_q = n_k`; Thm. 1's hypotheses not being jointly
   satisfiable in self-attention (and whether to demote it to a lemma with Cor. 1 + Assn. 2
   promoted); precision-side guarantees against a recall-critical task; and the Slot-Attention
   and expert-choice-routing characterisations in related work. See
   `claude/marsea_iclr_audit_2026-08-28.md` §6–§7. Each is a framing change, not a defect fix.

---

## 1. Identity

- **MarSea** — Joint Optimization in Competitive-Exclusion Attention. Renamed from JOCA
  2026-08-21. Filenames were `joca.*` until the 2026-09-02 rename to `marsea.*` (§36).
- **Title:** *MarSea: Joint Optimization in Competitive-Exclusion Attention* /
  *Content-Dependent Degree Constraints on Both Margins of the Attention Graph.*
  "Content-Dependent" is load-bearing: Sinkformer also constrains both margins; *uniform vs.
  learned* is the difference.
- **Open reservation:** MarSea has ambiguous pronunciation where JOCA was clean. Reversible with
  one `sed`.

---

## 2. The relation class: competitive exclusion

A set `A` jointly bears on `b`; members **compete for a finite influence budget**; which
prevails is settled by comparison **within A**, not by any property of a single pair.

| Form | Ordering from | Example |
|---|---|---|
| **Ranked** (preemption) | a rule over kinds | officer ≻ emergency vehicle ≻ hand-held sign ≻ pedestrian ≻ light |
| **Graded** | a measurement | fastest-moving / shortest-queued / ice-free lane |
| **Configural** | scene configuration | the car with a red light, inside the intersection, in the ego's path |
| **Referential** (new, §17) | denotation | a pronoun's antecedent; a variable's binding in scope |

**Configural is the sharpest** in vision: competitors are **same-class** (all cars) so no
category prior helps; the criterion is **conjunctive**; and it is **target-conditioned**.
**Referential is the sharpest overall**, because it is the only form with gold annotation of
the target set itself.

**Core argument — relevance is a superlative, and superlatives are set-relative.** No function
of one lane and one vehicle can decide whether that lane is *the fastest*.

### ⚠️ "Little to nothing", not "nothing"

At α=2, `p_ij = [τ_j s_ij − ψ_j]_+` — affine on its support. Allocation is **graded among
survivors, exactly zero only past the threshold**. A close rival keeps a small share.

**Why the name stays "exclusion" not "prioritization":** grading is *not* the scarce
capability — row-softmax already ranks and already gives more to the better placed. What
attention cannot do is **end** the ordering at a content-decided cutoff.

### Terminology discipline (Remark 1)

- **competitive exclusion** = the class. Neither half suffices alone: softmax competes without
  excluding; fixed top-k excludes without competing.
- **preemption** → ranked case only. **graded competition** → measured case.
- **Avoid "dominance"** — names a relation between *two* alternatives, the pairwise reading the
  paper argues against. **Avoid bare "exclusive"** — implies one winner; m_j > 1 routinely.

---

## 3. ⚠️ Orientation: attention direction ≠ influence direction

**Caused two substantive misreadings. Read before touching the intro.**

Semantic influence (A → b) is **not** attention direction (key → query), and **MarSea does not
fix the correspondence** — that is part of what it provides. Signals-as-keys → preemption is a
**row** statement; signals-as-queries → the *identical* relation is a **column** statement.
Which holds is decided by how the representation is learned and may differ across heads,
layers, datasets. A one-margin design silently assumes an orientation it cannot verify.

**Do not pin the relation to one margin.** An earlier edit did ("a signal's content must reach
the driver, so the signal *is* a key") and was corrected.

---

## 4. Mechanism — five components, not two

⚠️ **Authoritative. Both `marsea.tex` and `marsea_iclr.tex` implement exactly what is below.**

**Stage 1 — fan-out (column); origin of all non-query-locality:**
```
p_·j = argmax_{p∈Δ} { ⟨p, τ_j s_·j⟩ + H^T_α(p) }   (α=2, sparsemax)
Ã_ij = c_j · p_ij
```
`Σ_j c_j = n_q` over the **active** keys (per block in the causal form) is a **normalization,
not a constraint of the program** — no theorem uses it (§24.6). ⚠️ It is a *matched total*, not
a duality: ungated keys emit outside the budget and demand falls short whenever a row has slack
(§25.2(7)).

**Stage 2 — fan-in (row), capped projection at a fixed unit cap:**
```
A_i· = argmin_{a≥0, Σ_j a_j ≤ 1} ½‖a/τ_i − Ã_i·‖²  ⟹  A_ij = τ_i[Ã_ij − θ_i]_+
```
θ_i is the **dual** of the unit cap — a threshold on `Ã_i·`, set jointly by the whole row.
One-sided inequality (**≤ 1, not = 1**), so it is not a normalization.

⚠️⚠️ **The regime is set by `τ_i·R̃_i`, not by `τ_i` alone**, where `R̃_i := Σ_j Ã_ij` is the
row's supply (§27.3). The cap is slack iff `τ_i R̃_i ≤ 1`. The normalization makes `R̃_i = 1`
only *on average*, and ungated keys add softmax-scaled mass on top, so the clean reading
"below one is slack, above one binds" holds **only on unit-supply rows**. ⭐ **This also scopes
the second margin**: the cap bounds fan-in **mass** unconditionally but fan-in **degree** only
where it binds — a slack row leaves `supp(A_i·) = supp(Ã_i·)` untouched and Stage 2 is a
positive scalar gain.

⚠️ **θ_i is a dual, not a knob, and its range is bounded** (§25.2(9)). As `τ_i` ranges over
`(0,∞)`, `θ_i` traverses `[0, max_j Ã_ij)` strictly increasingly, and `θ_i = 0` is *forced* on
any row arriving within budget. Row-side intervals can be **unreachable** at every admissible
`τ_i`; in particular a row whose weakest true source is its largest entry can never shed it, so
`R_i ≥ 1/|K_i|` unconditionally.

⭐ **The heads, and what each reads** (§24.13, §26 — every one is a function of a participant
*and* the field it faces):

| | fan-out (key `j`) | fan-in (query `i`) |
|---|---|---|
| gate — *whether* | `ω_j = ω^K_φ(k_j, s_·j)` | `ω_i = ω^Q_φ(q_i, Ã_i·)` |
| exclusivity — *shape* | `τ_j = τ^K_φ(k_j, s_·j)` | `τ_i = τ^Q_φ(q_i, Ã_i·)` |
| capacity — *scale* | `c_j = c_φ(k_j, {γ_ij}_i, ν_j)` | **fixed at 1** |

⚠️ **Five heads, and the margins do NOT share one** (§26). Active key set `K = {j : ω_j>0}`.

Pipeline, no fixed point: `s_·j → ω^K_φ → τ^K_φ → p_·j → ν_j → c_j → Ã_·j`, then
`Ã_i· → ω^Q_φ → τ^Q_φ → θ_i → A_i·`.

⭐ **Eq. 5 is also displayed in scale-separated form** (§26.2):
`A_i· = τ_i · argmin_{u≥0, Σ_j u_j ≤ 1/τ_i} ½‖u − Ã_i·‖²`. **`1/τ_i` is the row's effective
budget in supply units**; `τ_i` restores the unit scale. `a·τ_i` would invert every regime
statement.

⚠️ **Causal form:** the gate decision and `τ_j` are **sealed at the block boundary**, as
`c_j^(b)` is. Prop. 23's monotonicity is proved at fixed `τ_j`.

**Component 3 — conditional activation (gate).** See §13. **Component 4 — hierarchical top-K
selection.** See §14. **Component 5 — native causal/prefix form.** See §21.

**Budget conditioning:** `c_j = c_φ(k_j, {γ_ij}, ν_j)`, `ν_j = ‖p_·j‖₂⁻² = 1/(1−2·H^T_2(p_·j))`
- **Non-circular**: p_·j depends only on τ_j, s_·j; c_j enters after.
- ν_j = **effective cardinality**, smooth (|supp| is piecewise constant, no gradient).
- **ν_j is an INPUT, never a definition.** `c_j := ν_j` would slave scale to shape.

**How ordering becomes allocation (§3.2):** relation supplies an ordering, training puts it in
s_·j, Eq. 4 converts it. **τ_j selects the regime**: large → ranked preemption; small → graded
competition. Cor. 3 gains a semantic reading: *no single global temperature serves a preemptive
relation and a graded one at once.*

---

## 5. Theory inventory

**Negative:** Thm 1 (query-local attains *exactly* ∏ᵢ𝓡ᵢ) · Cor 1 (capacity/cardinality
non-product-form; ⚠️ hypotheses `c>0`, `1 ≤ m < n_q` now stated) · Assn 1 + Prop 3 (replication
pair bridges to constrained domains) · Prop 4 (conservation).

⚠️ **Prop. 4 restated correctly as of §25.2(6) and §27.3.** It forbids bounding **every** column
below a common `n_q/n_k`. MarSea does **not** escape it by leaving rows un-normalized — saturating
rows do sum to one, and MarSea's own normalization gives `max_j c_j ≥ n_q/|K|`, the same
pigeonhole. It escapes the **conclusion**: its bound is the learned, non-uniform `c_j`, so the
unavoidable total is allocated unequally and by content, which a uniform column marginal cannot
do. **Never again write "the fan-out bound survives because rows are not normalized."**

**Positive — brackets the gap from below:**
- **Prop 11 (realizability)** — every competitive selection at any cardinalities is exactly and
  simultaneously realizable, robust to score perturbation < ½ sup-norm. ⚠️ As of §27.3 the
  cardinalities are free **only subject to `Σ_j m_j = n_q`**, which a partition forces; the
  widest spread is scope 1 alongside scope `n_q−n_k+1`, **not** scope `n_q`. Floors are **not**
  free witnesses — they are duals of the unit cap.
- **Prop 12 (where the difficulty lies)** — (i) selection at a *single* target **is** available
  to query-local schemes — a deliberate concession; (ii) bounded **scope** is not; (iii) the
  non-local content is their **conjunction**. ⚠️ In `marsea_iclr.tex` (iii) is now proved by a
  **cardinality** route (`|supp(A_·j)| ≤ k*(τ_j)` via Lem. 1) rather than a mass route via
  Markov, which also removes an unstated `sup_i τ_i < ∞` hypothesis. `marsea.tex` was already
  correct here — its Markov use is in `prop:cardinality`, a genuine mass claim.
  ⚠️ **This concession governs which experiments count.**
- **Rem 10 (budget–cardinality)** — under c_j = m_j, Σ_j c_j = n_q **iff** {T_j} partitions the
  queries. Signed residual n_q − Σ_j m_j is a logged diagnostic.

**Structural:** capacity · fan-out cardinality (Markov) · fan-in bound · monotone exclusivity ·
three-level rejection · global sparsity · Lem 1 prefix property.

### ⭐⭐ Recovery — Thm 2, with SHARP endpoints (§27.2)

```
supp(Ã_·j) = T_j   ⟺   τ_j ∈ [ 1/(W_j + m_j δ_j),  1/W_j )
```

**Both endpoints are equivalences**, because `G(m_j+1) = τ_j(W_j + m_jδ_j)` is computable rather
than merely bounded below. **The interval is non-empty on every column with δ_j > 0** — i.e.
under Assn. 2 alone.

⚠️ **The old form `[1/(m_jδ_j), 1/W_j)`, non-empty iff `m_jδ_j > W_j`, is a *sufficient* interval
produced by a lossy bound.** It is a strict subset. On a 200,000-column planted-margin sweep,
**27.4% of columns that the old condition declares infeasible are ones where exact recovery is
in fact attained**.

- δ_j sets the lower endpoint jointly with W_j; the m_j multiplier reflects all targets pushing
  the threshold together. δ_j > 0 is **necessary** — by Lem 1 the support is a *prefix*.
- ⭐ **What the sharp form changes in the story**: exact recovery is never obstructed *on the
  column*, so the obstruction moves from "the interval may be empty" to "**the head must find
  it**", and the measurable quantity becomes the interval's **relative width `m_jδ_j/W_j`**
  rather than a binary feasibility fraction. This is a better target for E7/V4 and a better
  argument for a field-reading head.
- ⭐ **Cor. 3 (necessity) was FALSE and is now true and stronger.** It inferred non-recovery from
  falling outside a merely sufficient set. Corrected hypothesis: `W_{j₂} + m_{j₂}δ_{j₂} ≤ W_{j₁}`.
- ⭐ **Prop. printerval is now a genuine two-sided characterisation**, which makes the ICLR
  abstract's *"exactly a precision–recall interval, one endpoint per error type"* true rather
  than an overclaim.

**→ Fidelity pillar: see §12.**

---

## 6. ⚠️ Errors found and corrected (do not reintroduce)

1. **Remark 7 row-cardinality claim — FALSE, fixed.** Thm 1 concerns the **column**; a row has
   no product structure across keys, so row-sparsemax *can* bound row cardinality.
2. **Deformable attention structurally incompatible** — weights/offsets are *linear projections
   of the query feature*, not inner products (no s_ij); offsets query-specific, so **k is a
   per-query slot index, not a key identity** and A_·j is undefined. Backbone #2 → **DINO
   decoder self-attention**. Deformable moved to *baselines*.
3. **Symbol collisions:** R_i (set vs scalar row mass) → set renamed **𝓡_i**; 𝒜 → scheme now in
   words; c_θ vs c_φ → standardised on **φ**. ⚠️ `R̃_i` is the pre-truncation row mass and is
   back in the ICLR notation table as of §27.3, flagged as *not* row recall `R_i`.
4. **Free index j** in Thm 1 / Cor 1 → "Fix a key j." **Def 1 untyped** → x_i ∈ ℝ^d × Γ^{n_k}.
5. **"contributing nothing, not merely less"** → "little to nothing" in four places.
6. ⭐ **"SOTA is LLM-based, so a small encoder lands far below leaderboard and familiarity
   hurts."** — **FALSE for coreference**. Corrected in §17.

---

## 7. Datasets — vision

**Principle:** competitive exclusion must be *annotated ground truth*, not incidental.

| Dataset | Role | Key facts |
|---|---|---|
| **KINS** (KITTI) | **k-level preemption** | 14,991 imgs, 190,076 instances. **Integer occlusion order**. Max order > 6 in only **1.54%**. |
| **COCOA** | non-driving companion | 5,073 natural scenes |
| **DRAMA** | **graded/configural, same-class** | 17,785 2-s scenarios, 91 h Tokyo. **71.9% vehicles**. Limitation: m_j = 1, no ranked order. |
| **DTLD** | motivating domain | ~230k annotations; *relevancy* only **two-level** |
| **Cityscapes-PPS / PASCAL-PPS** | primary dense benchmark | cardinality varies 4× *within* one object |
| **CLEVR-Ref+** | controlled sweep | where V8 trains c_φ |
| **COCO** | harm check only | never a headline |

- ❌ **PACO retracted** — merges same-class parts; no per-instance assignment.
- **OOSIS** formalizes the occlusion-order task + metrics. **InstaOrder** (101K/2.9M) in reserve.
  **Rank2Tell** cited as motivation only.
- **Gap stated in the paper:** KINS ordering is *geometric*; the motivating hierarchy is
  *normative*. No public benchmark annotates a ranked normative control hierarchy.

---

## 8. Experiments

- **Staged gates S0–S4** with pre-committed pivots. **S2 (~30 H100-h) is the decision point.**
  ⚠️ S3's gate criterion changed in §27.3: it was "the fraction of keys satisfying `m_jδ_j > W_j`
  is substantial", which is no longer a meaningful test; it is now **the interval-hit rate above
  chance**.
- **V1** replication probe *already run*: slopes softmax +0.163, +gate +0.101, Sinkformer
  +0.167, MarSea −0.000. ⚠️ Figure is a **watermarked reconstruction** — replace. ⚠️ It is a
  *possibility result*: `c_j = 1` fixed and Stage 2 inert.
- **V4** measures (m_j, δ_j, W_j). ⚠️ As of §27.3 it reports the **relative interval width
  `m_jδ_j/W_j`**, not the feasibility fraction. **V4b** is the interval-hit rate, now scored
  against the sharp interval. **V4(d)** truncation-compatibility margin, a *recall* diagnostic.
- **V6** degeneracy watch. **V8** synthetic cardinality; trains c_φ.
- **V9** — accuracy vs. occlusion depth k on KINS. **Report the slope, not the pooled mean.**
- **V10** — gate fidelity vs. annotated participation (§13).
- **V11** — **fidelity**: log S_j, report (P_j, R_j, ρ_j) per key stratified by m_j, against the
  matched-sparsity baselines. Carries the central quantitative claim (§12).
- **V12–V14** — language/coreference protocol (§17), 40 H100-h from slack. After S3.
- **V15–V18** — multivariate-reasoning protocol (§21), 220 H100-h as a *separate* line item.
  ⚠️ V15 requires the length-only control or it is confounded.
- **Compute:** 1128 A100-h / **565 H100-h**; 672 h available → 107 h slack.
  ⚠️ **The ICLR draft's Table 4 did not sum** (`1+15+38+90+80 = 224` against a stated `≈275`).
  **Fixed in §27.7**: the stages are printed as run estimates summing to a stated subtotal of
  224, the `1.5×` debugging allowance is applied once at the bottom, and the total is `≈336`,
  which is now arithmetically checkable. ⚠️ **This changed the paper's headline compute figure
  from 275 to 336 — Luke should confirm which reading of the multiplier he intended.**
  `marsea.tex`'s own table was checked and **does** sum (1128 A100-h / 565 H100-h).
- **Baselines that exist to pre-empt one objection:** row-entmax/top-k row, and **softmax at
  matched sparsity**. ⚠️ **The decisive one — column-entmax at uniform `c_j` — is named in the
  text but absent from Fig. 1** (audit §6.7).
- **Backbones:** YOLOv12 area attention; DINO **decoder self-attention**; ViT.

---

## 9. Open items

1. ~~**Row-side shape control.**~~ — **CLOSED 2026-08-26.**
2. **Page limit — BACK IN FORCE for the ICLR draft.** 9 pp main text, hard. ⚠️ As of §27 the
   margin is **zero**: the Tier A corrections consumed every line of slack and were paid for by
   moving E4–E6 and two numbered results to the appendices. Any further main-text addition needs
   a matching subtraction, and §27.5 records what actually moves the boundary.
3. **v1_probe.pdf** is a watermarked reconstruction. **The single most visible soft spot**, and
   now also what blocks a confirmed page boundary. See §0.6 item 1.
4. ~~`joca_ws.tex`~~ — **DROPPED**.
5. **No graded-competition ground truth** for lane-preference examples.
6. **No large-scale split-antecedent resource** — 697 gold instances total, 110 in test (§17).
7. **Typicality stress test not carried over** from the Cona-sp era.
8. **KV-cache interaction** in the causal form. Ablated, not resolved.
9. ~~Project instructions still say ICLR~~ — correct as of §0.1.
10. Author's threads: further theory inspection; re-examining the motivating example.
   **Do not rewrite motivating examples unilaterally.**
11. **Table 7 / Table 4's compute estimates are the last un-reproducible numbers.** They now
   sum (§27.7), but they remain estimates rather than measurements — the one standing exception
   to §20.4's rule.
12. ⭐ **Most of Tier C is unapplied** (Tiers A and B are done — §27; audit §6.4 is discharged
    by §28). See `claude/marsea_iclr_audit_2026-08-28.md` §6–§7. What remains is Luke's call and
    includes the strongest reviewer attacks: Prop. 2's near-vacuity in a decoder where
    `n_q = n_k`, Thm. 1's hypotheses not being jointly satisfiable in self-attention,
    precision-side guarantees against a recall-critical task, and two mischaracterisations in
    related work (Slot Attention, expert-choice routing).

13. ⭐ **Selective exclusivity is stated but not developed.** §28. The four obligations are in
    `marsea.tex` §5.6: the budget anchor `Σ_j c_j = n_q` has no mask-independent replacement;
    `prop:realizability` needs re-deriving on a covered set; fidelity measurement needs
    `E_·j` annotated alongside `T_j`; and every gap reported against `m_j` must also be reported
    against `|E_·j|`. ⚠️ **None of the experiments in either draft evaluate it.**

---

## 10. Venue — see §24 first

⚠️ **This section records the TMLR analysis, which stands on its merits but is no longer the
active plan.** As of 2026-08-25 the target is **ICLR 2027**, with TMLR set aside as the
fallback (§0.2, §24). Read it as the fallback's case, not as the current decision.

**TMLR, targeting October 2026.** TMLR uses rolling submission (no deadline), so October is a
self-set target rather than an external one.

**Why this was the right call:** TMLR's acceptance criteria are only (i) claims supported by
evidence and (ii) some audience would find the claims interesting — explicitly *"accepted...
even if the contribution or significance is modest"* — and there is **no page limit**. Those two
facts dissolve the two largest identified risks: page-limit compression (was 10–15 pts of ICLR
acceptance probability, *controllable*) and unfamiliar benchmarks (was 5–10 pts, *irreducible
under conference review*).

**Consequences for how the paper is written:**
- **Do not compress.** Length is not a cost.
- **The claim bar is "supported", not "beats SOTA".**
- **Reviewers are named and the process is iterative** — pre-committing to pivots is worth more
  here than at a conference.

**Format:** `tmlr.sty` + `tmlr.bst`, `10pt` `article`, `\usepackage{tmlr}` for submission,
`[accepted]` for camera-ready, `[preprint]` for arXiv. Requires `lmodern`. Text width 6.5in
(vs. ICLR's 5.5in).

**If TMLR rejects:** AISTATS is the conference fallback.

⚠️ **A note from the 2026-08-28 hostile-reviewer pass** (audit §4): given that only E1 has been
run, the reviewer's own conclusion was that this is *"a much stronger TMLR submission than an
ICLR one"* — TMLR judges correctness and interest rather than empirical novelty, and the
theory-plus-protocol shape fits its criteria. **This is not a recommendation to switch**; the
venue decision is §0.2 item 1 and is Luke's. It is recorded because it is a reviewer's
independent read of the same facts.

---

## 11. Session-hygiene note

All sessions have been Cowork **cloud** sessions. The first became unreachable through context
overflow; this record exists so that cannot cost the work again. Running "on your computer"
would **not** prevent it. Project docs are the only surviving layer.

---

## 12. Fidelity pillar — precision & recall

Own theory subsection `sec:fidelity`, proofs in `app:fidelity`, **experiment V11**.

**Def 5:** support precision `P_j = |S∩T|/|S|`, support recall `R_j = |S∩T|/m_j`, target mass
`ρ_j = Σ_{i∈T_j} p_ij`.

- **Rem (why counts):** under any normalized allocation mass-precision = mass-recall = ρ_j and
  cannot be separated. The two error types become *definable* only once exact zeros exist.
- **Prop (exact fidelity):** in Thm. 2's interval, P = R = 1 and ρ = 1.
- ⭐ **Prop (the interval IS a PR interval).** ⚠️ **As of §27.2 this is exact on both sides.**
  Lower endpoint `τ ≥ 1/(W+mδ)` ⟺ **precision**; upper `τ < 1/W` ⟺ **recall**. Since the
  interval is never empty, both are attainable at one temperature on **every** column with a
  positive margin; what a hard column costs is the interval's **width**.
- ⭐ **Prop (precision ceiling):** *any* full-support mechanism has `P_j = m_j/n_q` **exactly**,
  whatever its scores or temperature — an identity, not a tendency. ⚠️ **Scoped as of §27.3**:
  under conditional activation an ungated key keeps full-support attention, so on a row with a
  fraction of ungated candidates the ceiling is **rescaled by the inverse gate rate rather than
  removed**. E3 must report that rate.
- **Prop (graded recall):** for `τ > 1/W`, precision stays 1 and `R = k(τ)/m`. Targets shed
  weakest-first — ⚠️ in **non-decreasing** order, ties shedding together (§27.3).
- **Prop (dilution):** column-softmax leaks `≤ ((n_q−m)/m)e^{−τδ}` → holding it at ε is
  *ensured* by `τ ≳ (1/δ)log(n_q/ε)`. ⚠️ **As of §27.3 this is stated as sufficiency**, with
  necessity recovered honestly in the extremal configuration where the bound is tight. The
  `log n_q` story survives; the old "requires" was a modal error.
- ⭐ **Prop (asymmetry):** precision has **ONE** failure path; recall has **THREE**.
  **Recall is the fragile side.** Never report a single F-style aggregate.
- ⭐ **Cor (joint attainment):** P = R = 1 on **both** margins simultaneously.
  ⚠️ **As of §27.3 it carries the consistency hypothesis `K_i = {j : i ∈ T_j}`** that its own
  numerical verification always assumed, the floor bound is quantified as a **minimum over the
  columns claiming `i`**, and row separation is *derived* rather than assumed. The old statement
  was false on any row mixing gated and ungated keys.
- ⭐ **Prop (stage composition) is a ratchet, not a cancellation** (§27.3). Recall cannot rise
  through Stage 2, so a recall deficit between the two readings is Stage 2's. **Precision is not
  bracketed the same way**: `θ_i` is a row threshold while `P_j` is a column quantity, so
  precision can fall as well as rise. The old E7 text said "precision cannot fall through
  Stage 2 … any precision deficit is Stage 1's", and that attribution logic was unsound.

⚠️ **Framing discipline.** Precision is the **centre**; recall is **protected, not traded**.
The word *trade* now has **two** correctly-scoped homes, not three: the two-margin coupling, and
the falsification clauses. ⚠️ **The third — "the region where `m_jδ_j > W_j` fails" — is gone,
because that region is empty** (§27.2). The replacement concession, now in both papers, is that
the interval is never empty but can be arbitrarily **narrow**. Conceding a real difficulty where
it exists is what makes the no-trade claim credible; the narrow-interval concession does that
job now.

---

## 13. Conditional activation (`sec:method-gate`)

⚠️ MarSea is **part-time**, on both margins.

- **The gates read the participant and its field:** `ω^K_φ(k_j, s_·j)` and `ω^Q_φ(q_i, Ã_i·)`.
  Participation is a property of the *relation*, not of the participant alone. ⭐ A gate reading
  the key alone would be deciding a multivariate question pairwise.
- **Cost is unchanged.** Any statistic of `s_·j` the head needs accumulates by an online
  reduction during the score pass, `O(n_q)` time / `O(1)` memory.
- ⭐ **The fan-in side is triggered by a disjunction** (§24.8). Query `i` enters Stage 2 if
  **(1)** some active key allocated to it **OR** **(2)** its own gate fires. (1) is *induced*:
  the row mixes budget-scaled with softmax-scaled entries and the cap applies to the mixture.
  ⚠️ **The cap bounds that mixture's total; it does not re-commensurate the two scales** (§27.3).
  Eq. 5 applies one offset and one gain, both uniform across `j`, and a group-blind affine map
  cannot correct a group-dependent scale mismatch — it thresholds both groups at the same
  absolute level. This is an approximation the mixture forces, not a repair, and it is why
  Cor. joint is scoped to consistent ground truths on fully gated rows.
- **A checkable relation:** the row activation rate is **bounded below** by the fraction of
  queries the active keys reach. E8 checks the bound.
- ⚠️ **Degenerate optimum:** *both* gates closing everywhere is standard attention. Both
  activation rates are first-class diagnostics from epoch 1. We have a *detector*, not a
  training scheme that provably avoids it.
- **E8** scores gate decisions against annotated participation.

---

## 14. Scaling the fan-out program (`sec:method-scale`)

**Two constructions, different obstacles — do not conflate.**

**Blocking → incrementality.** Per-tile budgets `c_j^(b)`. Per-step cost `O(n_k B log B)`.
⚠️ **This is NOT "independent of context length"** (§27.3) — in a decoder `n_k` **is** the
context length, so the cost is linear in it exactly as attention already is. What blocking
removes is the growth of the per-column program and of the sort factor with the prefix,
replacing `t log t` by `B log B`. The old phrasing appeared three times in `marsea.tex` and twice
in `marsea_iclr.tex` and is now corrected in both. Residual is *emission*, not compute.

**Hierarchical top-K → scale.** Retain top-K per block, merge recursively, one exact program at
the root, always on **original** scores.
- **Prop (exactness):** global solution *exactly* iff `K ≥ k*`, any block size, any merge arity.
  **Verified 0/2000.**
- ⭐ **Intermediate entmax is unnecessary.**
- ⚠️ **`k*` is regime-dependent.** Exclusive columns: median 1, p99 5, max 8. Mixed: max 14.
  Broad: p99 35, max 74. A fixed K in the tens is safe **on gated columns only**.
- ⚠️ **Fixed generous K, NOT a learned per-block c_j.**
- **Tested and refuted:** strided/interleaved partitioning does *not* rescue small K.
- **Fidelity under retention:** `P_j = 1`, `R_j = K/m_j`, `ρ_j = 1` when `K < m_j`. ⚠️ Note
  `ρ_j = 1` **coexists with lost targets**.
- ⚠️ The gate/`k*` interaction is **available, not guaranteed** — the gate is learned, not
  constructed. E8's realized-`k*` report tests it.

---

## 15. Non-vision applicability — general survey

**Hard filter:** dense inner-product attention over a shared key set at moderate n_q. Causal
decoding is **not** an exclusion — see §0.2 item 3.

⚠️ **Correction, 2026-08-23.** This section previously said adding an NLP benchmark cannot fix
the unfamiliar-benchmark risk because "SOTA is LLM-based". **That is false for coreference** —
see §17. It remains true for QA and most generation tasks.

**The attention-mechanism literature's own testbeds** are MT (IWSLT/WMT), small vision, point
clouds, synthetic — not parsing/coref leaderboards.

- **⭐ Coreference — chosen. See §17.**
- **UD dependency parsing** — biaffine parsing *is* attention producing a head distribution;
  the Chu–Liu–Edmonds fix is **Prop 12 in the wild**. Held in reserve.
- **IWSLT14 de-en + alignment error rate** — community home testbed. Second reserve.
- Also viable: HotpotQA/MuSiQue supporting-fact selection; entity linking; SRL.
- **Poor fit:** multiple-choice reasoning (ARC/MMLU/HellaSwag) — one-winner but **no scope
  coupling**.

---

## 16. Audit trail

`claude/marsea_audit_2026-08-23.md` holds the itemized change log of the consistency pass that
made §3.1 describe all four components, added V11, added the matched-sparsity baselines, and
added the three missing Limitations paragraphs. Its "Still open" section is superseded by §9.

⭐ **Two newer audit documents supersede it as the live record:**
`claude/marsea_iclr_audit_2026-08-28.md` (the adversarial audit of the ICLR draft — roughly 45
findings, triaged into Tiers A/B/C) and `claude/marsea_iclr_tierA_applied_2026-08-29.md` (what
Tier A did, with the judgment calls listed for reversal).

---

## 17. Referential exclusion in language — the second domain (`sec:language`)

**The user's insight, 2026-08-23:** *"it/its", "she/her/hers", "he/him/his", "they/them/their"
refer to one specific entity/entity group exclusively. This is exactly a multivariate exclusion
relation MarSea has advantages in.*

This is correct and is now a full section of the paper. It is the **only application outside
vision that meets the scope conditions without reformulation**.

### 17.1 ⚠️ The route analysis — read before citing any number

A singular pronoun has **m_j = 1**, which is **exactly the case Prop 12(i) concedes** to
query-local schemes. GAP, WSC273, WinoGrande, WinoBias and Winogender are all
one-pronoun-one-antecedent, so **they test the conceded half and a gain there supports nothing.**

The load is carried by **three routes only**:

| Route | Mechanism | Why non-product-form |
|---|---|---|
| **1. Disjoint reference** | *John told Bill that **he** should call **him*** | A constraint on a **pair of rows**. Thm 1 exactly. |
| **2. Split antecedents** | *John met Mary. **They** left* — antecedent is a **set** | m_j > 1, target set chosen **jointly**. |
| **3. Bounded entity fan-out** | A salient entity is not the antecedent of every pronoun | **Prop 12(ii) verbatim** = over-clustering = the **attention sink** with annotated ground truth. |

**The Winograd schema is a hand-built existence proof of non-product-form**, constructed decades
early for another purpose.

### 17.2 The lexical grounding of Cor. 3 — the sharpest new argument

| Expression | Target set | Regime |
|---|---|---|
| *he/him/his*, *she/her*, *it/its* | m_j = 1, singular | ranked; large τ_j |
| *they/them/their*, split antecedent | **m_j > 1, set chosen jointly** | multivariate; moderate τ_j |
| *they*, epicene singular | m_j = 1 | ranked; large τ_j |
| *each other*, *one another* | m_j ≥ 2, reciprocal | multivariate; moderate τ_j |

⭐ **The required exclusivity is a function of the key's content in the most literal sense
available — the identity of one token.** And *they* is **ambiguous between the two regimes**, so
a global τ must compromise. **V13 is the falsification**, and it is nearly free.

### 17.3 The architectural fit

- **e2e-coref is query-local by Def. 1.** `P(y_i) = exp s(i,y_i) / Σ_{y'∈Y(i)} exp s(i,y')`
  with dummy antecedent ε. **Prop 4 applies verbatim.** ⭐ The dummy ε is a near-miss: a per-row
  escape valve exactly like the attention-sink remedies, and like them it relieves the row
  without constraining the column.
- **Higher-order inference** exists *because* independent per-mention scores over-commit.
  ⚠️ **Do NOT claim HOI works** — Xu & Choi found its impact "negative to marginal". The claim
  is narrower and better: **the coupling was sought in the right place and applied at the wrong
  layer.**
- **Scope conditions met without reformulation:** bidirectional encoders, shared candidate key
  set, post-pruning candidate counts in the hundreds.

### 17.4 ⭐ The venue-risk argument runs the *other* way here

| System | Class | Params | CoNLL F1 |
|---|---|---|---|
| s2e-coref (Kirstain 2021) | Longformer-large | 494M | 80.3 |
| wl-coref (Dobrovolskii 2021) | RoBERTa-large | 360M | 81.0 |
| ASP (Liu 2022) | FLAN-T5-xxl | **11B** | 82.5 |
| Link-Append (Bohnet 2023) | mT5-xxl | **13B** | 83.3 |
| **Maverick_mes (2024)** | **DeBERTa-large** | **504M** | **83.6** |

**A 504M encoder beats 11–13B seq2seq systems.** GPT-4 zero-shot is **62.9** (its failure is
mention detection: with gold mentions it reaches 88.4). So a mechanism evaluated in a ≤500M
encoder is **in the competitive model class for this task**, which is not true of most language
benchmarks. Maverick's own subtitle — *"Defying Recent Trends"* — is a citable precedent.

### 17.5 Data, including what is unfavourable

| Resource | Size | Reaches route |
|---|---|---|
| OntoNotes / CoNLL-2012 | 1,940/222/222 docs, 1.3M words | 1, 3 |
| **ARRAU** (Release 2) | 552 docs, 348K tokens | **2** |
| — split antecedents therein | **697 total; 507/80/110 split** | **2** |
| CODI-CRAC dialogue | 4 corpora, dev/test only | 2 |
| LitBank | 100 works, 210K tokens, **singletons annotated** | 3 |
| GAP | 8,908 pairs, 2 candidates | *conceded* |
| WinoBias / Winogender | 3,160 / 720 sentences | *conceded* |
| WSC273 / WinoGrande | 273 / 44K | *conceded* |

⚠️ **ARRAU is the only gold split-antecedent resource, and it is tiny.** 110 test items. Report
with confidence intervals, treat as a diagnostic, and say so. Also LDC-licensed.

⚠️ **OntoNotes annotates neither singletons nor split antecedents.** LitBank and PreCo annotate
singletons and are the supplements for route 3.

**The conceded rows are still run, as controls.**

### 17.6 Protocol — V12/V13/V14, 40 H100-h, gated on S3

- **V12 (route 3)** — entity fan-out vs. document length on OntoNotes + LitBank.
- **V13 (route 2 + Cor. 3)** — stratify learned τ_j by surface form. **Sharpest and nearly free.**
- **V14 (route 2)** — ARRAU split antecedents, Universal Anaphora Scorer. **The only place where
  m_j > 1 is annotated directly** — and the most underpowered.

### 17.7 Reasoning models — the argument, and why we decline to claim it

**What transfers conceptually:** every referring expression in a chain of thought refers to
exactly one item in reasoning state. **Variable binding under lexical scope is the ranked form
in its purest shape.** The predicted failure mode is a chain that stays **locally coherent while
being globally wrong**.

**Where the claim now stands:** architecture is **not** an obstruction, and end-task metrics
**are** valid performance evidence. What remains true is the labelling distinction — a gain on a
single-target benchmark is evidence for the training effect, not for Thm. 1.

---

## 18. Both-margin fidelity, and the stage-composition result (2026-08-23)

**User instruction:** *"make sure the Theory, especially Fidelity analysis, accounts for both the
fan-out program and fan-in program."*

### 18.1 ⚠️ Symbol collision found and fixed

`R_i` meant **two different things**: the pre-truncation row mass and row recall.
**Resolution: pre-truncation row mass is `R̃_i`**, and `R_i` is reserved for row recall.
⚠️ `R̃_i` had been dropped from the ICLR notation table and is back as of §27.3, because the
corrected `τ_i·R̃_i` regime statement needs it. It is flagged there as *not* row recall.

### 18.2 What was added

- **Def. 5 now defines both margins.** Five quantities, not three.
- **Rem. `rem:rowmass` — why the row has no single mass counterpart.**
- **Precision ceiling now proves both**: `P_j = m_j/n_q` **and** `P_i = |K_i|/n_k`.
- **Prop. `prop:rowgraded` — graded row recall.** ⚠️ Its *rate* `k(θ_i)` was asserted and never
  defined in the ICLR draft; defined and proved as of §27.3.
- ⭐⭐ **Prop. `prop:stagecomp` — composition of the two stages.**
  1. `S^(2) ⊆ S^(1)`, so **Stage 2 can only lose recall**.
  2. **`P^(1) = 1 ⟹ P^(2) = 1`, unconditionally.**
  3. If `P^(1) < 1`, Stage 2 *may* raise precision — ⚠️ **and may also lower it** (§27.3).
  4. **A Stage-1 false negative is unrecoverable.**

### 18.3 ⭐ Why this matters — the ratchet, and the stage ordering

**Precision errors are repairable downstream; recall errors are not.** ⚠️ **The stages do NOT
"cancel in precision"** (§27.3) — the ratchet is one-directional. Consequences:

- It turns the failure-path count from an *enumeration* into a *consequence*.
- **It independently justifies the stage ordering.** Running the exclusion-producing stage first
  and the repair-capable stage second is the order in which (ii) is available. **This answers
  "why this order?" with a theorem rather than a rationale.**

### 18.4 Failure-path count, four-way

| | Paths | Which |
|---|---|---|
| Column precision | **1** | Stage 1 under-exclusive (τ_j too low) |
| Column recall | **3** | τ_j too high; θ_i too high; K < m_j |
| Row precision | **1** | θ_i too low |
| Row recall | **2** | θ_i too high; **inherited** Stage-1 false negative |

⚠️ The audit notes the row count omits at least two precision paths once the gate and the causal
approximations are in scope — ungated keys injecting full-support entries, and frozen-prefix
over-admission. The qualitative conclusion is unaffected; the tally is Tier B.

### 18.5 Numerics — all zero violations

Consolidated into `verify_all.py`. See §20.2 for the current counts.

### 18.6 Experiment consequences

`rem:fidmeasure` has **four** consequences. **Report all four count quantities.** ⚠️ The fourth
changed in §27.3: it read the recovery condition as an operating-point question via
`m_jδ_j > W_j`; it now reports the **relative interval width** and the interval-hit rate, since
feasibility no longer varies.

---

## 19. ⚠️ Content-loss incident, 2026-08-23 — process note

Two edits from the audit pass were found **silently missing** from both `marsea.tex` and the
delivered PDF, most likely from a stale `cp` between `/home/claude/marsea.tex` and
`/home/claude/build/marsea.tex`:

1. The **§3.1 four-component overview** had reverted to "two stages, one per margin".
2. The **Fidelity and Conditional-activation groups of Table 1** were gone entirely.

Both were restored. A `grep` sweep confirmed everything else from that pass survived.

**Lessons:**
- **Work in exactly one directory.** Never `cp` *back into* build.
- **Verify edits in the rendered PDF, not only in the `.tex`.**
- A separate real bug this surfaced: the notation table had floated to the **end of the
  document**. Now `[!htbp]`.

⚠️ **This incident is the reason a full re-emission of a project doc is treated as a hazard**
(§0.4). Where a doc is large enough to come back as a local file path, edit it surgically
instead.

---

## 20. Appendix B rebuilt: all verifications consolidated and re-run (2026-08-23)

**User instruction:** put *all* numerical verification into Appendix B.

**Action:** wrote ONE self-contained harness (`verify_all.py`, numpy-only, <35 s on one core,
each block separately seeded) covering **every** row, and rebuilt Appendix B as five
subsections. The paper now ships the code that produces its own table.

### 20.1 ⚠️ A claim that had to be weakened — `k*` is regime-dependent

The paper asserted, in **two** places, *"over 2000 random columns the largest `k*` observed was
13"*. Re-measuring across a wider exclusivity range shows the flat claim is **not supportable**:

| Regime | median `k*` | 99th pct | max |
|---|---|---|---|
| Exclusive, τ ∈ [0.5, 6] | 1 | 5 | **8** |
| Mixed, τ ∈ [0.1, 20] | 1 | 7 | 14 |
| Broad, τ ∈ [0.02, 1] | 4 | **35** | **74** |

⭐ **But the corrected claim is better than the one it replaces**, because it exposes an
interaction: **the gate can admit a column only when the column it reads is well separated —
exactly the exclusive regime where `k*` is small.**

⚠️ **This interaction is *available*, not guaranteed.** The gate is **learned, not constructed**.
Do not harden this language back.

⚠️ **It took two attempts to land this correction.** See §0.5.

**Consequence for experiments:** V10 reports the realized `k*` distribution **jointly with the
gate decision**. Falsification: if `k*` on gated columns approaches `K`, **no V11 fidelity number
is meaningful until `K` is raised.**

### 20.2 Trial counts — current as of 2026-08-29

⚠️ **The recovery block was rewritten in §27.2** and the table gained two rows. Every other count
is unchanged and was reproduced exactly, which is what validated the transcription.

| Claim | Count |
|---|---|
| Thm. 2, **sharp** recovery interval | **0 / 6000** (no rejection) |
| Thm. 2, **lower endpoint sharp** (`P_j<1` below it) | **0 / 12000** |
| Thm. 2, **upper endpoint sharp** (`R_j<1` above it) | **0 / 10098** |
| Prop. 22, fidelity under retention | **0 / 7200** (no rejection as of §28.5) |
| Prop. 20 (all three parts) | 0 / 17928 each |
| Prop. 19, graded row recall | 0 / 18000 |
| Prop. 23 (all three parts) | 0 / 4000 each |
| Cor. 5, joint attainment | **0 / 3000** (no rejection as of §28.5) |
| Prop. 24(iii) row sums | (0,1,2,0,2) |

**Every claim verifies at zero violations, over 394,149 checks** (§28.4; it was 146,842 before
the selective-exclusivity block, and ⚠️ the 146,842 figure itself under-counted the shipped
program by 2,000 — see §28.6).
⚠️ The old ICLR figure "36,355 accepted trials" was the sum of five blocks read as a grand total;
corrected in §27.3.

⚠️ **No block uses rejection sampling any more** (§28.5). Retention and joint attainment were
the last two, and both were sampling `τ_j` from the *old sufficient* interval `[1/(m_jδ_j),1/W_j)`
and discarding the columns on which it was empty — a residue of §27.2 that the Tier A pass fixed
in the recovery block and nowhere else. Both now sample the sharp interval and reject nothing.

### 20.3 What B.4 concedes

They catch algebra errors, off-by-one errors in sorted-order arguments, and vacuous hypotheses.
**They are not evidence about real data.** ⭐ The last of the three is how the non-sharp endpoint
of Thm. 2 came to light (§27.2), once the block was rewritten to test the two failure directions
rather than only the interior.

Also documented: the joint-attainment test draws `{T_j}` as a **partition** so that
`K_i = {j : i ∈ T_j}`. ⚠️ **That is precisely the consistency hypothesis Cor. 5 now states**
(§27.3) — for two days the verification tested a hypothesis the theorem did not carry.

### 20.4 Standing lesson

**Any number in the paper that came from a script should be reproducible by a script the paper
ships.** Demonstrated in fact on 2026-08-25 and again on 2026-08-29. The two figures that still
do not satisfy it are `v1_probe.pdf` and the compute estimates in Table 7 / Table 4.

---

## 21. Causal form as a native component + the reasoning plan (2026-08-23)

**User rejected all three of my objections to applying MarSea to reasoning models. All three
rejections were correct.**

### 21.1 ⚠️⚠️ STANDING INSTRUCTION: the causal/prefix form is NATIVE DESIGN

**User, verbatim:** *"for attention decoders, MarSea adapts its fan-out program on key j to only
scoring the queries before it -- the prefix, and that adaptation should be considered as an
integrated part of MarSea's native design. The same to MarSea's fan-in program. ... do not raise
this as an non-implemented issue again in the future."*

**Never again frame causal decoding as an obstruction.** In a causal model there is **no
counterfactual "full" column** that the prefix approximates.

### 21.2 ⭐ New theorem this produced — Prop. 23 (prefix monotonicity)

- **(i) ψ_j^(t) is non-decreasing as the prefix grows.**
- **(ii) Exclusion is permanent.** Admission is provisional; exclusion is not.
- **(iii) The frozen-prefix variant OVER-admits**: `supp(frozen) ⊇ supp(exact)`.
  ⚠️ **Its error is one-sided in *support*, not in *fidelity*** (§27.3). `P_j` and `R_j` are
  measured against `T_j`, not against the exact support, and Prop. 23 assumes no margin
  condition, so an extra admitted index may itself be a target and precision may **rise**. The
  two coincide only under Assn. 2 with `τ_j` in the recovery interval.

**Verified 0/4000 on each.**

⭐ **Why this matters:** prefix growth is *one more edge-removing operation*.
1. The streaming implementation is **eviction-only**.
2. **The two approximations compose in complementary directions.** Frozen-prefix errs by
   over-admitting — ⚠️ **the error class Stage 2 can remove at all, rather than the one it
   provably cannot.** That Stage 2 *will* remove any given over-admission is **not** claimed
   (§27.3); Prop. 20(iii) is a "may".

Limitations: a **KV-cache interaction**, with two emission policies ablated.

### 21.3 The other two corrections

**(2) No ground truth ≠ no experiment.** The right structure is a **division of labour**:
annotated domains supply **mechanism** evidence; reasoning benchmarks supply **performance**
evidence.

**(3) Scoring is a strength, not a confound.** Multi-hop **evidence** selection has m>1. And
τ_φ and c_φ are trained, so the exclusion-shaped objective **backpropagates into the scoring
function**. That is a mechanism effect.

⚠️ **The one distinction worth keeping:** a gain on a **single-target** benchmark is evidence for
the *training effect*, not for Thm. 1; a gain that **grows with m** is evidence for both.

### 21.4 ⭐ The headline prediction — and it explains a known phenomenon

With `m` gold facts among `n` candidates:
- Prop. 14 pins any full-support mechanism's evidence precision at **exactly `m/n`**.
- Prop. 21 makes the temperature needed to hold leakage fixed grow like **log n** — ⚠️ in the
  **worst case** over configurations meeting Assn. 2 (§27.3).
- MarSea's support size involves **no n at all** — ⚠️ **on rows whose candidate keys are gated
  in** (§12).

**So the theory predicts that evidence selection degrades with candidate count for
architectural reasons.** Degradation of that shape is exactly what "lost in the middle" reports.

⚠️ **The audit's sharpest unanswered objection lives here** (Tier C): `δ_j` is a min minus a
**max over the non-target pool**, so it *decreases* in `n`. Both bounds scale as `1/δ_j` and the
honest difference is a `log n` factor. Presenting one with `n` displayed and the other with `n`
hidden in a symbol is a comparison a theory reviewer will contest.

### 21.5 Datasets (all verified against primary sources)

| Dataset | Size | m | n | Evidence annotated |
|---|---|---|---|---|
| **MuSiQue-Ans** (primary) | 24,814 | **2,3,4 by hop** | 20 paras | supporting paragraphs |
| HotpotQA distractor | ~113K | 2 fixed | 10 paras | supporting **sentences** |
| 2WikiMultihopQA | 192,606 | 2 (4 bridge-comp.) | 10 paras | sentences + triples |
| **RULER** MV/MQ-NIAH | 500/length | **configurable** | **configurable** | synthetic |

- **MuSiQue is primary because gold count = hop count.** Not saturated.
- HotpotQA secondary only. RULER is the controlled grid.
- ⚠️ HELMET: synthetic recall does **not** predict downstream.

### 21.6 Protocol V15–V18, 220 H100-h, gated on S3

- **V15 (headline)** — sweep n at fixed m.
- ⚠️⚠️ **MANDATORY CONTROL:** Du et al. show **13.9%–85% degradation from context length alone**.
  Every sweep point runs twice. **Without this the result is confounded and worthless.**
- **V16** — sweep m. Gap flat in m ⟹ training effect; growing in m ⟹ Thm. 1 doing work.
  **Both outcomes reportable**, pre-committed.
- **V17** — single-target control (m=1). A *large* gap is the interesting negative.
- **V18** — gate activation + realized k* stratified by whether the position is a gold fact.

**Positioning discipline:** compare against **the same decoder with unmodified attention**.
**Declare fine-tuned vs zero-shot loudly.** Frame as "we extend the standard k-document protocol
from m=1 to m>1".

---

## 22. Session log — what happened when

| Session / date | What was settled |
|---|---|
| **S1** (lost, ~2026-07 to 08-14) | Cona-sp predict-then-verify gate → proved to sit *inside* the query-local class → pivot to JOCA. Column-only theory. |
| **S2** (2026-08-15 → 08-22) | Stage 2 (fan-in) added. Notation table, traffic-intersection example, k-level datasets, JOCA→MarSea rename, positive theorems, fidelity pillar (§12), conditional activation (§13), hierarchical selection (§14), venue analysis. Master record created. |
| **S3** (2026-08-23, part 1) | Rigorous audit pass: §3.1 rewritten to four components, V11 added, matched-sparsity baselines added, three Limitations paragraphs added. |
| **S3** (2026-08-23, part 2) | Workshop paper dropped. **ICLR → TMLR.** §6 referential exclusion in language added (§17). |
| **S3** (2026-08-23, part 3) | Both-margin fidelity + stage composition Prop. 20 (§18). `R_i` symbol collision fixed. Content-loss incident (§19). |
| **S3** (2026-08-23, part 4) | Appendix B rebuilt; **k\* claim weakened and made regime-conditional** (§20). |
| **S4** (2026-08-24) | Causal/prefix form as native component + Prop. 23 (§21). Reasoning objections withdrawn. V15–V18 protocol. |
| **S5** (2026-08-25) | **Propagation pass (§23):** four stale-consistency defects fixed. Paper independently rebuilt; `verify_all.py` re-run. |
| **S6** (2026-08-25, evening) | ⚠️ **Pivot to ICLR 2027 (§24).** Reasoning-only 9 pp draft built. Chowdhury read and cleared. E1–E9 with pre-committed falsifications. |
| **S7** (2026-08-26) | ⭐⭐ **Six mechanism revisions in one day** (§24.6–§24.13). `β_φ` deleted, `τ_i` added, fan-in trigger a disjunction, every predicted parameter made to read its field. |
| **S8** (2026-08-28) | ⭐⭐ **`marsea.tex` caught up to `marsea_iclr.tex`** (§25). Nine theory statements invalidated by the port and restated. |
| **S8b** (2026-08-28, second pass) | ⭐ **Heads separated per margin** (§26). Confirmed `a/τ_i` is not a typo. |
| **S10** (2026-08-30) | ⭐⭐ **Selective exclusivity added to both drafts (§28).** The restriction-transfer pass; Assn. 2 shown asymptotically vacuous on a whole column and the scope margin shown `n_q`-free; `prop:conservation` weakened to `μ/n_k` under a mask; the `|E_·j|=1` degeneracy proved invisible to the existing diagnostics; two residual rejection-sampling sites from §27.2 fixed; harness at **394,149 checks**. ICLR 26→29 pp with the main text still ending p9; `marsea.tex` 68→76 pp. |
| **S9** (2026-08-29) | ⭐⭐ **Adversarial audit of `marsea_iclr.tex` (§27), Tier A applied to BOTH drafts.** Thm. 2's endpoints made **sharp**, which repaired a **false** `cor:necessity` and a false half of `prop:printerval`; `cor:joint` given the consistency hypothesis its own verification assumed; the `τ_i·R̃_i` regime error corrected; the wrecked `prop:tournament` proof rewritten; three §25.2 fixes finally propagated **back** to the ICLR draft; `verify_all.py` rewritten and re-run. Main text held at 9 pp. |

### 22.1 Corrections that reversed earlier advice — the list

Every one of these was stated confidently and later found wrong. **A new session should assume
more remain.**

| Claim | Status | Where |
|---|---|---|
| Remark 7: row cardinality unavailable "by Thm 1" | **FALSE** — Thm 1 is about the column | §6.1 |
| Deformable DETR encoder as a host | **Structurally impossible** | §6.2 |
| "MarSea trades recall for precision" | **Wrong framing** | §12 |
| "SOTA is LLM-based so a small encoder embarrasses itself" | **False for coreference** | §17.4 |
| "Largest k\* observed was 13, so fixed K in the tens is safe" | **Regime-dependent** — max 74 | §20.1 |
| "Causal decoding needs an unimplemented reformulation" | **Not an obstruction** | §21.1 |
| "No ground truth ⟹ no experiment" | **Conflation** | §21.3 |
| "Reasoning benchmarks only test the conceded half" | **Two errors** | §21.3 |
| "The k\* correction has been applied" (2026-08-23) | **Incomplete** — two of three sites | §20.1, §23 |
| "`Σ_j c_j = n_q` is a constraint the program must satisfy" | **Over-stated** — a normalization | §24.6 |
| "Gate on the distribution, predict capacity from content" | **Backwards on both halves** | §24.10(b) |
| "Gate and τ heads should read content only" | **Reversed two rounds later** | §24.13 |
| "A predicted fan-in capacity `β_φ` earns its place" | **Retracted by Luke; the result is better** | §24.12 |
| `A_ij = [Ã_ij − θ_i]_+ ≤ Ã_ij` used for support containment | **FALSE once `τ_i > 1`** | §24.10, §24.11 |
| Prop. 10's prefix monotonicity stated unconditionally | **Needs `τ_j` held fixed** | §24.13 |
| "A query may admit nothing: `θ_i ≥ max_j Ã_ij`" | **Unreachable under a unit cap** | §25.2(1) |
| "`\|E\| < n_q/θ_min`" over all queries | **Vacuous** — scoped to binding rows | §25.2(2) |
| "The fan-out bound survives because rows are not normalized" | **Wrong reason** — because `c_j` is non-uniform | §25.2(6), §27.3 |
| "Supply and demand are exact duals" | **Overclaim** — a matched total | §25.2(7) |
| "A constant `τ_i < 1` is exactly `softmax₁`" | **Not a special case** | §25.3 |
| "A programmatic sweep certifies a propagation pass" | **False for semantic consequences** | §25.4 |
| "One `τ_φ` head serves both margins" | **Not well-typed** | §26.1 |
| "`a/τ_i` in Eq. 5 is a typo" | **No** — it fixes the direction | §26.2 |
| ⭐⭐ **"The recovery interval is `[1/(m_jδ_j), 1/W_j)`, non-empty iff `m_jδ_j > W_j`"** | **A *sufficient* interval from a lossy bound.** The sharp one is `[1/(W_j+m_jδ_j), 1/W_j)` and is **never empty**. 27.4% of columns the old condition calls infeasible do recover. | **§27.2** |
| ⭐⭐ **`cor:necessity`** | **FALSE as stated** — inferred non-recovery from a merely sufficient set. Repaired and strengthened by the sharp form. | **§27.2** |
| ⭐ **`prop:printerval`'s precision half** | **False in general**; its own proof retreated to "the extremal configuration", i.e. `W_j = 0`. | **§27.2** |
| ⭐ **"For `τ_i < 1` the cap is slack"** | **Wrong quantity** — the cap binds at `τ_i·R̃_i > 1`, and `R̃_i = 1` only on average | **§27.3** |
| ⭐ **`cor:joint` as stated** | **False on mixed rows** — needed the consistency hypothesis its own verification had always assumed | **§27.3** |
| ⭐ **"precision cannot fall through Stage 2"** (E7) | **False** — `θ_i` is a row threshold, `P_j` a column quantity | **§27.3** |
| ⭐ **"holding leakage at ε *requires* τ ≥ …"** | **Modal error** — sufficiency from an upper bound; necessity holds only in the extremal configuration | **§27.3** |
| ⭐ **"scope one and scope `n_q` coexist"** | **Impossible under a partition** — the max is `n_q−n_k+1` | **§27.3** |
| ⭐ **"`O(n_k B log B)`, independent of context length"** | **False** — in a decoder `n_k` *is* the context length | **§27.3** |
| ⭐⭐ **"Assumption 2 is a mild regularity condition"** | **False asymptotically.** `δ_j` is a min minus a **max over a growing pool**, so it degrades like `√(ln n_q)` and eventually turns **negative**: the assumption fails outright, and every recovery result becomes vacuous rather than weak. Measured: `δ_j < 0` on average by `n_q = 128`. | **§28.2** |
| ⭐ **"MarSea's requirement has no `n_q` in it at all"** | **Overstated as a comparison.** `δ_j` carries `n_q` on **both** sides of Prop. leakage; the honest residual is the `log` factor alone. Now conceded in both drafts. | **§28.2** |
| ⭐ **"The `146,842` figure is the harness total"** | **Under-counted by 2,000** — the ICLR verification table omitted the `prop:rowfid` precision-monotone row that the program prints. Row restored; total is **394,149** after §28. | **§28.6** |
| ⭐ **"Tier A removed the harness's rejection sampling"** | **Only in the recovery block.** Two other blocks kept sampling `τ_j` from the old sufficient interval for a further day. | **§28.5** |

### 22.2 What Luke has been consistently right about, and I was not

On every occasion where Luke pushed back on an analysis, he was right. The pattern in the errors
is mine: I have repeatedly been **too quick to convert an engineering inconvenience into a
theoretical limitation**, and too quick to treat a mechanism's benefit as illegitimate if it
arrives through training rather than through the separation theorem. When his framing and my
caution conflict, take his framing seriously enough to work out what theorem it implies — twice
that produced a real result rather than a re-label.

**S7 added a refinement:** working out a position that is later retracted is not wasted, because
the anchoring analysis is what makes the reversal a two-line change instead of a redesign.

⭐ **S9 adds a second, and it is about my own work rather than his.** The sharp-endpoint error
was mine, it survived four sessions, and it survived *because every check I ran was conditioned
on it* — the harness rejection-sampled on `m_jδ_j > W_j`, so the region where the bound was
loose was never sampled. **A verification that assumes the hypothesis under test cannot falsify
it.** When a condition appears both as a hypothesis in the theory and as a filter in the
harness, that is the place to look first.

---

## 23. Propagation pass and independent rebuild (2026-08-25)

**Occasion:** a fresh session read the whole context set and the full `marsea.tex` cold, looking
for places where a correction recorded here had not reached every site. Four were found. None
was a new error of substance; all four were **stale text left behind by an earlier correct fix**.

### 23.1 The four defects

| # | Site | Defect |
|---|---|---|
| 1 | Limitations, "$K$ is a hyperparameter…" | Still asserted *"k\* was at most 13"* — **the exact figure App. B.3 retracts three pages later**. |
| 2 | §3.1, overview | *"What the **four** components jointly buy…"* — stale once the causal form became the fifth. |
| 3 | Limitations, "Requires dense attention…" | Read as if the mechanism were encoder-only, in tension with fine-tuning a **decoder**. |
| 4 | `rem:pr` | Carried the **same title verbatim** as `prop:printerval` and restated three formal results in prose. **Removed.** |

### 23.2 ⭐ The general lesson — a new class of defect

§22.1's entries are all *claims that were wrong*. These four are different: the claim was
corrected, and **the correction did not reach every site that stated it**. Neither a clean
compile nor a reference check can catch this; only a grep for the old figure can.

The other three share a shape worth noticing: **each was created by a correct improvement.**
Improvements leave residue, and the residue is invisible from the change itself.

### 23.3 The independent rebuild

- **`marsea.tex` → 60 pp**, `pdflatex ×3`: **0 errors, 0 undefined references, 0 undefined
  citations, 0 overfull boxes.**
- **Float placement checked.** Nothing floated to the end.
- **All four edits verified in `pdftotext` output**, not only in the source.
- **`verify_all.py` re-run:** every figure in Tables 8 and 9 reproduced **exactly**.

⭐ §20.4's rule *"any number from a script must be reproducible by a script the paper ships"* was
executed by a session that did not write the original numbers, and it held.

### 23.4 ⚠️ What could not be reproduced

**`v1_probe.pdf` could not be retrieved.** Project *docs* round-trip; project *PDF attachments*
come back as extracted text. Two consequences:
1. **A practical one.** A session asked to produce a compiled PDF needs `v1_probe.pdf` attached
   or a folder connected. Plan for that rather than discovering it at build time.
2. **An argument for §0.6 item 1.** Regenerating the figure would produce a *script*, which is
   round-trippable where a binary is not.

---

## 24. Pivot to ICLR 2027 — the reasoning-only paper (2026-08-25, evening)

### 24.1 What the ICLR paper is

`marsea_iclr.tex`. Title unchanged. **Vision is gone entirely**; **language/coreference is gone
too**. The paper is **reasoning only**, and its application section is multivariate evidence
selection in a causal decoder.

⚠️ **There is no "deferred formal statements" appendix**, as instructed. Each statement that
left the main text for space is stated in prose where it is used, with its formal statement
immediately above its own proof in App. E. ⭐ **As of §27.4 that convention covers two more
results** — `prop:stagecomp` and `prop:fullsupportceiling` — leaving `def:fidelity` and
`cor:joint` as the remaining full statements in the main text.

### 24.2 The experiments — E1 through E9

Each carries an explicit prediction and an explicit falsification.

| | What | Falsification |
|---|---|---|
| **E1** | replication probe (**already run**) | no replication pair on real columns ⟹ Cor. 1 does not bite |
| ⭐ **E2** | **`m`-sweep at fixed difficulty — load-bearing** | gap flat in `m` ⟹ training-effect claim; **pre-committed to reporting either** |
| **E3** | `n`-sweep with **two mandatory controls** | baseline precision not tracking `m/n` ⟹ headline prediction withdrawn |
| **E4** | single-target control, `m=1` | a *large* gap is the interesting negative |
| **E5** | externalisation | corroborates E2, never replaces it |
| **E6** | the **ranked** claim | flat gap across depth ⟹ ranked framing withdrawn |
| **E7** | fidelity, both margins, recall read twice | — |
| **E8** | mechanism diagnostics | `k* → K` ⟹ hierarchy silently truncating |
| **E9** | ablations | — |

**Compute:** five staged gates, **≈275 H100-h**. S2 = 38 h and is the decision point.
⚠️ Implementation time, not GPU time, is the binding constraint. ⚠️ The table does not sum (§8).

### 24.3 ⚠️ How the 9 pages were reached — log, so it is not re-litigated

The draft entered at **11 pp** and converged to 9 by **structural moves, not sentence-shaving**
(about fifteen shaving passes were absorbed by reflow first — worth knowing before attempting
this again).

- **Moved to the appendix in full:** Related work (App. C); blocking/hierarchical selection
  (App. B); compute table and staging (App. G); E7/E8/E9 detail (App. G).
- **Moved to sit above their own proofs:** `lem:prefix`, `prop:tournament`, `prop:prefix`,
  `prop:graded`, `prop:printerval`, `prop:realizability`, `cor:necessity`, `prop:scope`,
  `prop:leakage`. ⭐ **`prop:stagecomp` and `prop:fullsupportceiling` joined them in §27.4.**
- **Deleted outright:** the standalone monotone-exclusivity proposition.
- **Typographic:** `\thm@preskip/postskip` reduced to 5pt; Fig. 1 made a `wrapfigure`.
- **Kept whole, deliberately:** all nine experiments with their falsifications; both
  obstructions; the recovery theorem; the full-support ceiling; the joint-attainment corollary;
  the scope concession; "what we are not claiming".

**Margin is thin — as of §27 it is zero.**

### 24.4 ⭐ The Chowdhury paper — read and cleared

**Chowdhury (2026), *Lost in the Middle at Birth*, arXiv:2603.10123v1.**

**What it proves.** The U-shaped position curve exists **at initialization**, before any training
and independent of positional encoding. Peak-to-trough ≈10² at init and ≈10³ after pretraining:
**training does not fill the valley — it deepens it relatively.**

**Why there is no collision with MarSea.** Four independent reasons, all stated in the paper:
1. **Different explanandum** — *where* a gold item sits vs. *how many* candidates accompany it.
2. **Different measured quantity** — Jacobian norm vs. support precision.
3. **No competing remedy** — it proposes no fix.
4. **It is an ally, not a rival** — an independent instance of the same shape of argument.

⭐ **What it changes.** E3's **position control moves from optional to mandatory.**

**Citations added:** `chowdhury2026birth`, `wu2025emergence`, `herasimchyk2026residual`,
`liu2023lost`, `zhang2025iheval`.

### 24.5 Official style files installed

⭐ **The official style is *more* compact than the stand-in.** The paper came in at 9 pp with
about half a page to spare, so three things were restored to the main text: **Def. 2**,
**Prop. 7 (stage composition)** and **Table 1 (staged compute)**.
⚠️ **Two of those three have since moved out again** — Prop. 7 in §27.4 and Table 1 in §24.8 —
because the slack is gone. Def. 2 remains.

Other changes: statement headings match ICLR's template; `\iclrfinalcopy` present but
**commented out**; title breaks over three lines; the left-margin ruler is correct for
submission; links are **boxed** (the template's default); **there is no `.bib`** — the `.bbl` is
hand-maintained; `v1_probe.pdf` is deliberately **not** in the package.

### 24.6 ⚠️ `Σ_j c_j = n_q` — demoted from constraint to normalization

**The quantity yes, its status in Eq. 2 no.** Every numbered result takes `c_j` as given and
needs only `c_j > 0`.

**What it genuinely buys:** `c/n_q` is a **learned column marginal** — the object transport-based
attention fixes to *uniform*; scale anchoring; and the degeneracy diagnostic.

**What was wrong:** it sat inside a display introduced by *"For each key j independently"*;
nothing said how it is enforced (now: normalize across the **active** keys — this couples
*keys*, not queries, so Def. 1 is untouched); the causal/blocked scope had not propagated; the
gate scope had not either.

⭐ **Why this is better than leaving it as a constraint.** A reviewer who asks "why this
arbitrary total?" now gets an answer *and* is told the theory does not rest on it.

### 24.7 ⭐ Stage independence, the row gate, and the β anchor

**(a)** The §2.1 normalization paragraph was bloated; cut to five lines.

**(b)** ⭐ **Stage 2 does not presuppose Stage 1.** Three consequences: it is what makes the gate
coherent; every row-margin guarantee inherits it; and ⭐ **the converse is the valuable half** —
applied to an ordinary attention row, Eq. 4 is **query-local**, so it bounds fan-in while leaving
fan-out untouched. *The departure from the query-local class is Stage 1's alone.*

**(c)** ⚠️⚠️ **The fan-in cap had no reference point.** Resolved by anchoring at 1 —
⚠️ **superseded by §24.12**, which fixes it at 1 outright.

### 24.8 ⭐ The fan-in trigger is a disjunction

**Luke's specification:** two OR conditions — an incoming allocation from a gated-in key, OR the
query's own gate.

- **(1) is induced and says nothing about query `i`.** ⭐ This is what the anchor is
  structurally for.
- **(2) is intrinsic**, and is the row gate.

**A checkable relation:** the row activation rate is **bounded below** by the fraction of queries
the active keys reach. E8 checks it.

⚠️ Table 1 moved back to App. G here — **the second time it moved**. It is the marginal item at
this page budget. Do not keep re-litigating it.

### 24.9 Appendix E audit, restatements, and the notation table

**Audit result: nothing was missing.** All claims had proofs. ⭐ **But the report was correct
about the symptom:** Sec. 3 had no pointer to App. E, and proofs appeared as bare
`\begin{proof}[Proof of Thm. 1]` with no statement.

**Fixed:** a pointer at the head of Sec. 3, a preamble in App. E, and **restatements**.

⚠️ **Lesson:** **a cut that reached every site and was still wrong.** When compressing,
cross-references are load-bearing, not filler.

**Notation table restored** as Table 2 in App. A.

### 24.10 ⭐⭐ Row exclusivity `τ_i`, and content-only gating

**(a)** `τ_i` closes §9 item 1. Adopted form: `A_ij = τ_i[Ã_ij − θ_i]_+`, so larger `τ_i` ⇒
smaller support, matching `τ_j`. ⭐ Writing it this way keeps `θ_i` a threshold **on `Ã_i·`**, so
no theorem needed restating.

**(b)** Content-only gating — ⚠️ **reversed at §24.13.**

⚠️ **Process failure this round:** a patch script raised on its third replacement and therefore
**wrote nothing**, but printed enough output to look partly successful.

### 24.11 Symbol and theory consistency sweep

**Six real defects found**, four created by §24.10 itself. Most serious: ⚠️⚠️ **`cor:joint`'s
proof was wrong, not merely stale** — it argued `A_ij ≤ Ã_ij`, **false for `τ_i > 1`**.

⭐ **A load-bearing invariant:** the harness sweeps `θ_i` directly, and every row-margin
proposition quantifies over `θ_i`, so all row counts remain valid. ⚠️ **§27.3 sharpens the
*reason*:** `θ_i` is not free — it is the dual, and `τ_i ↦ θ_i` is a strictly increasing
bijection onto `[0, max_j Ã_ij)`, so sweeping `θ_i` over that range is *equivalent* to sweeping
`τ_i`. The conclusion survives; the old justification was a category error.

### 24.12 ⚠️ CORRECTION — the fan-in cap is fixed at 1

**Luke, retracting his own earlier steer.** `β` is gone: **0 occurrences.**

⭐ **Better in three ways.** (1) A **pointwise** guarantee replaces a limitation. (2) Supply and
demand become commensurable in scale — ⚠️ **not "exact duals"** (§25.2(7)). (3) ⭐ One parameter
does both jobs.

**On the title claim:** "content-dependent degree constraints on both margins" survives — the
row's *degree* is content-dependent through `τ_i`.

⚠️ **The likeliest reviewer misreading:** with the cap literally 1, Eq. 4 looks like a
normalization. It is an **inequality at one, not an equality at one**.

### 24.13 ⭐⭐ Every predicted parameter reads its field

⭐ **The decisive argument is a theorem the paper already has.** By Thm. 2 the admissible
temperatures are functions of the score column, so a head seeing only `k_j` **cannot target an
interval it cannot observe**. ⭐ **§27.2 strengthens this argument**: the interval is never empty
but can be **narrow**, so the head is not choosing between feasible and infeasible columns — it
is aiming at a small target whose location only the column reveals.

⭐ **And content-only gating was itself query-local.**

⚠️⚠️ **Prop. 10's monotonicity needs `τ_j` held fixed**, so the gate decision and `τ_j` are
**sealed at the block boundary**.

⭐ **A new experiment this enables:** E7's **interval-hit rate**. ⚠️ **It must be scored against
the sharp interval** (§27.2).

⚠️ **Limitations is a `\paragraph`, not a `\section`**, purely to hold 9 pages. Restore
`\section` for the 10-page camera-ready.

---

## 25. `marsea.tex` caught up to `marsea_iclr.tex` (2026-08-28)

All eight §0.3a divergences ported. `marsea.tex`: 60 pp → **66 pp**, clean; `verify_all.py` re-run
with **every count identical**, which is the `τ_i` invariant of §24.11 holding in fact.

### 25.1 What was ported, and what deliberately was not

**Ported (mechanism):** unit fan-in cap with `β_φ` deleted; row temperature `τ_i`; field-reading
heads; the fan-in disjunction; Stage-2 independence; `Σ_j c_j = n_q` demoted; block-sealing.

**Not ported (length):** the dropped vision material, the cut appendices, the `\paragraph`
Limitations. ⚠️ *Mechanism ports forward, compression does not port backward.*

**Written fresh:** V4b (interval-hit rate), the field-stripping ablation, `τ_i` in the degeneracy
watch, both gates in V10/V18, the row-gate lower-bound check.

### 25.2 ⚠️⚠️ Nine theory statements the port invalidated

An independent adversarial audit found all nine; **six were errors I had introduced or left
standing.**

1. ⭐⭐ **`prop:rejection`(iii) — query-level rejection became unreachable.** Under a unit cap,
   `θ_i ≥ max_j Ã_ij` is inconsistent with KKT. **No admissible floor can empty a non-zero row.**
   Restated: the third level is *inherited* from Stage 1 and the key gate.
2. ⭐ **`prop:edges` became vacuous.** Mean row mass is exactly one, so many rows run at
   `θ_i = 0` **by design**. Scoped to `B := {i : θ_i > 0}`.
3. **`cor:budget` is an ℓ₁ claim** — the only one `τ_i` genuinely breaks. **The "τ_i moves no
   support" defence does not rescue a magnitude claim.**
4. **`prop:realizability` used `{θ_i}` as free witnesses.** θ_i is a *dual*. Witnesses are now
   `{τ_i}`, and the construction yields `A_ij = min(τ_i,1) > 0` at **every** admissible `τ_i`.
5. **A surviving "non-increasing"** where monotone-and-fixes-zero was needed.
6. **`prop:conservation`'s non-applicability justified by the wrong reason.** ⚠️ **This one did
   NOT reach `marsea_iclr.tex` until §27.3.**
7. **"Supply and demand are exact duals" was an overclaim, in the abstract.** ⚠️ **Also did not
   reach `marsea_iclr.tex` until §27.3.**
8. **`prop:special`'s "equivalently at (iv)" was false.**
9. **`θ_i` is reachable only over a bounded range.** ⚠️ **Also did not reach `marsea_iclr.tex`
   until §27.3.**

### 25.3 ⚠️ A false step found in `marsea_iclr.tex` — the shorter draft was the wrong one

The audit was pointed at `marsea.tex` and found, out of scope, that `marsea_iclr.tex`'s
`prop:scope`(iii) proof still contained the **`A_ij ≤ Ã_ij` step that is false for `τ_i > 1`**.

⚠️ **For a stretch, the 66 pp draft was more correct than the 9 pp one it was being caught up
to.** *A propagation sweep on draft A does not certify draft B, even when B is where the change
originated.* ⭐ **§27.3 shows the same lesson in the other direction.**

The `softmax₁` claim was also softened in both.

### 25.4 What this says about the process

⭐ **The audit earned its cost several times over.** Nine invalidated statements; **the sweep
found zero of them** — every one was a *semantic* consequence, invisible to grep. §23.2's "new
class of defect" was about stale claims surviving a correction; this is the class after that:
**claims that are still syntactically valid and have quietly become unreachable, vacuous, or
false.**

---

## 26. Separate heads per margin, and the `a/τ_i` question (2026-08-28, second pass)

### 26.1 Right, and the argument is stronger than "different content-dependence"

1. ⭐ **A single head is not well-typed.** `τ_φ(k_j, s_·j)` maps `R^d × R^{n_q}`;
   `τ_φ(q_i, Ã_i·)` maps `R^d × R^{n_k}`. These coincide only in encoder self-attention.
   ⚠️ **The audit notes this argument is weakened by the paper's own implementation**: if the
   field enters through an `O(1)` streaming summary, both heads take arguments in the same space.
   Tier C.
2. ⭐⭐ **The two heads are aimed at different intervals.** `τ^K_φ` must land in Thm. 2's
   interval; `τ^Q_φ` must place the dual `θ_i` in the row interval. **This is the better
   argument and it is the one in both papers.**

Third, smaller: it decouples the ablation into four independent tests plus two *tying* rows.

⚠️ **Both of Luke's proposed row letters collided** — `δ_i` with the inter-set margin (in the
ICLR abstract) and `κ_i` with the scope bound. **A future session proposing symbols should check
the i/j-conjugate of every candidate letter, not just the letter.**

**Chosen: superscripted heads, quantities unchanged.** `τ_i` and `θ_i` untouched, so all
propositions, proofs and `verify_all.py` needed no change.

**Side benefit:** the gate *outputs* are now named `ω_j`, `ω_i`, so the active key set has a
symbol, `K = {j : ω_j > 0}`.

### 26.2 Not a typo; the placement fixes the parameter's direction

Under `a/τ_i` the effective budget is `1/τ_i`, so larger `τ_i` = more selective, the same
direction as `τ_j`. Under `a·τ_i` **every regime statement in the paper inverts**.

⭐ **But he read it as a typo, so a reviewer will.** Both drafts display the scale-separated form:
`A_i· = τ_i · argmin_{u≥0, Σ_j u_j ≤ 1/τ_i} ½‖u − Ã_i·‖²`.

### 26.3 ⚠️ The 9-page accordion, again — and what actually moved it

**Roughly fifteen rounds of sentence-shaving moved it by less than two lines.** What worked was
structural: moving the positional paragraph to App. D; moving the head-separation argument to
App. D; moving Limitations detail to App. D; ⭐ **repositioning the `wrapfigure`** — the additions
had pushed Fig. 1 from p8 to p9, where it consumed space *and* left p8 under-filled by ~9 lines;
and compressing E4/E5.

**Check float pages, not just line counts** — a float that has drifted costs twice.

**A process note.** The write-before-you-fail rule bit twice more; the second time a
two-replacement script whose *second* assertion failed aborted before writing, silently
discarding the first. **The assertion protects correctness but not completeness** — after a batch
that aborts, re-check every edit in that batch.

---

## 27. Adversarial audit of `marsea_iclr.tex`, and Tier A applied to both drafts (2026-08-29)

**Occasion.** `marsea_iclr.tex` had never received the §25.4-class audit that found nine
invalidated statements in `marsea.tex`, and §25.3 had already shown the short draft could be the
*more* wrong one. Three independent readers were run over it cold — a mechanism-consequence
auditor, a proof-step auditor, and a hostile ICLR reviewer — plus a direct read. **Roughly 45
findings**, triaged into Tiers A/B/C in `claude/marsea_iclr_audit_2026-08-28.md`. **Tiers A and B
are now applied to both drafts** (§27.2–§27.3, §27.7). **Tier C is not**, and is Luke's call.

⭐ **Three readers, low overlap.** Of ~45 findings the three passes agreed on only two, which is
the argument for running more than one and for giving them **different mandates** rather than the
same one three times.

### 27.1 State

| | `marsea_iclr.tex` | `marsea.tex` |
|---|---|---|
| length | **26 pp, main text ends p9**, Fig. 1 p8 (29 pp as of §28) | **68 pp** (was 66; 77 as of §28) |
| build | 0 err / 0 undef / 0 overfull / 0 multiply-def | same |
| claims / proofs | 16 / 16 | unchanged |
| restatements | 7 → **5** | n/a |
| harness | **146,842 checks, 0 violations** (394,149 as of §28) | same file |

⚠️ **The ICLR build used a placeholder `v1_probe.pdf`**, calibrated by first reproducing the
recorded baseline exactly. A sweep over aspect ratios `h/w ∈ [0.60, 0.80]` changed nothing, so
the boundary is **not** sensitive to the figure's height — it was sensitive to its **anchor
paragraph**, now `\paragraph{Setup.}`.

### 27.2 ⭐⭐ The headline: Thm. 2's endpoints are sharp

See §5 for the statement and §20.2 for the counts. In brief: `G(m_j+1) = τ_j(W_j + m_jδ_j)`
**exactly**, so both endpoints are equivalences and the interval is **never empty**. Verified at
**0/6000 with no rejection**, **0/12000** below, **0/10098** above; an independent 200,000-column
sweep found **54,858 columns (27.4%)** that the old condition calls infeasible where recovery is
in fact attained.

**Repaired by it:** `cor:necessity` (was **false**), `prop:printerval`'s precision half (was
false in general), the ICLR abstract's "exactly a precision–recall interval" (was an overclaim),
§2.1's "the admissible temperatures are …", and E7/V4's measurement target.

**Cost:** `verify_all.py`'s recovery block rewritten; Table 3 / Appendix B gained two rows; one
of §12's three "trade" sites is gone.

### 27.3 Everything else applied, to both drafts

**Reverse-propagation gaps.** Three of the nine §25.2 fixes had never reached `marsea_iclr.tex`:
the conservation non-applicability reason, the "exact dual" overclaim (which §2.2 contradicted
one paragraph later), and `θ_i`'s bounded reachability.

**Errors in numbered results.** `cor:joint` gained the consistency hypothesis its verification
always assumed, a quantified `min` over columns, and a corrected biconditional;
`prop:realizability`'s "scope one and scope `n_q` coexist" is impossible under a partition;
`prop:leakage`'s "requires" became "suffices" with necessity recovered in the extremal
configuration; `prop:prefix`(iii)'s fidelity sentence is scoped to support; `prop:scope`(iii)
now proves a cardinality bound via `k*` instead of a mass bound via Markov (ICLR only —
`marsea.tex` was already correct); `prop:graded`(ii)'s rate `k(θ_i)` is defined and proved;
`cor:capacity` states its `c>0`, `1 ≤ m < n_q` hypotheses.

**The `τ_i·R̃_i` regime error.** See §4. Both stated regimes were wrong on non-unit-supply rows.

**Reference and bookkeeping (ICLR).** `prop:tournament`(ii)'s proof had **two circular
self-references and stale sub-numbering** — a fossil of the merge that created it; the
mathematics was always correct. E7's "precision cannot fall" mis-citation corrected. E9's pointer
to its own appendix removed, as was App. D's misattribution of a fan-in bound to Thm. 1 and the
unused `def:productform` label.

**Arithmetic (ICLR).** "36,355 accepted trials" → **146,842 checks**, with attempted counts
beside accepted ones.

**Consistency owed after the conservation fix.** The abstract's "rules out query-side
normalization", §1's "incompatible … at all", §3.1's "the abandonment of query-side
normalization" and contribution (ii) all said something §2.2 now denies. All four restated.

**Cost claim.** `O(n_k B log B)` is not "independent of context length" — three sites in
`marsea.tex`, two in `marsea_iclr.tex`.

### 27.4 ⚠️ Judgment calls — reverse these if you disagree

1. **`prop:stagecomp` and `prop:fullsupportceiling` moved from the ICLR main text to App. E**,
   keeping their content as prose. §24.5 had deliberately restored both. **Reason: this is the
   paper's own convention** — eight other results already work that way and these were among only
   four exceptions. One edit from coming back.
2. **E4, E5 and E6 moved to App. G in full**, with a one-sentence summary each — the treatment
   E7–E9 already had.
3. **The Limitations paragraph dropped its "degenerate optimum" parenthetical**, which App. D
   still carries in full.
4. **Fig. 1's anchor paragraph changed** to `\paragraph{Setup.}`.

### 27.5 ⭐ The page limit closed on a float, not on prose — again

Roughly ten rounds of shaving moved the boundary by under two lines. What moved it: re-anchoring
Fig. 1, moving E4–E6 to the appendix, and **merging two `\paragraph` headings**.

⭐ **Two diagnostics worth reusing.** **Count text lines per page** — p9 held 49 where p7 held 60,
and the gap was `\parskip` from six paragraph headings in §5. And ⭐⭐ **build once with the figure
deleted**: that gave 24 pp with the main text ending on p9, which proved the prose already fitted
and the entire overflow was the float. One build replaced several rounds of guessing.

### 27.6 Method notes

⭐ **Validate the transcription before editing.** Both `marsea_iclr.tex` and `verify_all.py` were
copied into a clean directory and reproduced every recorded invariant exactly before a single
edit. Without that step no later measurement would have meant anything.

⭐ **`marsea.tex` was edited surgically, not re-emitted.** At 286 KB it comes back from
`project_read` as a **local file path**, so all 33 replacements were applied by exact-match
patches with every `old` string asserted to occur **exactly once** before anything was written,
then grep-verified. This is the §19 hazard avoided by construction, and it is worth checking for
any large doc before planning a rewrite.

⚠️ **This record was re-emitted in full**, because it is small enough to come back inline. That
is the risky path; §0.4 explains when it can be avoided.

### 27.7 Tier B applied to both drafts (2026-08-29, second pass)

Tier B was the "scoping and bookkeeping, cheap, no judgment call" tier. All of it is now applied;
the audit document's Tier B list is discharged.

**Statement hygiene.**
- **`κ_j` was used and never defined** in `marsea_iclr.tex` — main text, appendix and notation
  table alike. Now defined at first use as the scope bound the mechanism is asked to respect,
  `|T_j| ≤ κ_j`, with a notation-table entry. (`marsea.tex` already defined it properly, in its
  competitive-selection definition.)
- **Assn. 2's second clause was not an assumption.** `W_j ≥ 0` is a theorem of its own
  definition; only `δ_j > 0` is assumed. Both drafts now say so. This matters for how the
  assumption reads: it had made `m_jδ_j > W_j` look like part of the same package.
- **The empty-support convention** (`P = 1` on an empty support) was introduced mid-proof inside
  `prop:stagecomp`(ii). Moved to Def. `fidelity` in both drafts, where every result inherits it.

**Scoping.**
- **`c_j` cannot be fixed when key `j` is formed** — it reads `ν_j`, a statistic of `p_·j`, so it
  is unavailable until the column program has run. Both drafts now say the per-block budget is
  emitted when the block **closes**.
- **The budget normalization has a streaming consequence** that "admitting a query perturbs only
  its own block" concealed: because budgets are normalized across a block's active keys,
  admitting a query changes **every** `c_j^(b)` in that block. Stated in App. B of the ICLR draft.
- **Table 1's Stage-1 row said the stage acts on `A_·j`**; it produces `Ã_·j`, and that
  distinction is load-bearing in three proofs. Fixed in both.
- **The failure-path tally is a lower bound, not an exact count.** The 3:1 and 2:1 ratios
  enumerate the two programs only; two further *precision* paths enter with the rest of the
  mechanism — an ungated key contributing full-support attention to a gated row, and
  frozen-prefix over-admission. Both are precision-side, so the asymmetry stands. Stated in both.
- **Why some verification rows carry few trials** — `prop:fullsupportceiling` is a deterministic
  identity, so its rows are checked on a small spanning grid rather than a large sample. Said in
  both appendices, so that "0 / 12" beside "0 / 17928" reads as a design choice.

**Arithmetic.** The ICLR compute table now sums (§8). `marsea.tex`'s was checked and already did.

**Framing.** ⭐ **"Degree" is now defined as *weighted* degree** — the sum of incident edge
weights — in both drafts, at the point where the title claim is first made. This is the cheap
answer to the audit's sharpest structural objection (§6.1 there): Cor. 1 names *capacity* and
*cardinality*, MarSea clears capacity by construction and cardinality only under Assn. 2, so a
reviewer reads "degree constraints" as promising more than is delivered. In a weighted graph the
degree of a node **is** its incident weight sum, so `Σ_i Ã_ij = c_j` is literally a
content-dependent degree constraint, and cardinality control follows from the exclusivity. Two
lines, and it removes a reviewer's opening move.

**E1's tense.** ⚠️ §3.1 said *"E1 measures `(a,M)` directly rather than assuming it"* in the
present tense, but the `(a,M)`-measuring version is **S0 and unrun** — the probe already run is
synthetic and constructs near-equivalence by hand. Corrected. This mattered more than its size:
it was an overstatement of evidence at the exact point where the evidence is thinnest, since
Assn. 1 is what Cor. 1's non-trivial half rests on.

**State.** `marsea_iclr.tex` 26 pp, **main text still ends p9**, Fig. 1 p8; `marsea.tex` **68 pp**
(Tiers A and B grew it from 66; it has no page limit). Both
0 errors / 0 undefined / 0 overfull / 0 multiply-defined; 16 claims / 16 proofs; no proof cites
itself; every `\ref` resolves; all 32 ICLR bibitems cited and none uncited.

⚠️ **The page limit went over twice during Tier B and was recovered both times by the §27.5
method** — not by shaving. What paid for the weighted-degree sentence was moving the
per-target-selection explanation of Prop. `scope`(ii) into App. D, and the sentence itself was
**folded into an existing clause rather than added as a new one**, which halved its cost.

---

## 28. Selective exclusivity added to both drafts (2026-08-30)

**Occasion.** Luke proposed on 2026-08-29 that a key is not exclusive over its whole column but
only over the queries it stands in a relation with — a left-turn arrow is exclusive over the
left-turn lanes and not over the through lanes, nor over how many cars are waiting — and that the
same conditionality holds on the query margin. He asked for it on scratch paper first
(`claude/scratch_selective_exclusivity_2026-08-29.md`), then, on 2026-08-30, directed that it be
added to `marsea_iclr.tex` as a next-step generalization **in pre-emption of Tier C questioning**,
and to `marsea.tex` as a fully developed generalized mechanism.

**His diagnosis of the current mechanism was correct and is now in both papers.** The fan-in cap
already gives part of the effect from the row side, but nothing stops a key from *spending* its
budget on queries it has no relation to, because `Σ_i p_ij = 1` ranges over the whole column. The
column constraint is mis-specified and the row constraint has been patching it downstream.

### 28.1 What was added, and where

| | `marsea_iclr.tex` | `marsea.tex` |
|---|---|---|
| main text | one clause in Limitations, plus two pointers at Prop. leakage's own `n_q` claim | ⭐ **new §5, six subsections**, between Theory and Experimental Plan |
| statements | none (appendix prose) | 4 numbered: `prop:selrestrict`, `prop:selmargin`, `prop:selsingleton`, `prop:selconserv`, plus `def:subgraph` and 6 remarks |
| proofs | n/a | new App. A.8, all four proved |
| tables | App. E: transfer accounting + margin measurement + checks | §5: transfer accounting + margin measurement; App. B: 9 new rows |
| honesty edits | Prop. leakage's `n_q`-free claim re-read in App. E | ⭐ new `rem:leakhonest` beside Prop. leakage, and a new Limitations bullet |
| length | 26 → **29 pp**, main text **still ends p9** | 68 → **77 pp** |

**Notation.** `\mathcal{E} ⊆ [n_q]×[n_k]` is the **exclusive subgraph**; its slices are written
in the paper's own margin convention, `E_·j` and `E_i·`, which sidesteps the `F_i` collision with
the local `F_Q` of Prop. 23's proof and the retired `E`. `|E_·j|` is key `j`'s **scope**.

### 28.2 ⭐⭐ The headline: Assumption 2 is asymptotically vacuous on a whole column

This is the strongest thing in the section and it is **not** the argument Luke gave; it is what
the proposal turned out to buy.

`δ_j := min_{i∈T_j} s_ij − max_{i∉T_j} s_ij` is a min minus a **max over a growing pool**. For
i.i.d. non-target scores with unbounded support above, `max → ∞` a.s., so `δ_j → −∞` and
`Pr[Assn. 2 holds] → 0`; for `σ`-sub-Gaussian scores the degradation is `Θ(σ√(2 ln n_q))`.
**Assumption 2 does not weaken at scale — it fails**, and past that point every result in the
recovery section is *vacuous*, not merely weak. No scoring function within the class avoids it,
because only the non-target *location* is learnable and the two effects are additive.

Measured with a scorer that has **already** separated non-targets by two standard deviations
(`|E_·j| = 8`, `m_j = 3`, pool `N(−2,1)`, 3000 columns per row, `τ_j` inside the scope's interval):

| `n_q` | `δ_j` full column | `δ_j` on `E_·j` | recovery, full | recovery, on `E_·j` |
|---|---|---|---|---|
| 8 | 1.127 | 1.127 | 100.0% | 100.0% |
| 32 | 0.296 | 1.127 | 63.6% | 100.0% |
| 128 | **−0.321** | 1.127 | 14.9% | 100.0% |
| 512 | **−0.791** | 1.127 | 1.2% | 100.0% |
| 2048 | **−1.184** | 1.127 | 0.0% | 100.0% |
| 8192 | **−1.555** | 1.127 | 0.0% | 100.0% |

⭐ **This discharges audit Tier C §6.4**, which said `δ_j` hides `n` in Prop. leakage's comparison
and that a theory reviewer would contest it. The answer now in both drafts concedes the point and
then splits the gain in two, which is what makes it credible:

- **(i)** the `log|E_·j|` factor belongs to the **program** — a terminated tail against an
  exponential one — and the mask does not touch it;
- **(ii)** the reduction of the domain from `n_q` to `|E_·j|` belongs to the **mask**, and a
  masked softmax baseline would enjoy it equally, **so any experiment claiming (ii) must include
  one**;
- what restriction adds beyond either is that Assumption 2 stops being asymptotically vacuous,
  so that (i) remains a statement about something.

⚠️ **The `n`-dependence is relocated, not destroyed.** A continuous margin that provably degrades
is replaced by a discrete membership decision that *might* not. Whether membership is easier to
learn than fine-grained separation is an empirical question; for a **typed** relation it
obviously is, for an untyped one it may not be. **This is the claim the experiments would carry**,
and no experiment in either draft carries it yet.

### 28.3 The restriction-transfer pass — what survives, what does not

⚠️ **The scratch note could not honestly say "verbatim" and this pass is why it now can, for a
named list.** §25.4's precedent — a port that looked clean invalidating nine statements — was the
reason to do it statement by statement rather than assert it.

**Verbatim with `[n_q] → E_·j`, `[n_k] → E_i·`** (each is a statement about a sparsemax or a
truncation *over an index set*, and none counts the set): Lem. 1, monotone exclusivity, Thm. 2
with both endpoints, `prop:printerval`, graded recall on both margins, `prop:tournament`,
`prop:hierfid`, `prop:stagecomp`, `cor:necessity`, and Thm. 1 with the product over `E_·j`.

**Hypothesis changes:** Assn. 2's max is over `E_·j \ T_j`; `cor:capacity` and `prop:scope`(ii)
need `|E_·j| ≥ 2` and a replication pair with `M ≤ |E_·j|`; the full-support ceiling becomes
`m_j/|E_·j|` for a **masked** comparator and stays `m_j/n_q` for an unmasked one; `prop:leakage`
takes `n_q → |E_·j|` on both sides; ⭐ **`cor:joint`'s consistency hypothesis becomes structural**
— with one subgraph and `T_j ⊆ E_·j`, `K_i = {j : i ∈ T_j}` holds by construction rather than by
assumption, which retires the awkward hypothesis Tier A had to add.

**Genuinely different, and owed a proof:** see §28.4.

⭐ **Two structural gains worth recording.** Conditional activation is the special case
`E_·j ∈ {∅, [n_q]}`, so **the gate is a degenerate mask and the five heads become four** —
`ω^K_φ` and `ω^Q_φ` are subsumed. And **the two margins must be slices of one object**: with
independent per-margin masks, Stage 2 could floor entries Stage 1 never produced and neither
`prop:stagecomp` nor `cor:joint` survives. Both drafts say so and decline the two-mask reading.

⭐⭐ **Prop. 23 transfers if and only if the mask is *arrival-sealed*.** Its proof compares
`F_Q(ψ) = Σ_{i∈Q}[z_i−ψ]_+` on nested visible sets; restriction replaces `Q(t)` by
`Q(t) ∩ E_·j`, which is nested exactly when membership, once decided, is never revoked. A mask
that can reconsider breaks permanence of exclusion on **90.7% of prefixes** in simulation, with an
explicit three-element counterexample in the proof. ⭐ **This turns the parameterization choice
into a requirement**: because `e_ij = 1[⟨u_φ(k_j), v_φ(q_i)⟩ > 0]` depends on the pair alone, it
is arrival-sealed automatically, whereas a field-reading mask is not. The causal form *forces* a
pairwise mask.

### 28.4 ⚠️ What weakens — three things, and the first is not obvious

**(a) ⭐⭐ The separation theorem's force is parameterized by `|E_·j|` and vanishes at 1.**
Capacity on a singleton **is** product-form, so a query-local scheme enforces it by clipping and
Thm. 1 says nothing; the replication pair must be found inside the scope, so `M ≤ |E_·j|`.
⚠️ **And the collapse is invisible to the existing diagnostics**: there are scoped mechanisms
with `|E_·j| = 1` everywhere and `Var_j(c_j)`, `Var_j(τ_j)`, `Var_i(τ_i)` all bounded away from
zero. This is nastier than the both-gates-closed degeneracy, which is at least visibly standard
attention. **Consequence, stated in both drafts:** report the **distribution** of `|E_·j|`, not
its mean, from epoch 1; carry `|E_·j| ≥ 2` wherever Thm. 1 is invoked; and add a pre-committed
falsification — the gap should vanish as `|E_·j| → 1`.

**(b) Prop. 4 (conservation) weakens to `μ/n_k`.** With `μ := Σ_i Σ_{j∈E_i·} A_ij` the
in-relation mass, `max_j Σ_{i∈E_·j} A_ij ≥ μ/n_k`, and **this is the strongest bound of its
form**: a row-normalized query-local scheme can hold every scoped column sum below any `ε` by
parking mass off `E`. So the second obstruction bites only in proportion to coverage, and the
coverage fraction `μ/n_q` becomes a reported quantity. ⭐ Thm. 1 by contrast transfers intact on
any scope of size ≥ 2, which is why (a) and not (b) binds the design.

**(c) The capacity statement and the budget anchor.** `Σ_i Ã_ij = c_j` becomes
`Σ_{i∈E_·j} Ã_ij = c_j`, so E1/V1's "column mass exactly constant as `m` grows" becomes constancy
**on the scope**. ⚠️ **And `Σ_j c_j = n_q` has no mask-independent analogue** — the natural
replacement counts covered queries, which moves with the mask. **Both drafts say we do not have a
satisfactory answer**, and `marsea.tex` §5.6 lists it first among four obligations.

⚠️ **A framing cost was paid knowingly.** §13's rhetorical asset — *"a gate that cannot see the
field is deciding a multivariate question pairwise"* — is superseded, because under a typed mask
membership **is** decided pairwise. The replacement is better and is in both drafts: *whether this
key and this query stand in the relation* is a question about the pair, while *which of the
competing keys wins this query given the others* is not. **Pairwise decides who is in the room;
the joint program decides who wins.** ⭐ This also *clarifies* the field-reading argument rather
than retreating from it: that argument was always about `τ^K_φ` and `τ^Q_φ` hitting Thm. 2's
interval, a property of the whole column, and never about membership.

### 28.5 ⚠️ Two residual rejection-sampling sites from §27.2 — found and fixed

Tier A rewrote the recovery block to the sharp interval and **left two other blocks sampling the
old one.** `prop:hierfid` and `cor:joint` both computed `lo = 1/(m_jδ_j)` and rejected columns
with `m_jδ_j ≤ W_j`. Neither was *unsound* — `1/(m_jδ_j) ≥ 1/(W_j+m_jδ_j)`, so the sampled set is
a subset of the true interval — but **neither ever tested the lower part of the interval**, which
is exactly the 27.4% of columns §27.2 showed the old condition wrongly excludes.

Both now sample `[1/(W_j+m_jδ_j), 1/W_j)`. Consequences: retention goes **6536 → 7200 accepted**
and joint attainment **3000 of 3063 → 3000 of 3000**, and ⭐ **no block in the harness rejects
anything any more.** The "accepted counts" apparatus in both appendices is retired and replaced by
a note explaining why it is gone.

⭐ **Lesson, and it is §22.2's again in a new form.** A correction applied at the site where the
error was *found* is not a correction applied everywhere the error was *used*. §23.2 named the
class where a fix leaves stale prose behind; this is the class where a fix leaves stale *code*
behind, and it survived a day and a full Tier B pass.

### 28.6 ⚠️ The verification table did not match the program

Reconciling the harness total against the ICLR draft's `146,842` showed the program summing to
`148,842`. The gap is exactly one row: the ICLR verification table **omitted the
`prop:rowfid` precision-monotone check (0 / 2000)** that the program prints. `marsea.tex`'s table
had it. The reproducibility statement's claim that the program *"prints Table 3 exactly as it
appears here"* was therefore false in the ICLR draft.

Row restored; the claim reworded to "reproduces every figure in Tables 6 and 5, rows marked
*each* being printed one per case", which is what the collapsing convention actually does. New
total **394,149 checks, 0 violations**, runtime under 25 s.

**New harness block (11 reported lines):** Thm. 2 on a scope with both endpoint directions
(24000 / 12000 / 10604); the `n_q` sweep of §28.2 (18000, plus the printed margin table);
Prop. 23 under a nested scope (131180); Prop. 4's corrected `μ/n_k` bound and the escapability of
`n_q/n_k` (3000 each); the masked ceiling (4000); and `prop:stagecomp`(i),(ii),(iv) under one mask
(12953 each). ⭐ **Two of these check that something *fails***, which is a shape the harness did
not previously contain and is worth keeping.

### 28.7 Page-limit note — the pointer cost one line and had to be bought

The ICLR main text had **zero** slack (§9 item 2). The Limitations pointer to App. E was drafted
at three lines, then two, and only fitted at ~50 rendered characters — *"App. E frees
Assumption 2 from the candidate count"* — after buying a line back from the E4–E9 paragraph by
four small compressions and by writing the interval-hit rate as `\eqref{eq:interval}` instead of
spelling the interval out. ⚠️ **Consistent with §27.5: shaving alone did not do it; a whole line
had to disappear from some paragraph.** The measured budget was ~27 characters of free space on
p9's last line, which is worth knowing before the next attempt.

### 28.8 Where this helps least, and why it is not evaluated

⚠️ **The ICLR draft's own headline application may be the worst case for it.** In multivariate
evidence selection over `n` retrieved passages, `E_·j` for an answer query is plausibly the
*entire* candidate set — the passages are candidates by construction — so the generalization
would change nothing in E2 or E3. Its natural settings are the typed ones: a pronoun is exclusive
over **mentions**, not over every token; a variable binding over the **references** to that
variable. Those live in `marsea.tex` §7, which is one reason the fuller treatment went there.

**Fidelity measurement would also need `E_·j` annotated alongside `T_j`.** Typed structure
supplies it (lane type, mention-hood); multi-hop evidence does not.

**Status.** Stated and analyzed in both drafts; **evaluated in neither**. Both say so explicitly.

### 28.9 ⭐⭐ An adversarial proof-check was run over the new material, and it found fifteen defects

**Every one was in text written the same day**, which is the point: §25.4's lesson applied to new
material rather than to a port. The reader was given the new sections plus the original
statements of every result they cite, and told to attack. Nine defects were substantive and all
are fixed; the numbers and every `\ref` were clean.

**The real ones, in order of how badly they would have read to a reviewer:**

1. ⚠️⚠️ **`prop:capacity`(i) does not survive `def:subgraph`, and was not in the accounting at
   all.** Under conditional activation, a row that enters Stage 2 runs it over its *whole* row,
   mixed scales included, and the unit cap applies to the mixture. Under the scoped definition
   the program sees only `E_i·`, so the complement adds softmax mass on top and the row total can
   approach **two**. The pointwise row bound — one of the two margin guarantees the title claims
   — was silently lost. Now a row in both transfer tables and a lettered cost, with the two
   available repairs named and **neither chosen**: cap the whole row while scoping only the
   exclusivity, or keep the definition and state the guarantee on in-relation mass.
2. ⚠️ **`prop:monotone` was on the "verbatim" list and is the one result that names the set
   size** — `k(τ) → n_q` as `τ → 0⁺`. The proof's blanket sentence *"none of them mentions the
   cardinality of that set"* was false by exactly one item. Now the stated exception.
3. ⚠️ **`prop:selmargin`(ii) asserted a Gaussian lower bound it did not prove, and then
   over-quantified.** The sub-Gaussian *upper* bound is the direction that makes `δ_j` large; the
   degradation claim needs the lower bound. Worse, the hypothesis was "σ-sub-Gaussian", and a
   **bounded** scoring function is σ-sub-Gaussian with `E max → ess sup`, i.e. `Θ(1)`. So
   *"no scoring function within the class avoids it"* was false as stated. Now: (i) carries the
   failure claim under unbounded support above; (ii) gives the rate for the Gaussian case and
   cites the classical asymptotic rather than pretending to prove it; and the escape — a
   non-target law bounded above — is named as the negation of (i)'s hypothesis.
4. ⚠️ **`prop:selrestrict`(iv) was stated as a biconditional and is not one.** Revoking a
   *different* index in the same counterexample leaves permanence intact, so "only if" is false
   per subgraph. Now "whenever", with "the hypothesis cannot be dropped" carried by the instance.
   ⚠️ The two drafts had disagreed on this: ICLR said "if", `marsea.tex` said "iff".
5. ⚠️ **`prop:leakage` is a statement about the *comparator*, not about MarSea**, so restricting
   the program's domain does nothing to an unmasked softmax. The transfer clause said `n_q →
   |E_·j|` flatly, which contradicts `rem:selhonest` two subsections later — the remark whose
   whole point is that the domain reduction belongs to the mask and a masked baseline gets it
   too. Now "at matched domain".
6. **`prop:selconserv`'s escapability construction pointed at a construction in a different
   proof**, inherited its `n_q ≥ n_k ≥ 2`, and so did not cover the "every `n_q,n_k`" it claimed
   — at `n_k = 1` the claim is simply false. Now `n_k ≥ 2` is a hypothesis and the subgraph
   `E = {(i,j) : j ≥ 2}` is constructed in place. **"The strongest bound of its form"** was also
   an overclaim: what is proved is that no bound growing with `n_q/n_k` replaces it.
7. ⚠️ **"Invisible to the diagnostics" was an overreach.** The narrow claim — invisible to
   `prop:degeneracy`'s three variances — is right. But at `|E_·j| = 1` every column has
   `k* = 1` and `ν_j = 1`, and `T_j ⊆ E_·j` forces `m_j ≤ 1`, all three of which the papers
   already commit to reporting. ⭐ **Saying which instruments *do* see it is stronger than
   claiming none do**, and both drafts now do that.
8. **`rem:selgate`: five heads minus two gates is three, not four.** The drafts had also
   disagreed, ICLR subsuming only `ω^K` and TMLR claiming both but subtracting one.
9. **The `n_q = 32` row of the margin table does not show what the paragraph claims.** There the
   column margin is still positive, so `thm:recovery` guarantees recovery at that column's own
   temperature; the 63.6% is a temperature set from the scope and applied to the column. Both
   captions now say so and separate it from the rows where the assumption genuinely fails.

**Bookkeeping also caught:** the ICLR reproducibility statement said "under 35 seconds" while its
own appendix said 25 (measured 22.4); the papers said `91%` where the harness prints `90.7%`;
`tab:transferchecks`'s caption described its last two rows as confirming a negative when the
first of them is a positive; and ⚠️ **the ICLR body still asserted flatly, in main text and again
in the proposition, the very phrase — *"with no `n_q` in it at all"* — that the new appendix
quotes back and retracts.** That last one is the §23.2 class again: a concession added in one
place and not propagated to the claim it concedes. `marsea.tex` had it right because
`rem:leakhonest` sits directly beside `prop:leakage`; the ICLR draft now points from both sites.

⭐ **Method note worth keeping.** What made this pass work was giving the reader the *original
statements of every result the new text cites*, not just the new text. Six of the nine
substantive defects are mismatches between what the new text says a result says and what it
says — invisible to anyone reading only the new pages, and invisible to a grep.

### 28.10 Notation tables widened to the text block, in both drafts

Luke, 2026-08-30. Both notation tables were `tabular` with two fixed `p{0.285\textwidth}` meaning
columns, so each came to roughly `0.57\textwidth` plus the two natural-width symbol columns —
visibly narrower than the text block, and wrapping more than it needed. Both are now
`tabularx{\textwidth}` with the meaning columns as a ragged-right `X`:

```
\usepackage{tabularx}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\begin{tabularx}{\textwidth}{@{}l@{\hspace{...}}Y@{\hspace{...}}l@{\hspace{...}}Y@{}}
```

⚠️ **Ragged-right is not cosmetic here.** Justified `X` columns at this width stretch interword
space badly on rows carrying long inline math — `W_j`'s *within-set spread* row in `marsea.tex` was
the visible case. Plain `X` was the first attempt and it is the wrong one for this table.

State after: `marsea_iclr.tex` 29 pp with the main text still ending p9 and Fig. 1 on p8;
`marsea.tex` 77 pp; both 0 errors / 0 undefined / 0 overfull / 0 multiply-defined. Each table is
shorter as well as wider, so in the ICLR draft App. C now starts on the same page.

⭐ **A symbol collision this surfaced, and it is the §26.1 class again.** Reading `marsea.tex`'s
notation table rendered showed that **`E` is already taken**: `prop:edges` defines
`E := {(i,j) : A_ij > 0}`, the induced edge set, and it has a notation-table row. §28's
`\mathcal{E}` is *also* an edge set. Two edge sets in one document distinguished only by script
versus roman is exactly the trap §26.1 records — and the earlier check missed it because it
grepped for `\mathcal{E}` and `E_j`, neither of which matches a bare `$E$`.

**Not renamed, but distinguished**, since the relationship is worth stating and is now in
`def:subgraph` and the notation table: `\mathcal{E}` says which pairs the programs *range over*
and is fixed before they run; `E` says which pairs end up with positive weight and is read off
afterwards. **Neither contains the other** — a pair in `\mathcal{E}` may be zeroed by either
program, and a pair outside it may carry ordinary attention weight.

**Both notation tables gained a selective-exclusivity group** (`\mathcal{E}`, its slices, and in
`marsea.tex` also `e_ij` and `μ`), marked *stated, not evaluated*. They had been claiming to
collect every symbol in the paper while omitting the ones §28 introduced.

---

## 29. Audit §6.3, §6.4 and §6.5 answered, and Figure 1 added (2026-08-30, third pass)

**Occasion.** Luke returned to the audit's Tier C strategic findings with positions on three of
them, and asked for all three to be applied plus the precision–recall curve as a main-text
figure. His positions, and what each turned into:

### 29.1 §6.3 — Thm. 1's hypotheses in self-attention: Luke was right, and there is a stronger argument

**His argument:** the reviewer's objection (fixing `{k_j}` also fixes `x_i` in self-attention)
misses that when `k_j` is fixed the admissible set for that index collapses to a point, and the
collapse does not touch the `∏_i 𝓡_i` identity at all — Thm. 1's content is that query-local
attention attains a product, not a multivariate set.

⭐ **Correct, and the division of labour is the thing the objection collapses.** Thm. 1 supplies
the **shape** (a product, on whatever product domain it is given); Assn. 1 with Cor. 1 supplies
the **bite** (that the product is too big for `𝒞`). Only the second is a claim about the domain.

⭐⭐ **The stronger argument is linear-algebraic and is now in both drafts.** With
`q_i = W_q u_i`, `k_i = W_k u_i` and `W_q, W_k : ℝ^d → ℝ^{d_h}`, fixing every key confines `u_i`
to `u_i + ker W_k`, of dimension `d − d_h`; `W_q` restricted to that kernel has rank
`min(d_h, d−d_h)` unless `row(W_q) ⊆ row(W_k)`. So the queries still move in a `d_h`-dimensional
set once `d ≥ 2d_h`. **Fixing the keys does not collapse the admissible set.**

⚠️ **Four corrections the adversarial pass forced on that argument** (§29.6), all now applied:
- **Under pre-LN the count is different.** `LN(u)` lies on `{1ᵀu = 0, ‖u‖ = √d}`, a **(d−2)**-manifold, not a sphere of dimension `d−1`; intersecting with `ker W_k` leaves `d − d_h − 2`, so an open set of `q*` needs **`d ≥ 2d_h + 2`** — three or more heads, not two. The first draft said `d ≥ 2d_h`, "which multi-head attention always satisfies", and that is false at exactly `H = 2`.
- **`row(W_q) ⊆ row(W_k)` kills it outright** (`W_q` vanishes on `ker W_k`). Lebesgue-null but trained weights are not a generic draw, and the paper insists elsewhere that `(a,M)` be *measured* rather than assumed. Now stated as a measurable, unmeasured degeneracy.
- **Rank is not conditioning.** A small `σ_min(W_q|ker W_k)` makes the required displacement large — which is precisely what the pre-LN norm bound forbids. Reachability can fail while the rank count succeeds.
- ⚠️⚠️ **"Shrinking a domain cannot touch the shape" is false as written.** Thm. 1's `⊇` direction is proved *from* the product hypothesis, so restricting `𝒟` to a **non-product** subset breaks it. The claim is true only for shrinkages that preserve product structure — which is Luke's point stated correctly, and `marsea.tex` said the opposite four lines later.

**Framing now used in both drafts:** the remark establishes a **negative** claim (the key
constraint alone does not make Thm. 1 vacuous), not the positive one that every query is free.

### 29.2 §6.4 — discharged

Answered by §28's selective-exclusivity section. Nothing further.

### 29.3 ⭐⭐ §6.5 — the precision–recall trace is predicted in closed form, and is now Figure 1

**Luke's position:** MarSea is precision-focused, but the theory also estimates recall strongly,
and a balanced P–R curve is likely its advantage; and he doubted the reasoning benchmarks score
only recall.

**He is right on the theory, and the drafts had been underselling it.** Recall is not bounded
here, it is *characterized*: Thm. 2's upper endpoint is an iff for `R_j = 1`; graded recall gives
`R_j = k*(τ_j)/m_j` exactly; retention gives `R_j = K/m_j` exactly; stage composition makes it
monotone with an enumerated path set. Precision has one failure path, recall three — each with a
closed form.

⭐ **The consequence, which neither draft had drawn: the whole curve is predicted.** Sweeping
`τ_j` on a column meeting Assn. 2,

```
(R_j, P_j) = (1, m_j/k*)      below the interval
             (1, 1)           on it
             (k*/m_j, 1)      above          (1/W_j := ∞ if W_j = 0)
```

an **L with its corner at (1,1) and no interior point**, while a full-support mechanism does not
move from `(1, m_j/n_q)` at any temperature. Verified 0/24000, 0/16000, 0/20472, and the
L-shape on 4000 columns; the comparator's single point on 2000. `Lem. 1` is what supplies
`|S_j| = k*` and turns two support-inclusion statements into precision and recall values — the
first draft cited only Thm. 2 and graded recall, which do not give it.

**Figure.** `pr_curve.py` runs Stage 1 at 400 temperatures on one planted column
(`n_q=64, m_j=4, δ_j=0.70, W_j=1.20`) and a column-softmax on the same column at the same
temperatures, and plots what comes out. Nothing is drawn by hand and the script asserts all three
branches exactly. It is a `wrapfigure` in §3.5 of the ICLR draft and a `figure[t]` in
`rem:notradeoff` of `marsea.tex`.

⚠️ **A visual honesty point worth keeping.** The blue curve *terminates on* the grey square:
softmax's point is the `τ_j → 0` limit of MarSea's own trace. That is correct and it does not
weaken the contrast — the dense mechanism is stuck at the one point the sparse one passes through
— but a caption that says "a curve against a point" without saying so invites the reading that
they are disjoint objects.

⚠️⚠️ **The superlative had to be given back.** Compressing the main text moved *"the sharpest
single test of the theory"* off the interval-hit rate and onto the trace. That is backwards:
**given Assn. 2 and α = 2 the trace is an identity**, which is exactly why the harness confirms it
with no sampling risk. A departure indicts the assumption on that column or the implementation,
never the theorem. The hit rate — does the learned `τ_j` land in the interval? — is the only
genuinely empirical quantity of the two. Both drafts now say so, and the trace is described as
the *wider* test rather than the stronger one.

### 29.4 §6.5's benchmark question — his suspicion was right, and half the benchmarks were being under-used

Checked against primary sources.

| benchmark | precision-sensitive criterion | was the draft using it? |
|---|---|---|
| **HotpotQA** | official supporting-fact EM/F1, and **joint** metrics `P_joint = P_ans·P_sup`, joint EM requiring both exact | ⚠️ no — only answer F1 |
| **MuSiQue-Answerable** | `support_f1` beside `answer_f1` | ⚠️ no — only answer F1 |
| **RULER** | **none.** "recall-based accuracy", checking presence of the target values | ⚠️ and RULER is E2, the load-bearing experiment |

⭐ **Supporting-fact EM is exact set recovery** — the task-level image of `P_j = R_j = 1`, on a
leaderboard metric rather than a bespoke one. That is where the precision claim cashes out, and
the Metrics paragraph named neither it nor the joint metrics.

**Applied to both drafts:** report each benchmark's own precision-sensitive criterion as the
task-level number; **pre-commit E2/V16 on RULER to exact-set accuracy** — the returned set equals
the gold value set, no distractor admitted — reporting RULER's own score alongside, which costs
nothing because needle and distractor identity are generator configuration.

⚠️ **Two corrections the adversarial pass forced here** (§29.6):
- **MuSiQue-Full was misdescribed.** Its `An+Sf` / `Sp+Sf` scores are a **paired sufficiency
  classification** — an instance with a sufficient context is paired with one with an
  insufficient context, and a wrong call on either scores zero on the pair. That is abstention,
  not evidence precision, and `An+Sf` inherits the answer score's blindness to over-selection.
  It is now reported as an abstention diagnostic and removed from the precision list.
- ⚠️ **A silent granularity slide.** HotpotQA annotates supporting facts at the **sentence**
  level; `m_j` in this paper is a count of gold **paragraphs**. Both drafts now say which
  granularity each number is read at.
- **IHEval's depth stratification** is scoped to its multi-turn rule-following task, which is
  where the number of conflicting components varies; elsewhere its conflicts are between two
  levels.

### 29.5 ⚠️ The 9-page limit, a fourth time — and the diagnostic worked again

Adding a main-text figure to a draft with **zero** slack cost six lines, and roughly six rounds of
sentence-shaving moved the boundary by nothing, exactly as §26.3 and §27.5 record: every trim was
absorbed by reflow. ⭐ **The decisive diagnostic was §27.5's again** — build once with the new
figure deleted. The main text then ended on p9 with p9 holding only 42 text lines, which proved
the entire overflow was the float and that no further shaving would help.

**What actually paid for it**, in order of size: re-anchoring the E1 probe figure so it stopped
straddling a page break (~3 lines); moving the baseline *list* and the dataset *justification* to
App. G, keeping the decisive baseline and one-clause dataset roles in the main text (~5); a
`\paragraph{Reporting rules.}` in App. G that the main text had been promising and the appendix
did not contain (~2, and a dangling promise fixed); shrinking the figure to `0.29\textwidth` and
tightening both wrapfigures' `\vspace` (~2); and ⭐ **deleting the prose the figure made
redundant** — the old §3.5 sentence spelling out "lower endpoint ⟺ `P_j=1`, upper ⟺ `R_j=1`" is
what the figure now shows (~2).

⭐ **That last is the pattern to reuse: a figure earns its space by replacing prose, not by being
added to it.**

### 29.6 The second adversarial pass, and what it caught

Same method as §28.9 — the new material plus the original statements of every result it cites,
with instructions to attack. **31 findings, of which about a dozen were substantive**, all in
same-day text, all fixed. The four pre-LN/rank corrections (§29.1), the superlative inversion
(§29.3), the MuSiQue-Full misdescription and granularity slide (§29.4), plus:

- ⚠️ **The ICLR main text asserted "no interior point" with Assumption 2 deleted.** Without a
  positive margin a non-target can outrank a target, the prefix support contains it and misses a
  target, and `P_j < 1` and `R_j < 1` hold at once — an interior point. The qualifier was present
  in the appendix and in `marsea.tex` and absent from the ICLR main text. ⭐ **The harness could not
  have caught it**: `targets_scores` plants a strictly positive margin by construction, so the
  0/4000 row is conditional on precisely the hypothesis the main text dropped. **This is §22.2's
  lesson for the third time** — a check that assumes the hypothesis under test cannot falsify it.
- ⚠️ **The headline check count double-counted.** The "no interior point" row re-aggregates the
  same 60,472 τ-evaluations as the three trace rows into one boolean per column; counting its
  4,000 columns as further trials contradicted the protocol paragraph's own claim that every
  count is a count of trials actually run. The row is now **printed rather than reported**, and
  the total is **456,621**, which reconciles exactly: ICLR `tab:verification` 211,978 +
  `tab:transferchecks` 244,643, and TMLR's single table likewise.
- **`pr_curve.py`'s docstring was false** — it said nothing was drawn by hand while hard-coding
  the comparator's point at `m/n_q`. It now runs an actual column-softmax at the same
  temperatures, and its assertions check all three branches of the trace rather than only
  `P=1 or R=1`.
- **Compression damage, checked and clean.** Exactly one pure deletion in the main-text diff, and
  its pointer was relocated rather than lost; every appendix promise resolves. ⚠️ Two residues:
  *"All three are in App. H, staged cheapest-first at ≈336 H100-hours"* attached the whole
  programme's compute to E7–E9 (fixed), and the rebuttal to *"your ceiling is an artifact of full
  support"* now lives only in App. G — defensible, and the one deletion a hostile reviewer will
  notice.
- **TMLR `rem:jointhyp` forward-referenced Assumption 1** and has been moved to sit after it.
- **`rem:notradeoff` cited V4 twice where the material is V4b.** Fixed. ⚠️ **Open**: V4 has
  lettered items (a)–(d) cited as "V4(b)", "V4(d)", which collides with the distinct paragraph
  **V4b**. Pre-existing, eight sites, not renamed.

### 29.7 State

| | `marsea_iclr.tex` | `marsea.tex` |
|---|---|---|
| length | **30 pp, main text ends p9**; Fig. 1 (P–R) p7, Fig. 2 (probe) p8, refs p10–11 | **80 pp** |
| build | 0 err / 0 undef / 0 overfull / 0 multiply-def | same |
| harness | **456,621 checks, 0 violations**, 20 s | same file |
| new artifact | `pr_curve.py` → `pr_curve.pdf`, shipped with both | same |

⚠️ **`pr_curve.pdf` must ship beside both `.tex` files**, exactly as `v1_probe.pdf` and
`verify_all.py` do. It is reproducible from `pr_curve.py` with numpy and matplotlib, which is
more than can be said for `v1_probe.pdf` (§0.6 item 1).

---

## §30. Audit §6.6 / §6.7 / §6.2, the page-limit policy change, and the third adversarial pass

*Session of 2026-08-31. Both drafts, `v1_probe.py`, both `.bbl` files.*

### 30.0 Standing decision, new

⭐ **The 9-page ICLR limit is deferred until content is complete.** Luke: *"From now on, let us
not worry about ICLR page limit until all contents are ready in the paper draft, including the
experiment results. At this phase, the more important thing is to explain and present MarSea in a
clear and accurate way, and we will make wording concise in the final draft before submission."*
Every prior session fought the limit and lost material to it; that trade is now suspended. The
ICLR draft stands at **34 pp** and is not to be compressed until the experiments land. Adds to
the standing list: ICLR 2027 venue; `joca_ws` dropped; the causal/prefix form is native design and
is never to be raised as an obstruction; precision is the centre and recall is protected, not
traded; motivating examples are not rewritten unilaterally.

### 30.1 §6.2 — encoder and decoder coverage

Both programs are stated for arbitrary `n_q`, `n_k`, so MarSea covers encoder self-attention,
cross-attention and the causal decoder alike. What varies with the shape is not the mechanism but
the **strength of Prop. `conservation`**, and both drafts now say so explicitly (`rem:shapes` in
`marsea.tex`, the parallel paragraph in the ICLR appendix):

- **`n_k ≪ n_q`** (cross-attention, encoder–decoder, the DETR-style decoder): the bound `n_q/n_k`
  grows with the query count and the obstruction is severe. This is where the proposition works.
- **Self-attention `n_q = n_k`**: the bound is the constant 1, and the drafts now concede it is
  weak — ⚠️ **worse than weak under a causal mask**, where it is attained at the first position
  with `A_11 = 1` before anything is trained, so the witness is an artifact of the mask.
  Thm. `separation` is shape-independent and carries the argument in a decoder alone.
- ⚠️ **Conditional activation does not rescue it, and the drafts no longer imply it does.** The
  earlier text said the bound "sharpens under conditional activation" — but the same pigeonhole
  runs against *MarSea's own budgets*: `Σ_j c_j = n_q` over the active set gives
  `max_j c_j ≥ n_q/|𝒦|`, which **grows** as the gate closes. That is a concession about MarSea,
  not an obstruction for baselines. Now written as one.
- The subset generalization `max_{j∈𝒦} Σ_i A_ij ≥ μ/|𝒦|` was **deleted from Prop. `conservation`
  in both drafts**: it generalizes over *keys*, where the argument that follows needs *rows*.

### 30.2 §6.6 — related work and citations

⭐ **The most serious correction of the session: Slot Attention's two normalizations were
backwards in both drafts.** In MarSea's orientation slot = query, input = key. The **first**
normalization (softmax over slots) is `Σ_i attn_ij = 1` — **the column marginal**, uniform at
one, and the closest published relative of Stage 1. The **second** (divide by the sum over
inputs) is the ordinary row normalization. Zhang & Lipton identify the pair as a **single**
Sinkhorn iteration, and one iteration does not satisfy both marginals: **the second step destroys
the first**, so the operative matrix is row-stochastic with unconstrained columns and
Prop. `conservation` applies to it. `marsea.tex`'s paragraph heading was literally *"Competition on
the receiving axis"* — now *"…on the emitting axis"*. Both drafts rewritten.

Other corrections, all applied to both drafts:

- **Expert choice**: "nothing is dropped" overstated it. Correct: no *expert overflow* is
  discarded (unlike token choice), but a token chosen by no expert skips the layer. Also, expert
  choice does not **escape** Thm. `separation` — it is not an attention mechanism, so it was never
  in the theorem's universe. What it shares with Selective Attention is the *shape* of the move.
  Consequence: **Selective Attention is the one attention mechanism we know of that escapes the
  theorem**, not "two mechanisms".
- **Sanford et al.**: their triple-detection result does not "reverse the direction" between
  architectures — it is a *negative* result about attention itself, proved by communication
  complexity. What MarSea takes from that line is the standard for what a separation must exhibit,
  not a technique.
- **Differential Transformer**: Thm. `separation` covers it, but ⚠️ `Cor. capacity` **does not**
  — its weights are signed, and the realization argument lower-bounds a column sum by `Ma` over
  the *non-negative* orthant, which has no force where a column sum can be small by cancellation.
  Rows sum to `1−λ`, so Prop. `conservation` applies only with that constant. Now scoped
  explicitly in both drafts. (Flagged, not acted on: whether to add it as a baseline.)
- **Seven citations added** to `marsea_iclr.bbl` (32 → 39, all cited, none uncited):
  `berthet2020perturbed`, `carion2020end`, `leviathan2024selective`, `sanford2023representational`,
  `velickovic2024softmax`, `xie2020differentiable`, `ye2024differential`. Six to `marsea.bbl`
  (71 → 77).
- ⚠️ **Two bibliography entries were factually wrong in both files** and are now fixed against the
  sources: `shahbazi2025lotformer` was "Mahdi Shahbazi et al., *LOTFormer: Low-rank optimal
  transport attention*" → **Ashkan** Shahbazi, Chayne Thrash, Yikun Bai, Keaton Hamm, Navid
  NaderiAlizadeh, Soheil Kolouri, ***LOTFormer: Doubly-Stochastic Linear Attention via Low-Rank
  Optimal Transport***, arXiv 2509.23436. `yue2024adak` was "*Improving* the efficiency", no
  arXiv id, "preprint" → ***Boosting* the Efficiency of MoE-based LLMs**, Tongtian Yue, Longteng
  Guo, Jie Cheng, Xuange Gao, Jing Liu, **ICLR 2025** (arXiv 2410.10456, since withdrawn from
  arXiv by the authors — cite the proceedings).

### 30.3 §6.7 — E1's decisive baseline, and the null hypothesis

⭐ **The paper's own null lies exactly on the Sinkformer line.** A column program at a *uniform*
budget, normalized so `Σ_j c_j = n_q`, has `c_j = n_q/n_k` — the same slope Sinkformer attains.
So **panel (a) cannot separate learned from uniform marginals**: with `c_j` fixed by hand the two
differ only in *how `c_j` is chosen*, and a single-key sweep never varies that. The null is now
plotted (slope `+0.167`, coincident with Sinkformer) and the limitation stated in both drafts.

⚠️ **Panel (b) as first drafted was worse: it computed only analytic identities.** "Uniform gives
10.5, MarSea gives `m_j`" is `Σ_i p_ij = 1` drawn as a bar chart. **Rebuilt as a measurement**: the
budget head reads `ν_j = ‖p_·j‖₂⁻²`, and whether `ν_j` carries `m_j` is *not* an identity — it
depends on `τ_j` and on the score column. Six keys, true sizes 1–32 at `n_q = 63`, twelve seeds:

| | m=1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| per-key `τ_j` in its interval | 1.00 | 1.55 | 3.07 | 6.68 | 13.95 | **27.61** |
| one global `τ = 1.2` | 1.34 | 2.06 | 3.58 | 5.82 | 8.11 | **11.48** |

A factor of 32 compressed to 8.6. This is **Cor. `necessity` in training-free form**, and it is
why a uniform-exclusivity null cannot even supply the budget head with a usable input.

⚠️ **The two shipped construction parameters are calibrations and are now disclosed as such.**
`SIGMA = 0.30` and `GBAR = 0.62` were chosen *so that the regenerated slopes land on the
previously reported +0.163 and +0.101*. Both drafts now say that in those words. The three
quantities claims rest on are parameter-free: Sinkformer `n_q/n_k`, the uniform budget `n_q/n_k`,
MarSea `0`. The gate is now drawn from `Beta(0.62·12, 0.38·12)` on its own RNG stream so the mean
is exact and the score stream is unperturbed.

### 30.4 ⚠️ The "real" `v1_probe.pdf` does not exist

Luke: *"the real Figure 2 v1_probe.pdf needed in marsea_iclr.tex can be found in marsea.tex."* It
cannot. The TMLR copy is **byte-identical** to the build copy (md5 `6515d0a3151ba49df2e8d81dc0781cd5`),
and the project attachment extracts as text reading **"RECONSTRUCTED / PLACEHOLDER"** and naming
**"JOCA"** — i.e. it predates the rename. There is no earlier run to recover. `v1_probe.py` is
therefore the only source of the figure, and it now regenerates **all four recorded slopes**
(+0.163, +0.101, +0.167, −0.000) and **all four recorded Sinkformer masses** (0.833, 2.333,
4.000, 7.333 = 5/6, 14/6, 24/6, 44/6) exactly, from a construction that is written down.

### 30.5 State

| | `marsea_iclr.tex` | `marsea.tex` |
|---|---|---|
| length | **34 pp** (limit deferred, §30.0) | **83 pp** |
| build | 0 err / 0 undef cites / 0 undef refs / 0 overfull | same |
| harness | **456,621 checks, 0 violations** | same file |
| figures | `pr_curve.pdf`, `v1_probe.pdf` (both regenerable) | same |
| bibliography | 39 bibitems, all cited | 77 bibitems, all cited |

**Open items carried forward**: whether to add Differential Transformer as a baseline (flagged,
not done unilaterally); the `V4b` vs `V4(b)`/`V4(d)` name collision in `marsea.tex` (8 sites,
pre-existing).

---

## §31. Fourth adversarial audit — 19 defects, 15 applied

*Session of 2026-08-31, continued. Both drafts, both `.bbl` files, `v1_probe.py`.*

Method: full read of `marsea_iclr.tex`; two independent verification passes (literature against
primary sources; every number against the shipped scripts); each finding re-checked by hand.
Full account: project doc `claude/marsea_iclr_audit4_2026-08-31.md`. Open decisions:
`claude/marsea_iclr_open_decisions_2026-08-31.md`.

### 31.1 ⭐ The finding that matters: MESH already learns a non-uniform marginal

Both drafts said *"MESH improves the construction by changing the transport cost; the marginals
are what we change, and they are what both leave fixed."* **False.** MESH's Appendix A:
*"We learn $a = m\cdot\mathrm{softmax}(h(Z))$ with a neural network $h$ shared across the $n$
input elements… allows the model to put focus on important input elements"*, with
$\sum_i a_i = m$; a variant learns both marginals. Read in MarSea's orientation that is **a
learned, non-uniform, content-predicted key-side marginal normalized to a layer total** — the
same object as `c_j` with `Σ_j c_j = n_q`, and the closest prior art to Stage 1.

⭐ **This is a positioning problem, not a citation error.** The theory is untouched — a learned
marginal is query-*coupled*, so MESH sits on MarSea's side of Thm. 1, not against it; Prop. 4
applies to MESH with equality; MESH has no recovery result, no temperature, no interval, no exact
zeros. What it costs is one clause of contribution (i). Two differences survive and are now
stated in both drafts:
1. **The field argument.** MESH's `h` reads the input element **alone**; `c_φ` reads the key
   *together with* `ν_j`. This is the paper's own criterion for every head, Cor. `necessity`
   already proves it necessary, and Fig. 2(b) already measures it — so MESH is a *published
   mechanism that fails the paper's own criterion*.
2. **Equality inside a transport vs. a cap outside one.** MESH's marginals are equality
   constraints of a doubly stochastic problem; MarSea's row side is an inequality cap.

### 31.2 Applied — factual errors about others' work

- **Four bibliography entries had largely invented author lists** (⚠️ the worst kind of defect
  here): `hou2024relation` (Xiuquan/Meiqin/Senlin/Ping/Badong, not Xiaocheng/Mingsheng/Sheng/
  Pengfei/Bo), `du2025context` (four listed coauthors not on the paper), `zhang2025iheval` (three
  listed authors not on the paper), `herasimchyk2026residual` (Hanna/Robin/Tomislav). All fixed
  against ACL Anthology and arXiv. Every other entry re-checked and correct.
- **`softmax₁` was misattributed** to `xiao2024efficient`. It is **Evan Miller (2023)**;
  StreamingLLM cites it as prior work and prefers a learnable sink. `miller2023attention` added
  to both `.bbl` files (39→40, 77→78).
- **`tay2020sparse` does not constrain the attention matrix**: Sinkhorn is applied to a block
  *sorting* matrix and attention over the reordered blocks is ordinary row-softmax. Removed from
  the transport group in both drafts and explained.
- **Slot Attention's slots are not "a fixed bank of learned slots"** — they are sampled each
  forward pass from one shared learned Gaussian, deliberately, so no slot specializes. "What
  competes is parameters and not data" was false and is rewritten.
- **Three preprints are published**: `velickovic2024softmax` → ICML 2025;
  `ye2024differential` → ICLR 2025 (Oral); `leviathan2024selective` → ICLR 2025. Venues and
  display years updated. The `zhang2023mesh` `\natexlab{b}` artifact (no matching 2023a) removed.
- **Position bias over-attributed**: only Chowdhury claims the U-shape arises independently of
  positional encodings. Wu et al. have the causal mask *competing with* RoPE/decay masks;
  Herasimchyk et al. treat encodings as one of four forces. Both drafts now say the three differ
  and decline to adjudicate.

### 31.3 Applied — claims the paper's own artifacts contradicted

- ⚠️ **"No block rejects" was false.** `verify_all.py` has three `if W > 0:` guards, and the
  skipped trials are visible in the paper's own denominators: **10,098 / 10,604 / 20,472** against
  12,000 / 12,000 / 24,000. Reframed correctly in both drafts: those rows test the regime *above*
  `1/W_j`, which is `+∞` at `W_j = 0`, so an `m_j = 1` column has no such regime — the test is
  **undefined** there, not the draw rejected. (Two literal `while … continue` loops remain in the
  code; they reject nothing on these seeds.)
- ⚠️ **`tab:margin`'s caption contradicted its own numbers**: *"from n_q=128 … no temperature
  recovers"* beside 14.9% and 1.2% in the same rows. Both `δ_j` columns are **means**; recovery is
  per column. At `n_q=32` (mean margin positive) **26.5%** of columns already have `δ_j ≤ 0`; at
  `n_q=128` (mean negative) **24.5%** still satisfy Assumption 2, which is what the 14.9% is made
  of. Rewritten in both drafts.
- **A number the shipped script contradicts**: the paper said softmax's slope is `0.133` at spread
  1.0; re-running `v1_probe.py` gives **0.135** (0.165 and 0.163 reproduce). Fixed in both drafts
  and in the script's own comment.
- **The ICLR E1 slope list had four entries where the figure has five curves** — the uniform-budget
  null was plotted and discussed but not listed. Fixed.
- **"Sinkformer sits at n_q/n_k exactly" overstated twice**: Sinkformers run 3–5 iterations and at
  one iteration a Sinkformer *is* row-softmax, so a deployed one lies between the lines; and Sander
  et al. state only the **square** case, so the rectangular reading is our extension. Both scoping
  remarks added; the figure now says "a converged Sinkhorn scaling".

### 31.4 Applied — internal, mechanism-level

- ⭐ **`R̃_i = 1` on average was false in the regime MarSea operates in.** `Σ_j c_j = n_q` is over
  the **active** keys, but ungated keys keep ordinary attention and their mass lands in the same
  rows, so `E[R̃_i] = 1 + E[ungated mass] ∈ (1,2)` whenever any key is ungated — always, since
  MarSea is part-time by design. **The cap binds more often than "τ_i above one" suggests**, by a
  factor set by the gate rate. The whole two-regime paragraph rested on this reading. Rewritten in
  `marsea_iclr.tex`; `marsea.tex` already had a partial version.
- **`S_j^{(1)}`, `S_j^{(2)}` were used but never defined** (Prop. `stagecomp`, six table rows, E7).
  Def. `fidelity` now defines them.
- **Prop. `stagecomp`'s "the same four statements hold verbatim on the row margin" fails for
  (iii)**: its reason ("θ_i is indexed by the row while P_j is a column quantity") is
  margin-specific. Conclusion transfers, justification does not. Now stated as (i),(ii),(iv)
  verbatim plus (iii)'s conclusion by a different argument.
- **App. E leans on unbounded score support** (`δ_j → −∞`, rate `√ln n_q`) against the paper's own
  pre-LN model in App. D, under which scores are bounded. ⚠️ **Not yet applied** — see 31.6. The
  argument survives in a weaker sufficient form: `δ_j` is non-increasing in `n_q` and converges to
  `min_T s − ess sup(non-target)`, negative whenever any non-target can outrank the weakest target.

### 31.5 ⚠️ Not applied — the 456,621 is not a count of independent tests

**146,667 checks (32.1%) cannot report a violation by construction**, block by block:
Prop. 20's six rows (92,643) test `S² ⊆ S¹`, which `A = [Ã−θ]₊` *makes* true, counted three times
on the same columns; the "recovery at every `n_q`" row re-seeds `default_rng(7)` **inside** the
`n_q` loop, so 3,000 columns are reported as 18,000 and all six rows are byte-identical (the tell
is `delta scope = 1.127` on every row, which is a printed table column); Prop. 19's targets *are*
the top-`k` because the array was sorted; Prop. 18's threshold is drawn strictly between the two
groups; Prop. 14's rows assert `m/n = m/n` on a full support; Prop. 4's restricted row is
`max ≥ mean`. A further **60,472** (Fig. 2's trace rows) are arithmetic restatements of the Thm. 2
sharpness rows — **207,139 (45.4%)** in total. Units are also incommensurable: one block
contributes **131,180** by counting prefix timesteps.

⚠️ And the paragraph *"What these checks do and do not establish"* claims the harness catches
*"hypotheses that are vacuous."* It did not catch these. **Recommendation is not to shrink the
number but to stop leading with it.** Two of these are outright bugs (the re-seeded RNG; the
Thm. 2 interval row reporting columns where its scope twin reports evaluations); fixing the first
puts the honest total at **441,621**.

### 31.6 Open decisions, awaiting Luke

1. **How hard to reposition against MESH** — options A (local repair, applied), **B (sharpen
   contribution (i) to the *field argument*; recommended)**, C (reposition around exact zeros).
   B turns a novelty problem into a worked example of the paper's own thesis and suggests a
   sharper baseline than the uniform-`c_j` null: *a MESH-style marginal learned from the key
   alone*.
2. **How to present the verification count** — recommended: stop leading with the total, mark
   deterministic-identity rows, state units, fix the two bugs now; rebuild the vacuous blocks to
   be falsifiable **after** the experiments land, on the TMLR draft.
3. **App. E's unboundedness argument** (31.4, last item) — the fix strengthens the claim but
   changes a stated rate to a stated model, so it is held with the other two.

### 31.7 State

| | `marsea_iclr.tex` | `marsea.tex` |
|---|---|---|
| length | **35 pp** | **84 pp** |
| build | 0 err / 0 undef refs / 0 undef cites / 0 multiply-def / 0 overfull | same |
| harness | 456,621 checks, 0 violations (see 31.5) | same file |
| bibliography | **40** bibitems, all cited, none uncited | **78**, same |

---

## §32. Both decisions taken: option B on positioning, option B on the verification count

*Session of 2026-08-31, continued. Luke chose **B** for both open decisions of §31.6, and the
App. E item was applied with them.*

### 32.0 ⭐ Standing decision, new

**Contribution (i)/3 no longer claims the learned column marginal as the novel object. The novel
object is the marginal predicted from the participant *together with its field*.** This is now
load-bearing in both drafts and should not be relitigated: `zhang2023mesh` is cited in the
contributions themselves as prior art for the learned marginal, and `Cor. necessity` is what
carries the claim that reading the participant alone is *insufficient* rather than merely
different.

### 32.1 Decision 1 (B) — the contribution sharpened to the field argument

Applied to both drafts:

- **Contribution (i) rewritten** (ICLR) / **contribution 3 extended** (TMLR) to say plainly that a
  learned non-uniform column marginal is *not new*, name MESH as the closest prior art, and locate
  novelty in what the marginal is predicted from. `Cor. necessity` is cited as the reason it
  matters: two keys with disjoint recovery intervals admit no common parameter, so a head reading
  `k_j` alone provably cannot serve both.
- ⭐ **A sixth baseline: a budget head reading `k_j` alone**, `c_j = c_φ(k_j)` with `ν_j` and
  `{γ_ij}` withheld. Stated in both drafts as *not a strawman* — it is the form MESH uses, so the
  ablation is against a published mechanism. Both drafts now **pre-commit**: *"we predict this is
  the strongest baseline and that it still loses on the m-sweep; if it does not, contribution (i)
  is what fails."*
- ⚠️ **No sixth curve was added to Fig. 2(a), and the draft now says why.** A head reading `k_j`
  with the column withheld is **constant in `m`**, so in panel (a) it would trace a flat line
  *indistinguishable from MarSea's* — panel (a) cannot separate them any more than it separates
  learned from uniform. **Panel (b) is where they part**, because `ν_j` is precisely the input such
  a head is denied. Added to the E1 discussion.
- E9's field-argument ablation is now described as having a published comparator rather than a
  constructed one.

**What this does not cost.** The theory is untouched: a learned marginal is query-*coupled*, so
MESH sits on MarSea's side of Thm. 1 rather than against it; Prop. `conservation` applies to MESH
with equality; and MESH has no support-recovery result, no temperature, no interval, no exact
zeros. Only one clause of one contribution changed.

### 32.2 Decision 2 (B) — the count stops leading, and two harness bugs are fixed

⭐ **The advertised total is now 459,621, not 456,621** — and the change is the *sum of two bug
fixes pulling opposite ways*:

| | | |
|---|---|---|
| **RNG bug** | `gE = np.random.default_rng(7)` was re-seeded **inside** the `n_q` loop, so the six passes of the scope-recovery check were byte-identical and 3,000 columns were reported as 18,000. `pool` never enters that check, so the `n_q`-invariance is **structural, not empirical**. Now counted once. | **−15,000** |
| **Units bug** | `Thm. 2, sharp recovery interval` reported `n_col` (6,000 columns) where its scope twin reported 24,000 τ-evaluations for the identical 6,000 × 4 structure. Now counts evaluations. | **+18,000** |

Both drafts' tables reconcile **exactly** against the program's 41 printed rows at 459,621.

Presentation changes, both drafts:

- **The reproducibility statement no longer leads with a headline number.** It reports per claim
  and says why the rows are not summable.
- **`†` marks every row whose conclusion follows from how the generator builds its input**;
  **`‡` marks every row that re-expresses evaluations already counted.** 11 rows daggered in the
  ICLR draft, 15 in the TMLR draft.
- **A new paragraph says exactly what the daggered rows do and do not test**, naming the
  mechanism in each case: `A = [Ã−θ]₊` from non-negative `Ã` *makes* `S² ⊆ S¹`; the row-interval
  rows plant the threshold between the groups; the graded-row-recall row sorts before naming the
  top entries as true; the ceiling rows evaluate `m/n = m/n`; the restricted conservation row is
  `max ≥ mean`. ⭐ Verbatim: *"They confirm that the algebra in the proof is the algebra in the
  statement… they are not independent evidence that the propositions are true, because the
  generator cannot produce a counterexample."* And it names the fix as owed:
  *"Rebuilding them to sample inputs that could violate the conclusion… is the natural next
  version of this harness and we have not done it."*
- **The undaggered rows are named** as the ones that could have failed: Thm. 2's three, hierarchical
  fidelity at `K<m_j`, Cor. 5's full pipeline, Prop. `prefix`'s growing prefix.
- **Denominators are declared to be evaluations**, and the paragraph notes the incommensurability
  (`Prop. prefix` under a nested scope contributes 131,180 by counting prefix timesteps).
- The `n_q/n_k`-escape row now says the escape is **arranged by the mask density the generator
  draws** — it shows the bound is not automatic, not how far it falls.

### 32.3 App. E's asymptotic argument, corrected with the two decisions

The `√ln n_q` divergence assumed unbounded score support, which contradicts the paper's own pre-LN
model in App. D. Replaced by the **monotone-convergence form**, which is what the argument needs
and is model-free: `δ_j` is **non-increasing in `n_q` whatever the score law** and converges to
`min_{i∈T_j} s_ij − ess sup(non-target law)`, negative whenever a non-target can outrank the
weakest target with positive probability. The Gaussian rate is kept but explicitly labelled as
holding *only* under unbounded support, with the bounded case named as the one the paper's own
representation model implies. Weaker as stated, stronger as an argument, and no longer attackable
from App. D.

### 32.4 State

| | `marsea_iclr.tex` | `marsea.tex` |
|---|---|---|
| length | **36 pp** | **85 pp** |
| build | 0 err / 0 undef refs / 0 undef cites / 0 multiply-def / 0 overfull | same |
| harness | **459,621 checks, 0 violations**, 41 rows; both drafts' tables reconcile exactly | same file |
| bibliography | 40 bibitems, all cited | 78, all cited |
| baselines | **six** (was five) | **fourteen** (was thirteen) |

**Carried forward, unchanged**: whether to add Differential Transformer as a baseline (flagged,
not done unilaterally); the `V4b` vs `V4(b)`/`V4(d)` collision in `marsea.tex`; and ⚠️ **rebuilding
the daggered blocks to be falsifiable** — recommended for September, on the TMLR draft, not
against the ICLR deadline.

---

## §33. ⚠️ Correction to §32.1: the MESH baseline belongs in Figure 2(a), and does appear there

*Same session. Luke asked why MESH was not a curve in Fig. 2. He was right and §32.1 was wrong.*

### 33.1 The error

§32.1 recorded, as a deliberate decision, that no sixth curve was added to panel (a) because
*"a head reading `k_j` with the column withheld is constant in `m`, so in panel (a) it would trace
a flat line indistinguishable from MarSea's."* ⭐ **That reasoning omitted the normalization, and
is false.**

MESH normalizes its marginal to the total, exactly as MarSea does: `a = m·softmax(h(Z))` gives
`Σ_i a_i = m`, which read in this paper's orientation is `Σ_j c_j = n_q`. So a budget head that
(a) normalizes `Σ_j c_j = n_q` and (b) reads only the keys has

> **`c_j = n_q · σ_j`, with `σ_j` a function of `{k_j}` alone.**

In E1 the six keys are **fixed** while `m` grows, so `σ_j` is a constant and the realized fan-out
mass is a straight line of **slope `σ_j`**. ⭐ **It cannot be flat.** Flatness needs `σ_j ~ 1/n_q`,
and a head that cannot see the query set has nothing to compute that from.

### 33.2 What this is worth — it is stronger than what it replaces

This is a **result**, not a limitation, and it upgrades panel (a):

- The uniform null is the member `σ_j = 1/n_k`. **So the null and the key-alone head are one
  family**, differing only in slope — which is a cleaner statement than §31.3's "panel (a) cannot
  separate learned from uniform." Panel (a) cannot separate them by the *value* of `c_j`; it
  separates the entire family from MarSea by what `c_j` is **a function of**.
- **Every member grows; MarSea's is the only flat trace.** That is `Cor. necessity` in the
  fan-out-mass coordinate, and unlike the uniform-against-learned comparison **it needs no
  training**. It is the training-free half of the field-argument ablation.
- It gives contribution (i) (§32.0) a figure, not just a theorem and a promised experiment.

### 33.3 What was done

- `v1_probe.py` (both copies, byte-identical): a sixth series, `column program, c_j from k_j
  alone`, at `SIG_KEY = 0.06`; plus a **shaded wedge** over `σ ∈ [0.03, 1/6]` showing the whole
  family, with a legend proxy rather than an in-plot label. Measured slope **+0.060**, against
  softmax +0.163, gate +0.101, Sinkformer +0.167, uniform null +0.167, MarSea −0.000.
- ⚠️ **`σ_j = 0.06` is deliberately the baseline's *favourable* end** — the flattest line such a
  head can draw for this key — and the script and both drafts say so. Picking a large share would
  have been cherry-picking in our favour. (`0.10` was avoided only because it collides with the
  gate's `+0.101` on the page.)
- Two figure repairs found while checking: the band's upper edge was stretching the $y$ axis, now
  capped at the uniform share; and **softmax (+0.163) had been hidden under the thick Sinkformer
  line (+0.167)** — a pre-existing defect, now drawn above it.
- Both drafts: the false sentence removed, the correct account added to the E1 prose and to the
  figure caption, and the slope list extended to six. TMLR's "Four observations" is now five, with
  the new one marked *"and this one \emph{is} a result"*.

### 33.4 Lesson for the record

⚠️ **The claim was wrong in the same way the MESH claim itself was wrong** — by forgetting that
the marginal is normalized to a layer total. §31 caught that error in a sentence about someone
else's paper and then I reproduced it one turn later in a sentence about our own figure. Where a
budget is normalized, *anything* predicted from a set that does not grow scales with `n_q`; that
is the whole content of the uniform null lying on the Sinkformer line, and it applies to every
member of the family, not just the uniform one.

### 33.5 State

| | `marsea_iclr.tex` | `marsea.tex` |
|---|---|---|
| length | **36 pp** | **85 pp** |
| build | 0 err / 0 undef refs / 0 undef cites / 0 overfull | same |
| harness | 459,621 checks, 0 violations; tables reconcile exactly | same file |
| Fig. 2(a) | **six curves + the key-alone family as a band** | same figure |

---

## §34. ⭐ The modelling object stated as a 2×2, and MESH placed in it precisely

*Same session. Luke: "MarSea's multivariate relation is realized on both margins -- not just on
columns… both on columns and on rows, MarSea applies in two aspects: 1) model-predicted capacity
limits the whole or a part of the attention mass…; 2) attention distribution re-shaping
(including exact zeros) over the whole or a part of the attention mass… Let's make MarSea's
modeling object loud and clear against all existing work."*

### 34.0 ⭐ Standing decision, new — supersedes §32.0

**The modelling object is a 2×2, and every future draft states it as one.** The relation is
carried by **both margins**, and on each margin by **two** predicted quantities:

| | **capacity** — how much a node may commit/absorb | **shape** — how it is distributed, incl. exact zeros |
|---|---|---|
| **fan-out (per key)** | `c_j = c_φ(k_j, γ, ν_j)`, equality at a predicted value | `τ_j = τ^K(k_j, s_·j)` → sparsemax → exact zeros |
| **fan-in (per query)** | `min(τ_i R̃_i, 1)` — spent amount predicted beneath a fixed unit **ceiling** | `τ_i = τ^Q(q_i, Ã_i·)` → dual `θ_i` → exact zeros |

Both run over **the whole margin or a declared part of it** (conditional activation;
selective exclusivity). ⭐ **Exclusivity is the pair**: a capacity alone bounds a total without
deciding membership; a shape alone redistributes a total the mechanism does not control; one
margin alone leaves the other's degree free. §32.0's framing ("the field argument is the novel
object") was too narrow — the field argument is one of *two* distinctions inside a *larger*
object, and stating only it undersold the paper.

### 34.1 ⚠️ MESH corrected again, in Luke's favour on the substance and against his framing on one fact

Luke: *"MESH seems to only do capacity on the column end."* **Not quite — MESH learns BOTH
marginals**, and the draft must say so or a reviewer will. Verbatim, App. A: *"We learn both
`a = m·softmax(h_a(Z))` and `b = m·softmax(h_b(X))` with neural networks `h_a`, `h_b` shared
across the m slots or n input elements respectively."* So MESH fills **both capacity cells**.

⭐ **But Luke's substantive point is right and is the decisive one: MESH fills neither shape
cell.** Verified against the ICML version:
- the attention matrix is the output of **Sinkhorn on a modified cost — dense, no exact zeros**;
- MESH's contribution is gradient descent on the **cost matrix** to minimise entropy, ~4 steps;
- ⚠️ the strength is a **fixed global hyperparameter** (a learning rate λ in Eq. 11), **not
  predicted per slot or per input by any head**.

**Consequence, and it is the strongest single sentence available against the nearest prior work:**
MESH is a full-support mechanism, so `Prop. fullsupportceiling` pins its column precision at
`m_j/n_q` **exactly as it does row-softmax's, however well its capacities are learned.** That is a
difference in the modelling object, not in the parameterization.

Two smaller distinctions, both downstream of the same place: `h_a`/`h_b` read their participant
**alone** (the field argument, `Cor. necessity`); and their marginals are **equalities** in a
doubly stochastic problem where MarSea's fan-in is an **inequality** cap, which is what lets a
query spend less than a unit.

**Link**: <https://arxiv.org/abs/2301.13197>, ICML 2023, PMLR 202 —
<https://proceedings.mlr.press/v202/zhang23ba/zhang23ba.pdf>. Yan Zhang, David W. Zhang, Simon
Lacoste-Julien, Gertjan J. Burghouts, Cees G. M. Snoek, *Unlocking Slot Attention by Changing
Optimal Transport Costs*. MESH = "Minimize Entropy of Sinkhorn".

### 34.2 What was built

- ⭐ **A new main-text table in both drafts** (`tab:object` in ICLR, replacing `tab:taxonomy` in
  TMLR) placing **thirteen** mechanisms on the four cells: row-softmax, geometric bias/pairwise
  gate, `softmax₁`/sink, differential attention, row-entmax/top-k, adaptively sparse, Selective
  Attention, Sinkformer, LOTFormer, Slot Attention, MESH, expert choice, MarSea. **MarSea is the
  only entry filling all four.** The TMLR version keeps the `P_j` column, which makes the argument
  readable by columns: *every dense entry is pinned at `m_j/n_q`; every free capacity is unbounded
  by Thm. 1; the literature supplies the two halves separately.*
- **Contribution (i)/1 rewritten** in both drafts around the 2×2, with the capacity half
  explicitly disclaimed (`zhang2023mesh` cited **in the contribution**) and the shape half,
  the field argument and the inequality claimed.
- **Abstract sharpened** (ICLR) to name capacity and shape and to say prior mechanisms supply one
  cell or the other, never all four.
- **App. C's MESH paragraph rewritten** with the quoted both-marginals sentence, the dense/global-λ
  finding, and the `Prop. fullsupportceiling` consequence.
- `\usepackage{bm}` added to both preambles.

### 34.3 ⚠️ Two honest notes now in the TMLR draft

Recorded so they are not "discovered": **the fan-in capacity is weaker than the fan-out one** — an
inequality beneath a fixed unit ceiling, where `c_j` is an equality at a predicted value — and
**on the fan-in margin a single parameter `τ_i` serves both cells** (the amount spent and, through
`θ_i`, the shape), where the fan-out margin has two separate heads. The asymmetry is deliberate
and already argued in Sec. `method-fanin`, but it is an asymmetry, and a formulation predicting
the fan-in capacity independently remains open.

### 34.4 State

| | `marsea_iclr.tex` | `marsea.tex` |
|---|---|---|
| length | **37 pp** | **86 pp** |
| build | 0 err / 0 undef refs / 0 undef cites / 0 overfull | same |
| harness | 459,621 checks, 0 violations | same file |
| object table | `tab:object`, main text p3, 13 mechanisms | `tab:taxonomy`, rebuilt, + `P_j` column |

### 34.5 Open — Luke's row-side selective exclusion

Luke: *"I will propose a very light-weight selective exclusion scheme for MarSea on the rows."*
Not yet received. When it arrives, the accounting it has to clear already exists in
`sec:selective` / App. E and should be run statement by statement rather than assumed:
`Prop. selrestrict` (both margins must be slices of **one** object, or `Prop. stagecomp` and
`Cor. joint`'s consistency hypothesis reopen); **the unit cap then bounds only in-relation mass**,
so the row total can approach two (App. E cost (c), where two repairs are named and neither
chosen — a lightweight row scheme may be the thing that chooses); `Prop. selrestrict(iv)`'s
arrival-sealed condition if it is to survive the causal form; and `Prop. rowfid`/`rowgraded`,
which should transfer verbatim with `[n_k] → E_i·` since they are statements about a truncation
over an index set.

---

## §35. ⭐ The two-step selective form becomes the mechanism

*Same session. Luke proposed a two-step selective-exclusion scheme, we converged over four
exchanges, and it is now the mechanism in both drafts rather than App. E future work.*

### 35.0 ⭐⭐ Standing decision — supersedes §32.0 and §34.0 in one respect

**Both margins take two steps. Step 1 inherits the mass an unmodified layer would allocate; step 2
re-shapes that mass over the exclusive relation at a predicted exclusivity, producing exact
zeros.** In consequence:

⭐ **The fan-out bound is on DEGREE, not on MASS.** MarSea does not predict how much a key emits
into its relation — only how many members receive any of it. `Cor. capacity`'s **cardinality**
half is claimed; its **capacity** half is explicitly disclaimed in both drafts. Do not reintroduce
a predicted budget without revisiting E1, Table 1 and contribution (i) together.

**Fan-out.** Step 1: `Ā_·j = c_j·softmax(s_·j)` with `c_j := Σ_i A^sm_ij` (standard attention's
column mass) and `c'_j := Σ_{i∈𝓔_·j} Ā_ij` (the relation's quota). Step 2: entmax over `𝓔_·j`
alone, `Ã_ij = c'_j p_ij`, complement keeps `Ā`.
**Fan-in.** Step 1: project the whole row onto `Σ_j a_j ≤ 1`, giving `c'_i`. Step 2:
`argmin ½‖a/τ_i − Ã‖²` s.t. `Σ_{𝓔_i·} a_j = c'_i`, complement keeps step 1.

### 35.1 The exchange, and two things I got wrong

- I first analysed the wrong construction: App. E's "restrict both programs and leave the
  complement free." Luke's step 1 **fixes the complement**, which is what retires the objections.
- ⚠️ I then justified the equality `Σ_j a_j = 1` by saying it keeps Prop. 4's hypothesis intact.
  **Sloppy** — Prop. 4 is a statement about *baselines*; whether MarSea's rows sum to one is
  irrelevant to it. Corrected, and Luke took `≤ 1`, preserving the sub-unit row (69.3% of rows
  spend less than a unit in the harness).
- ⚠️ Luke's step 2 as first written, `‖a·τ_i − Ã‖`, **inverts the direction of `τ_i`** — support
  4→10 as `τ_i` grows, against 10→2 under `a/τ_i`. Confirmed by table; he took `a/τ_i`.
- ⭐ **The finding that changed the design**: `c_j` matching standard attention throws away the
  capacity half. Measured on E1's construction — whole-column mass and the relation's quota both
  track softmax at +0.347/+0.348 against MarSea's old +0.000. I proposed
  `c'_j = min(inherited, c_φ)` to keep both; **Luke declined and accepted the loss**, which is a
  legitimate and simpler design.
- ⚠️ **But my E1 test was the wrong regime**, and Luke was right to push. With the relation
  *fixed* and distractors added *outside* it, the inherited quota is flat (slope −0.0001) while
  the column grows at +0.283. And when the relation *grows*, **degree stays at `m_j` exactly**.
  The claim does not disappear; it relocates from mass to degree, which is where the abstract
  already had it ("MarSea's support size does not involve `n`").

### 35.2 Two new propositions, both verified

- **`prop:twostep`** (conserves and cannot relocate): (i) `Σ_i Ã_ij = c_j` exactly at any `𝓔`;
  (ii) hence `Σ_j c_j = n_q` **derived, not imposed** — which retires App. E cost (d) and
  improves on the previous design, where the anchor was imposed by hand and had no
  mask-independent analogue; (iii) `Σ_j A_ij ≤ 1`, sub-unit rows preserved; (iv) the complement
  is fixed by step 1 and untouched by step 2, so **no margin claim can be met by relocating mass
  out of the relation**. 0 violations over 14,937 columns / 80,741 rows.
- **`prop:scopemono`** (shrinking a relation only helps): `δ_j` non-decreasing, relative width
  non-decreasing, interval containment given Assumption 2 on the larger set, and ⭐ **the rescue
  direction** — Assumption 2 fails on `𝓔` but holds on `𝓔'` for **11.2%** of columns drawn.
  0 violations over 6,000 / 4,506.
- **`Cor. capacity` cardinality half**, measured: degree `= m_j` and precision `= 1` as
  `|𝓔_·j|` grows. 0 / 10,000 each.

### 35.3 E1 rebuilt — it measures degree now

The old probe measured **mass** and would have shown MarSea tracking softmax. Rebuilt: `m_j = 4`
fixed, `|𝓔_·j|` from 4 to 256, `τ_j` from Thm. 2's interval.

| `\|𝓔_·j\|` | 4 | 16 | 64 | 256 |
|---|---|---|---|---|
| MarSea degree | 4.00 | 4.00 | 4.00 | **4.00** |
| MarSea precision | 1.000 | 1.000 | 1.000 | **1.000** |
| softmax degree | 4 | 16 | 64 | **256** |
| softmax precision | 1.000 | 0.250 | 0.062 | **0.016** |

Slope **−0.0000** against softmax's **+1.0000**. Stronger than the flat mass line it replaces, and
the comparator's side is an identity (Prop. 14), not a tendency. Panel (b) is **saved and made
more load-bearing**: with the capacity inherited, `τ_j` carries the whole fan-out claim, so
`τ^K` now reads `ν_j` and panel (b) asks whether the statistic *that* head sees carries `m_j`.

### 35.4 App. E's cost list, rewritten

| | status |
|---|---|
| (a) separation force vanishes at `\|𝓔_·j\|=1` | ⚠️ **survives** — carry `\|𝓔_·j\|≥2`, report the distribution |
| (b) conservation weakens to `μ/n_k` | **retired** — the complement is fixed by step 1, nothing to park |
| (c) unit cap stops covering the row | **retired** — step 1 caps the whole row before restriction |
| (d) `Σ_j c_j=n_q` has no analogue | **improved on** — it is now derived |
| (e) *new* ⚠️ | one parameter per margin carries the whole claim; `τ_j→0` alone returns standard attention |

### 35.5 State

| | `marsea_iclr.tex` | `marsea.tex` |
|---|---|---|
| length | **39 pp** | **88 pp** |
| build | 0 err / 0 undef refs / 0 undef cites / 0 overfull | same |
| harness | **690,483 checks, 0 violations**; both drafts' tables reconcile exactly | same file |
| E1 | degree and precision vs `\|𝓔_·j\|`, 4→256 | same figure |

⚠️ **Pending on the TMLR draft only**: the *consequences* pass. `marsea.tex` has the two-step
method, both propositions, the rescoped corollary and the new table rows, but material built on
the predicted budget survives elsewhere in its 88 pages — `rem:subunit`, the V4b family, and the
taxonomy table's MarSea row. The ICLR draft is fully consistent; the TMLR draft needs a sweep for
`c_j`-as-predicted before it is submitted anywhere.

---

## §36. Fifth adversarial audit — the two-step form had two bugs; both fixed

*2026-09-02. Two independent agents (a cold implementation from Sec. 2 + spec; a hostile
reviewer on the experiments) plus my own pass. Full account: project doc
`claude/marsea_iclr_audit5_2026-09-02.md`.*

### 36.0 ⭐⭐ Standing decisions — supersede §35.0 in three respects

1. **Fan-out step 1 is the layer's own row-softmax column, `Ā.ⱼ := Aˢᵐ.ⱼ`**, not a column softmax
   rescaled to `cⱼ`. The column-softmax form preserved totals but not entries; "gates closed =
   standard attention" was false on 600/600 draws. Now true to 1e-16.
2. **Fan-in step 2 is a CAP, `Σ_{𝓔ᵢ.} aⱼ ≤ c'ᵢ`**, not an equality. The equality drove `θᵢ < 0`
   for small `τᵢ` and **resurrected Stage-1 zeros** on 314/600 draws, contradicting
   `prop:stagecomp(iv)`. The cap keeps `θᵢ ≥ 0` for every `τᵢ > 0`. The other coherent fix —
   equality with `τᵢ ≥ 1` — was not taken; it kills the slack regime.
3. **There are no gates.** `ω^K`, `ω^Q` are removed; the relation `𝓔` is the sole membership
   object, an empty relation is standard attention entrywise, and the row trigger is exactly
   `𝓔ᵢ. ≠ ∅`. The query gate was provably unreachable (max |ΔA| = 0.0).

### 36.1 ⚠️ Errors of mine that this audit caught

- My fan-in check (§35.1) tested row sum and complement, **not zero-permanence**. The harness's
  `stagecomp` block tested a hand-drawn floor, never `eq:rowprog`. The spec claimed INV-6/7 were
  covered; they were not.
- "R̃ᵢ is exactly one unit's worth of supply" — written by me in the cleanup pass, false.
- The old two-regime prose ("`Σ uⱼ ≤ 1/τᵢ`", "`min(τᵢR̃ᵢ, 1)`") survived my equation rewrite and
  contradicted it.
- `v1_probe.py` panel (b) used **negative temperatures** (−6.04, −18.50) on 3/72 columns where
  Assumption 2 failed on the draw; I had attributed the resulting deficit to "targets not exactly
  tied." Fixed; values moved to 1.59/3.20/28.21.
- `prop:twostep` and `prop:scopemono` had **no proofs** while Sec. 3 promised them. Now proved.
- App. H's E1 paragraph still described the retired mass experiment.

### 36.2 ⭐ E2 was arithmetically inverted; E3 is now load-bearing

Comparator precision is `m/n` (Prop. 14), rising with `m` at fixed `n`, so MarSea's gap
`1 − m/n` **shrinks** with `m` at ceiling. The old S2 gate ("gap grows with m") would have
rescoped the paper on the theory's own success signature. E2 now tests Thm. 2 (hit rate by `m`,
both endpoints ~ `1/m`); E3 (sweep `n`) is Fig. 2(a) on real data and is load-bearing, reporting
per `n` the `δⱼ > 0` fraction and precision on those columns vs all.

### 36.3 ⭐⭐ Open — the column theory has no ground truth in multi-hop QA (Luke's call)

Evidence selection is the answer query's **row**, which is query-local; Thm. 1 does not speak to
it and row-entmax controls its own membership. On columns, gold gives `Kᵢ` not `Tⱼ`; one answer
query makes `mⱼ ∈ {0,1}` — the conceded regime, and `mⱼ = 0` forces `Pⱼ = 0`. **E7's interval-hit
rate cannot be run on the chosen data.** Options: annotate token-level `Tⱼ`; keep QA as the row
test and add a typed column task (coreference, signal→lanes); or reframe around the row. Not
decided unilaterally.

### 36.4 Also applied

`def:fidelity` row support on the relation + **relation recall** as a named quantity; S0 gate
→ `M > mⱼ`; `νⱼ` init 1; Table 1 / `tab:stages` / notation / App. B / App. D / Prop. 12 /
"coverage residual" de-budgeted; inherited-capacity limitation stated in Sec. 2.1 ("losers lose
reach, not mass"); row-margin honesty note in Sec. 2.2; spec revised (preamble flags the three
changes; INV-10..12; all fidelity on the relation; `δⱼ` over the relation's complement; B5
matches relation support). TMLR: the two equations, the proposition and the table rows ported.

### 36.5 State

| | `marsea_iclr.tex` | `marsea.tex` |
|---|---|---|
| length | **39 pp** | **88 pp** |
| build | 0 err / 0 undef / 0 overfull | same |
| harness | **853,965 checks, 0 violations**; tables reconcile exactly | same file |

⚠️ Still open on the ICLR draft: C4 (App. H baseline paragraph's row-margin argument), C5 (an
aggregation paragraph in App. H: which layers, which head defines the row, passage support as a
union over token keys), C6 (constant-difficulty citation). TMLR: the consequences pass of §35.5
still pending, now plus §36.0's three changes beyond the two equations.

---

## §37. Rename: JOCEA → MarSea

*2026-09-02. Luke's call, taken after the fifth audit.*

**MarSea** = *multivariate relations by selective exclusion in attention*. The old expansion
("joint optimization in competitive-exclusion attention") named a design whose predicted budget
is gone and whose joint program is now two inherited-then-reshaped steps; the new one describes
the mechanism as it stands. A web search found no ML collision.

**Titles changed with it.** ICLR: *MarSea: Multivariate Relations by Selective Exclusion in
Attention*. TMLR: the same, with the old subtitle kept. The expansion is spelled out at first
use in both abstracts and nowhere else.

**Paths.** `build/joca_iclr.{tex,bbl}` → `build/marsea_iclr.{tex,bbl}` with
`\bibliography{marsea_iclr}`; `tmlr/joca.{tex,bbl}` → `tmlr/marsea.{tex,bbl}` with
`\bibliography{marsea}`; `jocea_impl_spec.md` → `marsea_impl_spec.md`;
`jocea_normalize` → `marsea_normalize` in the spec. Project docs moved to `marsea.tex`,
`claude/marsea_iclr.{tex,bbl}`, `marsea.bbl`, `claude/marsea_master_record.md`,
`claude/marsea_implementation_spec.md`; the dated audit docs were rewritten under
`claude/marsea_iclr_audit{4,5}_*.md` and the `joca_*` originals deleted.

⚠️ **Deliberately not renamed**: `joca_positioning_decisions.md` and `joca_ws`, which name
superseded documents that no longer exist as live files, and this record's references to them.

**Housekeeping.** Ten spent one-shot patch scripts (`tierB1-4.py`, `patch1-4.py`,
`tierB_tmlr.py`) and the pre-audit snapshot `joca_iclr_BASELINE.tex` moved to `archive/`. They
targeted filenames that no longer exist and would have confused a fresh session or Claude Code.

**Verified after the rename**: both drafts build at 0 errors / 0 undefined refs / 0 undefined
cites / 0 overfull (39 pp, 88 pp); harness at **853,965 checks, 0 violations** across 54 rows;
both drafts' verification tables reconcile against it exactly; every paper label the
implementation spec cites still resolves; `v1_probe.py` reproduces its table.

---

## §38. Notation `\bar c`, explicit relation/complement layout — and a formulation caught before it landed

*2026-09-02. Luke revised the four mechanism equations.*

**Applied**: `c'_j → \bar c_j`, `c'_i → \bar c_i` everywhere (both drafts, spec, harness labels);
the `\bar A` alias dropped in favour of `A^{sm}` directly; and Luke's explicit layout for Eq. 3
and Eq. 7 — the relation indexed as `\tilde A_{𝓔.ⱼ,j} = \bar c_j p_{𝓔.ⱼ,j}` with
`\tilde A_ij = A^{sm}_ij ∀ i ∉ 𝓔.ⱼ` written beside it, and likewise on the row — which reads
much more clearly than the implicit "off the set" prose.

⚠️ **Not applied, and why — verified numerically before deciding:**

1. Luke's Eq. 2 wrote `\bar c_j := Σ_{i∈𝓔} s_ij` — a sum of **raw scores**. Scores are signed, so
   this is **≤ 0 on 49% of columns**, and a non-positive scalar cannot scale a simplex vector into
   an allocation. His prose says "the result of standard attention," which is `Σ_{i∈𝓔} A^{sm}_ij`
   — what the draft already had. Kept the prose's meaning.
2. Luke's Eq. 3 set the complement to `\tilde A_ij = s_ij` (raw scores). Same slip; kept
   `A^{sm}_ij`, which is what "the rest of the column is standard attention" means.
3. ⭐ Luke's Eq. 6 wrote `a^{(1)}_{i·} = softmax(\tilde A_{i·})`. **This would destroy the
   mechanism**: `\tilde A` is already attention-space, so it is a second normalization that
   rescales every off-relation entry away from `A^{sm}`; and a Stage-1 exact zero is a *weight* of
   0, and `softmax` of a 0 is `e⁰/Z > 0` — **the zeros survive on 1% of columns**, and the fan-out
   degree bound (Fig. 2(a)) is gone. Kept the unit cap, which is the identity on any row summing
   to ≤ 1 (i.e. every row no column relation concentrated onto). The draft now says explicitly
   that step 1 is not a second softmax and why; the spec has the same note at the code line.

**The coherent reading of Luke's intent is the audit-5 design** — softmax once, in score space,
producing `A^{sm}`; both relations re-shaped in attention space and written back. If instead
Stage 1 were meant to act on *scores* with the softmax deferred to Stage 2 (which is what the
literal `softmax(\tilde A)` implies), the column zeros would have to be encoded as `−∞` rather
than `\bar c_j · 0`, and then `\bar c_j` and the magnitudes of `p` would play no role — Stage 1
would be pure support selection. That is a different mechanism from the one the paper describes,
and it is flagged to Luke rather than chosen.

State: both drafts 0 err / 0 undef / 0 overfull (39 pp, 88 pp); harness 853,965 / 0; tables
reconcile; spec labels resolve.

### 38.1 ⭐ Standing decision, confirmed by Luke 2026-09-03

**Softmax runs exactly once, at Stage 1 step 1, producing `A^{sm}`. No normalization of any kind
runs after it.** Both stages' step 2 re-shape a relation's slice of `A^{sm}` (or of `Ã`) in
attention space beneath an inherited quota and write it back; the complement is untouched.
`\bar c_j := Σ_{i∈𝓔.ⱼ} A^{sm}_ij` and `\bar c_i := Σ_{j∈𝓔ᵢ.} a^{(1)}_ij` read attention
weights, never raw scores. The literal `softmax(Ã)` of the 2026-09-02 proposal is withdrawn;
it would have destroyed the exact zeros (1% survival) and the degree bound with them.

## §39. Sixth pass — implementation spec v3, the backbone named, and a false limit retired

**Trigger.** Luke: "Give the implementation and experiments specs another thorough pass … One
question: what would be the borrowed backbone for all baselines and MarSea? please be specific."

### 39.1 The backbone (decision, verified 2026-09-03 against the Qwen blog, HF configs, transformers `main`)

| role | model | facts |
|---|---|---|
| primary | **Qwen/Qwen2.5-1.5B** (base) | Apache 2.0; 28 layers; 12 Q / 2 KV heads (GQA 6:1); head dim 128; hidden 1536; 32K native context; bf16 |
| cross-family | **meta-llama/Llama-3.2-3B** (base) | Llama 3.2 Community License, gated; 28 layers; 24 Q / 8 KV; hidden 3072; 128K via llama3 RoPE scaling |
| optional scale check | Qwen/Qwen2.5-7B | Apache 2.0; 28 Q / 4 KV |

Not Qwen2.5-3B: it is under the *Qwen Research License*, not Apache. Base checkpoints, not
Instruct, so no arm inherits an answer-format prior.

**Patch point**: `transformers` `eager_attention_forward` (Qwen2 and Llama share it): the single
line `softmax(attn_weights, dim=-1, dtype=float32)`. `S = attn_weights` after scaling and the
additive mask is `eq:scores`; RoPE is already in `q, k`; `repeat_kv` has expanded K to Q-heads.
Every arm (MarSea, B0–B5) is a `Normalizer` called at that line; projections, RoPE, GQA, value
mix, MLPs are the backbone's. Module swap on the patched layers only (subclass of
`Qwen2Attention`, weights copied); unpatched layers keep SDPA. Patched layers = those carrying
the backbone's retrieval heads (Wu et al. 2024, arXiv:2404.15574), `L_patch = 4`, ablated in E9.

Both drafts' Setup paragraph now names this (new bib entry `wu2024retrieval`, added by hand to
both `.bbl` files — there is no `.bib` in the build; **do not run bibtex**, it empties the
`.bbl`; I did exactly that this session and restored both from the 09-02b package).

### 39.2 ⚠️ A false claim, found numerically and retired from both drafts

Three places in the ICLR draft and one in the TMLR draft said "`τ_j → 0` alone returns standard
attention". **False.** By Lem. 1, `τ_j → 0` makes every prefix satisfy `G < 1`, so the support is
the whole relation and each entry tends to `\bar c_j/|𝓔.ⱼ|` — the *uniform* re-shaping, a column
of the wrong shape. Only `𝓔 = ∅` is standard attention (`prop:twostep`(vi)). Harness now checks
both facts ("App. E(e)" rows: 0/1992 each). Consequences: (a) App. E cost (e) and App. D's
degeneracy paragraph rewritten around *two* collapses — relation emptying (= B0) and `τ_j → 0`
(≠ B0, shows as degree `|𝓔.ⱼ|` on touched columns); (b) the implementation must warm-start
**through the relation** (near-empty, `ρ₀ = 5 %` of visible pairs), never through a small `τ`.

### 39.3 Consequences pass on the ICLR draft (gate vocabulary, E3 load-bearing)

- Sec. 5 opener said "E2 is load-bearing"; now E3, consistent with §36's inversion fix.
- `tab:compute`: S2 = E3 (+E2, E4), gate "precision flat in `n` on `δ_j>0` columns; hit rate not
  falling in `m`"; 60 h; S3 = E5/E7/E8, 68 h; totals unchanged (224 / ≈336).
- App. H baselines paragraph was still describing MESH as the field-argument test and predicting
  "loses on E2's slope". Rewritten: MESH = capacity-without-shape (both marginals learned, dense),
  decisive for contribution (i)'s shape claim; key-alone `τ^K` = the field-argument test, scored on
  E7's hit rate.
- E8 rewritten without gates: coverage on both margins, `|𝓔.ⱼ|`/`|𝓔ᵢ.|` histograms, `k*` by
  `|𝓔.ⱼ|`, one fan-in trigger (row-rate ≥ column-induced rate *by construction*; a violation is a
  bug, not a finding), `τ_i R̃_i/\bar c_i` about 1, membership-vs-`δ_j` correlation.
- E9: "gate policy" → relation parameterization (rank, pairwise vs key-alone, layer choice).
- Residues fixed in Sec. 4 application para, limitations, App. C, App. D (×5), App. G, efficiency
  reporting. App. D's "conditional on `τ_i R̃_i > 1`" corrected to `> \bar c_i`.
- Left alone deliberately: "elementwise gate" as the name of the *comparator* class (Thm. 1), and
  §2.3's sentence that the old gates are the empty-relation case.

### 39.4 Spec v3 (`rec/marsea_impl_spec.md`, project `claude/marsea_implementation_spec.md`)

New or rewritten: §1 backbone + patch point + facts to re-verify; §4.2 init through the relation
with calibrated `b0`; §4.3 gradient path — straight-through on the *interpolation* between the
untouched and re-shaped entry at the four places `𝓔` enters (hard forward keeps every INV; biased
backward stated, hard-concrete as the E9 alternative); §5.2 head architectures and field
statistics; §6 module swap, GQA (everything per Q-head; `U_φ`/`τ^K` read the KV-head key), dtype,
frozen-prefix decoding with the `𝓔=∅` token-identity test, memory arithmetic (≤4 patched layers
at 16K on 80 GB), exact chunked two-pass implementation with T11; §6.7 retrieval-head layer
selection; §9 training recipe — Phase A (`𝓔` forced empty, = plain LoRA; B0 *is* Phase A
continued) then Phase B, matched steps, LoRA r=16 on all layers for every arm; §12.3 the C2
constructions (value-key, question-key, supporting-passage columns) with the S0 **routing
check** that licenses them (mass fraction > 0.5 on ≥ 80 %), and the fallback if none passes;
§12.4 logging schema; §13 E1–E9 → code; §15 decisions Claude Code must not take alone.
INV-13, H9, H10, T0, T11–T13 added. Harness count updated to 857,949 / 0 / 56 rows.

### 39.5 State

Both drafts 0 err / 0 undef (ICLR 40 pp, TMLR 88 pp); harness 857,949 / 0 / 56; verification
tables carry the new row; spec labels resolve. Still open from audit 5: C4 (App. H baseline
row-margin argument), C5 (aggregation paragraph — now largely covered by spec §12.2, paper still
points to App. H), C6 (constant-difficulty citation); TMLR consequences pass (`prop:special`
still states the old `c_j ≡ n_q/n_k` limits; rem:subunit; V4b family; taxonomy row).

## §40. C2/C3/C4 settled by composition, not by a new component (2026-09-03)

**Discussion (no commit until the end).** Luke asked for (d) passage-level `T_j`, and raised the
`m_j = 0` column with a learnable cutoff on `max(s_·j)`. Sequence of positions:

1. (d) is feasible from shipped annotation: MuSiQue's decomposition gives the bridge entity as a
   string, so `T_j` at passage level = {later gold passages consuming `j`'s intermediate answer,
   question, answer} (`m_j` 2–5), and at token level = later mentions of the bridge string
   (exact match). RULER MV-NIAH has the same structure (`m` needles share a key phrase → `2m`
   later mentions). No new training runs; a routing check on B0 is the precondition (spec §12.3).
   Option (e) held in reserve: code def–use chains (AST-exact). **Luke has not yet committed to (d)**;
   the App. H paragraph says only "where those come from is stated with E7".
2. Abstention: I first proposed a per-key rule `\bar c_j/|𝓔.ⱼ| < λ/n_k` on the relation slice
   (scale-free; not `max(s_·j)`, which is row-normalised in the wrong direction and a percentile
   fixes the silenced *fraction*). Luke asked whether it is a trained gate — yes, in any form.
   One proposition (sharp `λ`-interval under a key-side margin) would be its theory; no
   separation theorem exists in the single-query regime because the rule is row-local there.
3. **Decision (Luke: "OK, draft it")**: no abstention component. The single-target column is
   the `τ_j ≥ 1/δ_j` regime of Thm. 2 (`W_j = 0`), continuous in `τ_j` (sparsemax 1-Lipschitz)
   — the "continuous gate" Luke asked for *is* `τ_j`, and the top gap in `col_stats` is its
   detector. At `m_j = 0` the key reaches its argmax query; a distractor whose argmax is not the
   answer query is an exact, permanent zero on the answer row after Stage 1; only distractors
   concentrating onto the answer query reach `θ_i`. Composition, no new part; INV-1/2 intact.

**Written.** Sec. 2.1 "Two limiting columns" paragraph (also fixed a dangling "with $H^T_α$…"
fragment there). App. H: C4 corrected (row-margin argument withdrawn; measured instead), new
paragraph "single-target and empty column are row-side tests" with the three-way distractor
split for E8. Harness: three rows (argmax at `τ ≥ 1/δ`, 0/9000; 1-Lipschitz continuity, 0/3000;
composition zero on the answer row, 0/12082; 8.5 % of keys left to the row program). Both
verification tables carry them. Spec §12.3a. Totals: **882,031 checks / 0 / 59 rows**.
Closed: C3 (mechanism story), C4 (measured), C2's `m_j = 0` half. Open: C2 multi-query
construction (d) — Luke's call; C5 largely in spec §12.2, paper pointer only; C6.

## §41. ⭐ Standing decision, re-confirmed 2026-09-04: step 1 is the ROW softmax (standard attention)

**What was cleared up.** Luke asked whether `A^{sm} := softmax_row(S)` in Sec. 2.1 was a typo for
`softmax_column(S)`, then whether `softmax_row` meant `softmax(s_·j)` over each column. It does
not. `softmax_row` normalises each **row** `i` over the keys a query looks at —
`A^{sm}_ij = exp(s_ij)/Σ_{j'} exp(s_ij')`, `Σ_j A^{sm}_ij = 1` — i.e. ordinary attention. The
column stage then reads the **column sums** of that row-softmax: `c_j = Σ_i A^{sm}_ij` and
`\bar c_j = Σ_{i∈𝓔.ⱼ} A^{sm}_ij`. No column softmax runs at step 1.

**Where the confusion came from.** Luke's 09-02 formulation had step 1 as `softmax(s_·j)` over
the whole column, rescaled by `c_j` to match standard attention's column total. Audit 5 replaced
it (§36): that form preserves the column *sum* but not the *entries*, so an empty relation is no
longer standard attention (harness: 600/600 disagree), rows do not sum to one before Stage 2's
cap (the cap binds almost everywhere), and the B0 warm start is lost. Luke's 09-03 confirmation
("softmax once at Stage 1 step 1, nothing after") was read as agreement with `A^{sm}`; today's
exchange established that the *direction* had not been said out loud. Sec. 2.1's parenthetical
already records the earlier column form.

**Reconciliation.** The column-normalised distribution over queries is not gone: step 2 on the
relation, `p = entmax_α(τ_j s_{𝓔.ⱼ,j})`, *is* a column softmax over the relation at `α → 1`,
and at `α = 2` the same object with exact zeros — scaled by `\bar c_j`, whose total step 1's
row-softmax fixed. And a row normalisation must exist somewhere in any attention mechanism
(the value mix needs each query's weights to be a distribution); since Stage 2 is a cap, not a
softmax, step 1 is the only place it can be.

**Decision (Luke, 2026-09-04): keep the method as written.** `A^{sm}` is the layer's own
row-softmax; the column stage reads its column sums and re-shapes the relation slice in step 2;
no normalisation of any kind after step 1 (§38.1 stands). Symbols, for the record:
`softmax_row(S)_ij = exp(s_ij)/Σ_{j'} exp(s_ij')` (rows sum to 1; USED);
`softmax_col(S)_ij = exp(s_ij)/Σ_{i'} exp(s_i'j)` (columns sum to 1; NOT used at step 1;
appears only as the α→1 limit of step 2 on the relation).

## §42. Spec v4 — the cold-read pass (2026-09-04)

**Method.** A second engineer (subagent, no access to this record) implemented spec v3 + paper
Sec. 2 from the text alone in PyTorch (`reference/marsea_cold.py`), tested every INV, compared
with `verify_all.py`'s Prop. A loop, and logged 31 ambiguities (`reference/ambiguities.md`).
The implementation itself passed everything (0 failures on INV-1..13, T0/T1/T13, gradient flow;
3e-15 vs the harness in fp64) — the *mechanism* text was sound; the *engineering* text was not.

**Would have broken a real run (all fixed in v4, marked [v4] in the spec):**
- `proj_le` at a zero row quota — the MAJORITY row under random scores (65 % of rows with a
  non-empty relation) — NaNs the backward in a batched `where`. Now `proj_le(v, s≤1e-9) := 0`
  with a safe divisor; H6 corrected (it wrongly said `g = 0` covers this); H11 general rule.
- `column_stats` divides by std = 0 on the last key of every causal block (one visible query).
  Guards specified; H12.
- `nu` on an empty column was `1/eps` → now `:= 1`.
- B2/MESH was unimplementable and its "rows sum to b_i" claim is FALSE under a causal mask
  (two-marginal transport generically infeasible). Pinned to Zhang et al. 2023 (MESH step:
  4 normalised-gradient descents on plan entropy, straight-through, `ε = 1` = softmax kernel);
  we end on the row scaling and log the column residual. **App. H now states this limitation
  of the baseline** (one sentence added to the MESH paragraph).
- INV-9 is false with a live `τ^K` (3310/6686 decreases) — holds only at sealed `τ_j`, which is
  the proposition's hypothesis and why §7 seals; T7 runs frozen.
- Fan-in step 2's target is `Ã` (pre-cap) with quota from `a^{(1)}` (post-cap); the `a^{(1)}`
  mis-reading passes every INV → INV-14 + T14 (a wrong-reading detector).
- §13 named no data sources/formats/values → §13.1 (RULER configs and args, MuSiQue/HotpotQA/
  IHEval fields, prompt template, inert-padding = same length with `num_needle_k = 1`, depth
  sweep for the position profile, eval decoding).

**Also fixed:** `sorted_prefix_stat` 0/1-based off-by-one; `vis` is `[B,1,n_q,n_k]` with KV
offset and fully-masked-row rule; ST gate masked by `vis`; `b0` = constant coordinate of the
paper's pairwise form (no paper change), trainable, calibrated at Phase-B start; post-RoPE keys
(stated); head-input normalisation (LN on vectors, `sign·log1p` on scalars); `τ` init as a
median calibration, not a per-column bias; B1's `+1` after scaling and masked sum; B4 bisection
details; B5 records `E` too, renormalises to the relation's softmax mass, runs on **B0's**
weights with MarSea's trace; Phase A = softmax for every arm, one cosine over 3500 steps, no
packing (right padding); T11 tolerance relative/fp64; T15 NaN sweep; Diagnostics carry `ψ_j`,
`supp_rel_i`, `logits`.

**Package:** `reference/` (cold implementation, tests, harness comparison, ambiguity log,
README saying what must be brought to v4 before reuse). Both drafts unchanged except the App. H
MESH sentence; ICLR 41 pp, 0 err / 0 undef.


## §43. Round 2 — pseudo-code from all four sources; `τ_i = 1` corrected (2026-09-05)

**Why a second round.** Round 1 (§42) withheld the record and the harness to test whether the spec
stood alone. Luke asked for the round the real implementer will have: spec v4 + paper + record +
harness + reference. The engineer produced `round2/pseudocode.md` (608 lines, every function
cited to its source) and `round2/contradictions.md` (30 contradictions, 13 residual ambiguities).

**The one substantive finding — audit 5 was wrong about `τ_i = 1`.** §36 said "`τ_i = 1` is not an
identity element; the row's identity is `\bar c_i/R̃_i`". That compared step 2's output with `Ã`.
Relative to `a^{(1)}` — the row step 1 produced — `τ_i = 1` IS the identity on every row: on a
slack row the quota equals the supply; on a cap-binding row the quota is what step 1's floor
left, and uniqueness of the projection recovers the same floor (verified: 0 differences on 4,338
cap-binding rows). Paper Sec. 2.2 rewritten (with the retraction stated), App. D and the notation
table corrected, spec §5.4 reversed, E8 logs `τ_i` about 1. `\bar c_i/R̃_i` remains the point where
step 2's own dual turns positive; it is not the boundary that matters.

**Cross-document contradictions fixed in the paper:** notation table (`Ã = c_j p`, "dual of the
unit cap", "binds iff `τ_i R̃_i > 1`"); `Σ_j A_ij = min(τ_i R̃_i, 1)` (App. D); predicted-budget
vocabulary in related work / App. D / causal paragraph (`c_j^{(b)}` → `\bar c^{(b)}_j`, "capacity
holding approximately" → "over-admits"); App. E's opening still said the relation "is not part of
the mechanism evaluated" — now "since the fifth revision it IS the mechanism"; App. E's closing
no longer says fidelity needs typed structure only.

**Engineering facts learned (spec v4.1):** RULER's `niah.py` gives every one of `num_needle_k`
keys `num_needle_v` values, so v4's E2 grid (K = 32, V = 16) was ≈ 8K tokens of needles at L = 8K
— infeasible; restated E2 at K = 8 / L = 8K and E3 at L = 16K (constant across the sweep, as the
control requires; commits S2 to the chunked path). Needle depth is random in stock RULER (a
`--gold_depth` patch is needed for the position control). `niah.py` strips `answer_prefix` from
`input`; the prompt must add it back. With SDPA loading the patched layer receives no additive
mask (None or boolean by version) — `vis` is built inside the layer from the 2-D pad mask.
Decode-time fan-out made explicit (per-key re-solve on the grown relation at frozen `τ_j`;
incremental `\bar c_j`; only row `t`'s entry emitted). `ν_j` not detached (paper: carries a
gradient). B5 is teacher-forced only. Dense arms' precision on both domains. `T_j` default =
coreference by string match, **pending Luke** (§40).

**Record hygiene:** a reading-order banner at the top now says §§1–34 are history and which
sections carry the standing decisions (the engineer flagged §4's "authoritative" heading).

**Verdict returned:** pseudo-code sufficient without questions for the seven arms, training,
data and T0–T16 in teacher-forced mode; five decisions remain and they are Luke's (spec §17).

## §44. Compute decision: Nebius, two-stage (Luke, 2026-09-05)

**Decision.** Rent H100s on **Nebius AI Cloud** for the whole programme, in two stages:
(1) **1–2 H100 VMs** for setup, T0–T15, S0, S1, the step-time measurement and all CPU-bound data
preparation (RULER generation at 8K/16K, tokenisation), ≈ 4 days, VM stopped when idle;
(2) **one 8×H100 VM** for S2 (E3, E2, E4) and the minimal S4, ≈ 2 days wall for the reduced
programme. The decision followed a comparison of CoreWeave (self-serve unit is the 8-GPU HGX
node at ≈ $49/h ≈ $6.16/GPU-h; no 2-GPU option), Nebius ($3.85/GPU-h on-demand, $2.15 preemptible;
1- and 8-GPU VMs; egress free; shared FS $0.08/GiB-month) and RunPod (Secure H100 SXM ≈ $2.99;
region-locked network volumes; 8×H100 availability less predictable). Nebius was judged the
most balanced: ≈ 35 % cheaper than CoreWeave, sells the two shapes the plan needs, and its
reliability risk (≈ 38 status-page incidents May–Sep 2026, a 10-h `us-central1` degradation in
August, credits-only SLA, 24–48 h support) is tolerable for a workload of independent single-GPU
runs with checkpoint/resume. RunPod's ≈ 25 % further saving (~$400 on the reduced programme) was
declined as not worth the capacity and region-lock risk against the deadline; RunPod is the
**fallback account** (same container image and data tarball kept ready).

**Budget (list price, on-demand).** Reduced programme ≈ 380–500 GPU-h ≈ **$1,500–1,950**; full
programme ≈ 700–800 GPU-h ≈ $3,000–3,500. Preemptible only for E9 reruns.

**Programme shape that fits 25 Sep.** Train every arm at 8K; evaluate E3 at 16K through the
chunked path. Phase B 3,000 → 2,000 steps. Three seeds on the four decisive arms (MarSea, B0,
B2 MESH, B3 key-alone τ), one seed on B1 and B4; E9 reduced to three ablations (τ_i pinned at 1,
inherited vs uniform quota, pairwise vs key-alone relation). ≈ 17 runs + evaluation ≈ 150
H100-h of work; ≈ 8–10 days end to end.

**Operating rules.** European region (Finland or Paris) for both VMs and the shared filesystem;
file the 8×H100 quota request on day one; on-demand for S2; checkpoint LoRA + heads every 250
steps to the shared FS; every run idempotent on restart; a queue script feeding eight
single-GPU processes (`CUDA_VISIBLE_DEVICES` per process, no distributed training) and exiting
when the queue drains; daily copy of checkpoints and parquet logs off-platform; stop VMs
between sessions; S1 measures step time at 8K and applies the pre-committed rule (> 1.2 s per
sequence → two patched layers or 4K training). Treat the 2-GPU stage as the platform's
reliability test before paying for the 8-GPU stage.

**Warm start, restated (asked 2026-09-05).** Required: Phase B starts at (near-)standard
attention through a near-empty relation (ρ₀ = 5 %), never through a small τ_j (§39.2); τ_i = 1 is
the identity (§43); an exactly empty relation gets no gradient. Convenience: Phase A (500 softmax
steps, run once, forked into every arm; B0 = Phase A continued) — droppable without touching a
claim, ≈ 45 min per seed. Nothing is warm-started from a MarSea run.

## §45. The fine-tuning procedure written out (2026-09-06)

Luke asked for the MarSea fine-tune "in detail". Written as `marsea_training_procedure.md`
(spec §9 now summarises it and defers to it on training). Content: what is trained vs frozen
(LoRA r=16 on q/k/v/o of all 28 layers; the arm's modules in the 4 patched layers only; fp32
masters); the data pool (RULER training seeds 100–102 over the sweep grid at 8K, MuSiQue train,
a 20K HotpotQA subsample; 1:1:1; 56K sequences, no repeats), sequence construction (B = 1, own
length, no padding/packing, drop-never-truncate, EOS after the answer), a seed-determined order
carried in the checkpoint; the loss (answer-token CE, **token-mean over the 16-sequence window**,
no auxiliary terms); optimiser and schedule (AdamW (0.9, 0.95); G1 LoRA 2e-4, G2 module weights
1e-3, G3 biases/b0/LN 1e-3 no decay; one cosine over 3,500 or 2,500 steps, 3 % warm-up, to 10 %,
not restarted at Phase B; clip 1.0; bf16 autocast with the normaliser in fp32; checkpointing on
the patched layers); Phase A once per seed and forked; the **Phase-B initialisation sequence**
(fresh modules; 8-sequence calibration batch not consumed; `b0` by bisection to ρ₀ = 5 % per
layer; `TauK` bias to the median `1/std_j`; `TauQ` bias to `τ_i = 1`; two sanity checks — loss
with `𝓔 = ∅` equals Phase A's to 1e-4, loss with the relation live within 5 % — and a STOP if
they fail; fresh Adam moments for G2/G3); the step in pseudo-code with the `ν` state passed
between patched layers un-detached; the E8 block every 50 steps (reported, never acted on); the
250-step eval and checkpoint; pre-committed failure handling (NaN → restart from checkpoint with
the same order; second NaN at the same step → halve G2/G3 LR once; third → the arm×seed is
reported failed); the per-arm table; and "what a healthy run looks like" so the first run can be
judged within an hour. Spec bumped to v4.2 (§9 aligned: B = 1 means no padding; reduced-programme
step count stated).


## §46. D-9 confirmed: `T_j` by coreference string match (Luke, 2026-09-07)

Luke confirmed the column ground-truth construction that §40 proposed and spec §12.3 carried as
a pre-registered default. Written into the paper as App. H "E7, the column ground truth": key =
first token of the first mention of a bridge string; `T_j` = first tokens of every later mention
plus the answer positions emitting its value; RULER MV-NIAH gives `m_j = 2m` (other needles,
question, `m` answer positions), MuSiQue's decomposition gives leaf `m_j = 2`, bridging 3–5;
gold passages placed in hop order (reversal changes `m_j` deterministically — a free check);
relation-recall misses kept separate from the interval (computed on `T_j ∩ 𝓔.ⱼ` against the
relation's complement; empty complement → `δ_j = +∞`, lower endpoint 0); S0's routing check on
B0 (mass > half the row max on ≥ 80 % of examples) licenses it, otherwise "not measurable on
this backbone"; dense arms reported on both domains, the relation domain the headline. App. E's
closing sentence now points to it. Spec v4.3; register D-9 closed; open items in spec §17 are
now 2–5 (grids, decode path, B5 mode, `transformers` pin).

## §47. INV-1 / INV-2 demoted (Luke, 2026-09-08)

Luke questioned two runtime invariants. **INV-1** (`Σ_i Ã_ij = c_j`): true, but a corollary of the
two statements the mechanism actually makes — the relation's quota is carried exactly
(`Σ_{i∈𝓔.ⱼ} Ã_ij = \bar c_j`) and the complement is untouched (INV-4). The spec now makes the
quota identity INV-1 and keeps the column total as INV-1', a cheap check for writes outside the
relation. **INV-2** (`Σ_j c_j = n_q`): an identity of the row-softmax (`Σ_iΣ_j A^{sm}_ij = Σ_i 1`),
true iff every real row is a normalised distribution; it says nothing about MarSea and was listed
only because the earlier design *imposed* it as the budget anchor. Moved to a backbone/mask
sanity group as SAN-1 ("every real row of `A^{sm}` sums to 1; pad rows are 0"). Spec v4.4.

**Paper (offered, not yet applied):** remove `prop:twostep`(ii) and the "derived, not imposed"
anchor sentences (Sec. 2.1, transfer table); state (i) as quota conservation with the column
total as a remark. Luke to say the word.

## §48. Runtime invariants re-audited item by item (Luke, 2026-09-08)

Each of the fourteen entries was tested for truth and necessity. **Kept (renumbered 1–9):**
quota carried exactly on `𝓔.ⱼ` (1); Stage-1 complement equals `A^{sm}` exactly (2, previously
unstated — `prop:twostep`(iv)'s column half); `A ≥ 0`, rows `≤ 1` (3); Stage-2 complement equals
`a^{(1)}` (4); the unit cap is the identity on a sub-unit row (5, new — "not a second softmax"
made checkable); zeros permanent (6, absorbing `supp ⊆` and `θ_i ≥ 0`, which are the same
statement); row quota a ceiling (7); masking total (8); empty relation = `A^{sm}` (9).
**Removed:** column total (corollary of 1 + 2); `Σ_j c_j = n_q` (row-softmax identity → SAN-1);
"one relation object" (rule of construction, §4); `supp ⊆` and `θ ≥ 0` (= 6); `ψ_j` monotone
(decode-time, sealed `τ_j` → T7); `τ_j → 0` uniform (a limit → T12/H9); fan-in target rule
(construction → T14). Added T16 (`τ_i = 1` identity). Harness gained three rows — Stage-1
complement exact (0/14,937), cap identity on sub-unit rows (0/80,741), `τ_i = 1` reproduces
`a^{(1)}` exactly incl. cap-binding rows (0/80,741) — **1,058,450 checks / 0 / 62 rows**; both
verification tables carry them; the Sec. 2.2 sentence "the harness confirms this on every
cap-binding row" is now literally true. Spec v4.5.


## §49. Spec §17 closed — D-27 to D-30 (Luke, 2026-09-08)

Luke confirmed the four remaining implementation decisions, noting he did so on an incomplete
grasp of the details; for the record, what each commits to:
- **D-27, grids.** E2: 8 keys × `num_needle_v = m ∈ {1,2,4,8,16}` at L = 8K (≤ 128 needle
  sentences ≈ 2K tokens). E3: `num_needle_k = n ∈ {8..128}`, 4 values each, at L = 16K for *every*
  n (the inert-padding/length control requires constant L; n = 128 is ≈ 8K tokens of needles).
  Consequence: the S2 stage's E3 evaluation runs at 16K, which the dense reference path cannot
  hold on 80 GB with 4 patched layers — the chunked implementation (§6.6, T11) is on the critical
  path before S2, not optional.
- **D-28, decode.** Frozen-prefix: past rows' outputs stand; the new row's fan-out entry is a
  per-key re-solve on the grown relation at the sealed `τ_j` with an incrementally updated quota.
  Consequence: exact-set accuracy (the headline task number) is measured under an over-admitting
  policy (`prop:prefix`(iii)); its per-step cost is `O(|𝓔.ⱼ| log|𝓔.ⱼ|)` per admitted key and is
  reported beside the number.
- **D-29, B5.** Teacher-forced only, "for now": B5 appears in the fidelity tables, not the task
  tables. Revisitable after S2.
- **D-30, pin.** `transformers >= 4.53`, exact release pinned in `requirements.txt`, signature
  asserted at import (the boolean-mask era; `vis` is built inside the patched layer regardless).
Spec v4.6: §17 records the closures; no implementation decision remains open. Register updated
(D-27–D-30; the open list is reduced to the paper items).

## §50. Implementation delivered; preliminary S0 probe; routing check refined before use (2026-09-08)

**Delivered (Claude Code, `/home/luke/Dev/marsea`, ~6,100 lines):** every module of spec v4.6,
T0–T16 ported to the tensor path (50 tests pass with invariants on; T0 bit-identical to the
unpatched eager backbone; T11 exact to 1e-9 in fp64 incl. gradients); harness reproduces 0
violations; E1 table reproduced. Detector on Qwen2.5-1.5B: `PATCHED_LAYERS = [14, 19, 22, 23]`,
`(l*, h*) = (19, 3)`. A full Phase A → B dry run (2 layers, 1K) completed on an RTX 5060 Ti.

**Preliminary S0 at 2K on the UNTRAINED base model:** no construction passed the 80 % gate;
C-a (value-key) reached 50 % at the retrieval head; the D-9 coreference construction was not
routed at any head of layer 19. Assessment: not a deal-breaker. (1) The detector selects
retrieval heads (answer position → needle value = construction C-a); coreference is the induction
pattern and lives in other layers — the probe never looked where it would be. (2) Induction heads
attend to the token AFTER the antecedent; a key pinned to the first mention's first token can miss
by one position. (3) C-a's 50 % is depressed by the untrained base model, the 2K context, and the
attention sink being the row's argmax. **Worst case** (no head anywhere routes coreference): the
pre-registered fallback in App. H — E7's column half "not measurable on this backbone", the
interval-hit rate on synthetic columns/harness only, E7 row-side. E3 (load-bearing), Thms. 1–2,
Prop. 14, Fig. 2, and the B2/B3 tests of contribution (i) are untouched.

**Refinement recorded before the real S0 run (spec §12.3 v4.7):** sweep all layers × heads;
span-tolerant matching (key = first mention + 1 token; query = any token of the later mention);
sink excluded from the row maximum; run on Phase-A weights at 8K; if a coreference-routing head
exists, add an induction score to the detector and its layer to `PATCHED_LAYERS`. These change a
measurement no result has yet been drawn from, and are logged here for that reason.

## §51. Code review of the delivered implementation (2026-09-09)

Three passes over the 26 uploaded files (numerical audit of the core in fp64 against the harness;
training loop vs the procedure; evaluation/S0/data vs spec §§6.7, 8, 12, 13). Full report:
`code_review_2026-09-09.md` (project `claude/marsea_code_review_2026-09-09.md`). The mechanism's
forward pass is right (harness match 1e-15; all invariants; decode and chunked forwards match
dense). **Ten blocking items** would have made S2 wrong: `run_s2.sh` trains a zero-step Phase A
and every arm forks it; E9 arms train their own Phase A; B4/B5 evaluation crashes; E3's chunked
path crashes or falls back to attention values as scores; vacuous interval hits on `|E_.j| < 2`
columns inflate the hit rate; `δ ≤ 0` columns scored as misses; the dense arms' headline
relation-domain precision is never produced; E3 cannot be stratified by n; the S0 gate is wired to
the pre-v4.7 check and the replication-pair statistic is unimplemented; and `torch.clamp` in
`proj_le` has zero subgradient at exactly-zero entries, cutting the ST gradient path (2)→(3) on
the dense path (chunked-vs-dense gradient mismatch of 0.39 on `U`; fix `where(v >= 0, v, 0)`).
Fourteen further correctness/bookkeeping items and four version risks listed. `marsea/data/*.py`
were not uploaded and the `T_j` annotator, grids and seeds remain unreviewed.

**Design finding for Luke (D-31 proposed):** rows can end with zero mass when a query's only
visible key re-shapes its quota away from it; in a BOS-less decoder that is the sink token at
position 0, whose head output at the patched layers becomes exactly zero. Proposed rule: exclude
position 0 from every relation (`E[:,0] = E[0,:] = False`), consistent with "the sink is not a
candidate key" in the S0 refinement.

## §52. Corrected S0 at 8K; D-31 adopted; the `T_j` split proposed (2026-09-09)

**S0 re-run (Claude Code, RTX 5060 Ti, base model at 8K, 40 MV-NIAH examples, m ∈ {1, 4}; the
v4.7 measurement — every head swept, span-tolerant matching, sink excluded).** The measurement
was the problem, not the construction. C-a clears the gate at 60 of 336 heads and reads 0.99 at
the detector's (19, 3), where the first probe read 0.50 against the sink. Coreference (later
mention → first mention + one token) is routed cleanly by 21 heads, best (14, 5) and (14, 3) at
1.00; layer 14 is already in `PATCHED_LAYERS`, so no layer joins; induction scores are in
`runs/detector.json`. A reduced Phase-A stand-in (100 steps × 8 sequences, RULER + HotpotQA,
loss 0.78 → 0.07) *sharpens* routing rather than moving it (heads at ≥ 0.8: 21 → 69 coreference,
60 → 72 C-a); 21 heads route coreference on both base and adapted weights; among those that also
route C-a, (14, 3) has the highest retrieval score (0.72). **Finding:** the answer positions do
not attend to the key phrase's first mention (2 of 336 heads); they route to the *value* tokens.

**Consequence for D-9 (proposed, awaiting Luke):** split `T_j`. The coreference column's
`T_j` = the later mentions of the same string (other needles, the question mention, and the
answer-prefix mention if present) — `m_j = m` on MV-NIAH, or `m + 1` with the prefix; the answer
positions belong to the value-key columns (C-a, `m_j` = number of answer tokens emitting that
value, typically 1). The column test (E7 hit rate, E2 stratified by m) runs on the coreference
columns at (14, 3); the row test stays at the detector's `(l*, h*)`. The paper's App. H sentence
"together with the answer positions that emit its value" must be revised accordingly, and
`m_j = 2m` becomes `m_j = m (+1)`. The full gate (200 examples, m up to 16, MuSiQue, true
Phase-A weights) still runs on Nebius before S1.

**D-31 adopted (Luke):** position 0 excluded from every relation (`E[:,0] = E[0,:] = False`,
mask on the logits; `b0` calibration excludes those pairs). Paper Sec. 2.3 sentence added; spec
v4.8 §4.1; register D-31. The E8 block also logs the fraction of rows ending with zero mass.


## §53. D-9a: the `T_j` split, `m_j = m + 1` (Luke, 2026-09-10)

Claude Code reported all review items fixed (52 CPU tests + 4 GPU tests; Phase A protected by
`--phase_a_only`; E9 arms fork the shared checkpoint; B4/B5 stageless scoring; chunked path honours
`keep_dense_layers` with fp32 scores recomputed from q, k; hit defined only where Thm. 2 speaks;
paired-relation domain via `--paired_marsea_ckpt`; S0 gate = the sweep with `M(a)` and
`column_measurable`; right-derivative at exact zeros; D-31 as a chunk- and cache-aware mask; sanity
5(a) exact). B11 was left to Luke, who first chose `2m + 1`; I pointed out that `2m + 1` is the
UNSPLIT count and that the S0 sweep found answer positions do not route to the first mention (2 of
336 heads), so an unsplit `T_j` would score the coreference column against a target the backbone
never routes to it. **Luke adopted the split: coreference column `T_j` = later mentions only
(`m − 1` other needles + question + answer prefix), `m_j = m + 1`; answer positions belong to the
value columns (`m_j` typically 1, the row-side regime of §40).** Paper App. H E7 paragraph
rewritten (two column kinds; why they are not merged; S0 described as the full-head sweep with the
sink excluded; the column test runs at the head that routes it, in a patched layer). Spec v4.9
§12.3; register D-9a.

---

# Part 2 — 2026-09-11 to 2026-09-15 (§54 onward)

> ⭐ **Merged into this file on 2026-09-15.** §§54–59 lived in
> `claude/marsea_master_record_part2.md` because this record comes back from `project_read`
> **inline**, so amending it meant re-emitting the whole document (the §19 hazard). That
> workaround is retired: with Luke's repo folder connected, the record is a file on disk and can
> be edited surgically. **`marsea_master_record_part2.md` is superseded — do not write to it.**
> §§1–49 are unchanged from the 2026-09-08 copy; §§50–53 were forward-ported from the project
> copy and verified byte-identical against it.

**Period covered: 2026-09-11 to 2026-09-15.** Six events: the implementation read-through; the
memory budget; ⚠️⚠️ an uncommitted-work-loss incident that cost three review rounds; two clean
review rounds; the round that surfaced the unit-cap problem; and ⭐ **§59, the unit cap solved in
general** — four commits on one threshold, ending in a result the paper should state.

⚠️ **Deadline: abstract 18 Sep 2026, paper 25 Sep 2026 AoE. As of 2026-09-15 no experiment beyond
E1 has been run.** The mechanism is settled; three one-line blockers stand between the tree and a
launch (§59.6), and the calendar, not correctness, is now the binding constraint.

---

## §54. Read-through of the delivered implementation (2026-09-11)

**Occasion.** Luke directed a cold read of eight documents in order — the implementation spec, the
paper, the training procedure, `verify_all.py`, the round-2 pseudocode, the reference
implementation, the decisions register and the master record — and then of the implementation
itself, so that the reviewer held every design decision before looking at any code.

**Output:** `claude/marsea_readthrough_2026-09-11.md`, findings **F-1 … F-8**. They are stated
there and are not restated here; the ones that recur in later rounds are **F-8** (keep only the
diagnostics a measurement needs — the origin of the per-head slicing that §58.3 is still repairing)
and **B-5** (`teacher_forced_pass` slices the relation to `h*`).

⚠️ **Process note that mattered more than any single finding.** Luke corrected mid-session, twice,
that the code in the project folder was first stale and then refreshed. **A read of code whose
provenance is not established is worth nothing**, and §56 is the full-price version of that lesson.

---

## §55. Memory budget — does 8K training fit one H100? (2026-09-12)

**Occasion.** Claude Code reported three profiling fixes and observed that 8K training "doesn't
seem to fit H100's 80 GB". Luke asked whether the three defects were genuinely fixed and, by
independent calculation, whether a single H100 holds the training model.

**Output:** `claude/marsea_memory_budget_2026-09-12.md`. The arithmetic that carries it:

| one `[1, 12, T, T]` fp32 tensor | 4K | 8K | 16K |
|---|---|---|---|
| | **0.81 GB** | **3.22 GB** | **12.9 GB** |

⭐ **The decisive structural fact: with gradient checkpointing on the patched layers, the peak is
set by ONE layer, not by four.** The naive reading — four patched layers each holding a full set of
`[1,12,T,T]` diagnostics — is what made 8K look impossible; under recompute only the layer
currently being recomputed materialises them. The budget document works the live tensor count per
layer and shows the headroom; the fixes that followed (head blocking, keeping diagnostics only for
the measured head, the sparse relation store) were about holding that peak, not about the total.

⚠️ **The document later did double duty as the rebuild spec** after the work loss of §56, which is
the only reason the memory work survived that incident.

**Still not measured on an H100.** Every number in it is arithmetic plus a CPU profile.
`preflight.sh` is what turns it into a measurement, and it has not been run (§58.7 item 7).

---

## §56. ⚠️⚠️ The uncommitted-work-loss incident (2026-09-12 to 2026-09-13)

**This is the §19 incident's successor and it was far more expensive. Read it before trusting any
code drop.**

### 56.1 What happened

Luke relayed that the engineer had fixed the review's findings and uploaded the updated code. Three
reviewers with different mandates read it independently and **all three found that none of the
eight claimed fixes was present**. My first diagnosis was a transport or flattening problem —
there were two `__init__.py` entries at one project path, which looked like a directory structure
collapsing on upload.

⚠️ **Luke corrected it, and he was right: "most of the files you listed as missing aren't in the
real repo either."** The revised diagnosis was that the repository had gone *backwards* — that the
work existed only in an uncommitted working tree and had been wiped. The engineer subsequently
confirmed exactly that: **nothing had ever been committed, and the folder was not a git repository
at all.** They then created the repo and recovered most of it, including all source.

### 56.2 ⭐ What made the diagnosis defensible rather than an accusation

The tell was a *forward* one, not a missing-file one. The engineer described fixing an `akw`
`NameError`. That error **could not exist in the tree I had been given**, because it is introduced
by wiring `head_block` — and `head_block` was absent from that tree. So their working tree was
demonstrably *downstream* of mine; the evidence pointed at loss, not at work never done.

⚠️ **Standing lesson: when a report and a tree disagree, look for a defect the report describes
that the tree cannot produce.** A missing fix is ambiguous between "not done" and "not delivered";
a missing *precondition for a described bug* is not.

### 56.3 ⭐⭐ The provenance protocol — now automated

Adopted, and it has held for three rounds since:

1. `git rev-parse HEAD` — the commit under review.
2. `git status --porcelain` — **must be empty**. An uncommitted tree is not reviewable.
3. `sha256sum` over every `*.py` and `*.sh` under `marsea/`, `scripts/`, `tests/`, as a manifest.
4. The reviewer verifies all three **before reading a line**.

The engineer automated it as `scripts/handoff.sh`, which writes `HANDOFF.txt` (hash, porcelain
status, manifest) and tars the tree beside it. **This protocol would have caught the rollback in
thirty seconds instead of three full review rounds**, and it now catches nothing — which is the
point.

### 56.4 ⭐ Two method changes this forced, both kept

- **Diff-based review.** Once two verified trees are on disk, the review is a diff, not a re-read.
  It is far more rigorous *and* far cheaper: ~2,000 lines at e982f83 against ~6,100 lines of source.
- **Three readers, different mandates** (record §27's rule, now standard practice here): fix
  verification and regression hunting; adversarial numerics and mechanism fidelity; the experiment
  programme end to end. Of the findings in §58 the three passes overlapped on **two**, which is the
  same low-overlap figure §27 recorded and the same argument for running more than one.
- **Reviewers execute.** From 30372ae onward torch has been available in the review container, so
  findings are *measured* rather than inferred. Every number in §58 was produced by running the
  real code.

### 56.5 ⚠️ A record-keeping loss, and the rule that follows

Luke also lost the review documents I had produced, having deleted the project copies. They were
recoverable only because the same content had been delivered as files. **Every review document
from here on is delivered three ways: as a downloadable file, as a project doc, and (where it
carries a decision) as a record section.** A document that exists in one place does not exist.

---

## §57. Two clean review rounds: d5bd980 and 30372ae (2026-09-13)

**d5bd980** — the first properly-verified tree, reviewed in `claude/marsea_review_d5bd980.md`.

**30372ae** — reviewed in `claude/marsea_review_30372ae.md`. Provenance verified 52/52. Executed:
`pytest` **90 passed / 12 skipped**; `verify_all.py` **0 violations over 62 counted checks**. The
12 skips are `transformers`/GPU-gated.

⭐ **Two items worth carrying forward.**

1. **The engineer improved on a review recommendation.** I had asked for
   `assert not (head_block and per_head)` — a guard. They instead threaded `head_offset` through
   both paths so the two **compose**, verified exhaustively over
   `Hkv∈{2,3} × per_head∈{0,H} × key_only∈{0,1} × hb∈{1..5}`: **max |ΔO| 8.9e-16, max Δparam-grad
   1.5e-11, no missing gradients.** That is a real fix, not a guard, and it is the pattern to
   prefer.
2. **A reviewer disagreement I adjudicated.** On `chunked.py`'s `acc.clamp_min(1e-300)`, one reader
   called it a live fp32 NaN and the other called it safe. The first was right: in fp32 `1e-300`
   rounds to `0.0`, so `x >= min` holds and `log`'s backward computes `0.0/0.0`. It was unreachable
   at B=1 with no padding, but the engineer fixed it with `torch.finfo(dt).tiny` and a reviewer
   measured the old tree producing **96 NaNs in `dQ`** against zero in the new one.

**§B of that review — three findings that affect results** — was expanded at Luke's request into
`claude/marsea_B_findings_detail.md`: B-1 (coreference site collapse), B-2 (`truncation_events`
divergence), B-3 (`frac_rows_zero_mass` reconstruction). ⭐ **Their common class is the lesson:**
none is a mechanism error. All three are **instrumentation** errors, where a quantity is computed
one way on the dense path and re-derived differently on the sparse path, with no test able to
distinguish them. The structural defence proposed there — one parity harness asserting every E8
key, every cache field and every column record across both paths, on a configuration chosen to
exercise the rare branches — **has still not been built**, and §58 found two more members of the
same class.

---

## §58. e982f83 — `cap_slack`, four blockers, and a paper claim that is now false (2026-09-14)

Full review: `claude/marsea_review_e982f83.md`. Provenance verified 52/52, tree clean. Executed:
`pytest` **97 passed / 13 skipped** (110 is the *collected* count, not the passing one);
`MARSEA_DEBUG=1` the same; `verify_all.py` **0 violations over 1,058,450 evaluations in 63 checks**.

⭐ **A reviewer independently diffed every `verify_all` denominator against `tab:verification` and
`tab:transferchecks` — 24000, 12000, 10098, 10604, 17928, 80741, 131180, 12953, 14937, 12082,
20472, 4506, 1992, 7200, 9000 — and all match; the paper's `0/24` ceiling row is the harness's two
rows of 12 summed. Those two tables are reproducible as printed.** Worth stating plainly: it is the
part of the paper a reviewer is most likely to check by hand, and §48's count survived the whole
implementation.

### 58.1 ⭐⭐ `cap_slack` — the best finding of the month, and it was the engineer's

They changed the unit cap's binding slack from the fixed `CAP_TOL = 1e-6` to
`cap_slack(n_k, dtype) = max(1e-6, 8·eps·√n_k)`, threaded into `proj_le_masked`'s `bind_tol` on the
dense, chunked and decode paths plus `cap_binds` and INV-3/INV-5. They found it **themselves, from
a failing 2K smoke test, after the review had moved on**. That is the behaviour to want.

**The symptom, reproduced on the old tree.** A real fp32 softmax row of 1938 entries sums to
`1 + 9.7e-7`, so from about T = 2000 the unit cap binds on rows nothing pushed over one unit. Three
runs of the same smoke gave `gap_5a` 4.6e-05, then 4.2e-04, then 0.0.

⚠️ **The nondeterminism was never floating point.** Thread count is irrelevant — `torch.softmax`
plus the row sum are bit-identical at 1/2/4 threads. At T≈2K only ~0.1 % of rows exceed `1+1e-6`,
so whether *any* row in a calibration batch crosses is a coin flip on seed and data:

| n_k | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|
| frac rows > 1+1e-6 | 0.10 % | 0.59 % | 8.2 % | **44.2 %** |

`CAP_TOL = 1e-6` is a **bulk-mode failure at long context**, not a tail event.

⭐⭐ **Why a 1e-6 threshold error is not a 1e-6 result error — record this.** `proj_eq_masked` is
`s·sparsemax(v/s)`, a **sparsemax, not a rescale**. On a marginally-over peaked row at 16K:

```
theta = 4.657e-10 | max|a1-A|_inf = 4.657e-10 | L1 = 3.189e-06 | entries ZEROED = 10038 of 16384 (61 %)
```

The mass change is 3e-6; **61 % of the row's entries are set to exact zero**, which also corrupts
`supp_rel`, `kstar` and every `A > 0` diagnostic on that row.

### 58.2 ⚠️ But the stated justification is wrong and the constant is ~25× too loose

The docstring attributes the growth to the fp error of a sum of `n_k` terms. **The row-sum
reduction's error is flat in `n_k`** — ATen uses a pairwise/cascade sum (measured 14.6× more
accurate than strict sequential fp32 at n=1024), so it is ~2e-7 everywhere. The growth is entirely
in the softmax **values**, and it is a **kernel artifact**: torch's fused softmax is ~80× more
biased than `exp(S−max)/sum`, and the error peaks at row entropy ≈ 2–3 nats, falling at both
near-uniform and near-one-hot rows. It scales with the score distribution's *shape*, not with
`√n_k`.

Worst-case search over logit scale and three score shapes, ~2M rows per length:

| n_k | worst measured row-sum error | `8·eps·√n_k` | margin | `8·eps·log₂ n_k` |
|---|---|---|---|---|
| 2048 | 1.48e-6 | 4.32e-5 | 29× | 1.05e-5 (7×) |
| 8192 | 3.35e-6 | 8.63e-5 | 26× | 1.24e-5 (3.7×) |
| 16384 | 5.75e-6 | **1.221e-4** | 21× | 1.34e-5 (2.3×) |

⭐ **Recommendation: `8·eps·log₂ n_k`.** It matches the measured scaling, still clears the worst
case by 2–7×, and shrinks the number the paper has to print by ~10×.

There is a second over-slack: `vis` is causal, so row *i* sums *i+1* terms, yet every row is
granted `cap_slack(n_k)` for the full length. At 16K, row 0 — one visible key, exactly zero
row-sum error — is granted 1.2e-4 of slack, **122× its own visible count's worth**.

### 58.3 ⚠️⚠️ INV-3 is now violated as a *mathematical* claim, and Prop. A(iii) must change

The docstring says "a row a column relation genuinely pushed over a unit exceeds it by O(0.1), so a
slack of 1e-4 costs the mechanism nothing." **True of the median, false of the distribution.** The
excess is `Σ_E(c̄_j p) − Σ_E A_sm` — continuous, and arbitrarily small whenever the relation barely
perturbs the row. Measured on the real `MarSeaNormalizer` at n=2048, ρ=0.05: median excess 3.6e-2,
p10 3.8e-3, **min 1.19e-07**.

Rows with an unambiguously genuine excess (above the 6e-6 fp noise floor) now left uncapped:

| | ρ=0.002 | ρ=0.01 | **ρ₀=0.05 (deployed)** | ρ=0.10 |
|---|---|---|---|---|
| **L=8192** | 10.2 % | 4.69 % | **0.78 %** | 0.59 % |
| **L=16384** | 9.38 % | 7.42 % | **1.17 %** | 0.59 % |

The bound is tight — Stage 2 can only reduce the row sum — so the worst case is exactly
`cap_slack(n_k)`:

> **L=8192: `Σ_j A[i,:] ≤ 1 + 8.63e-5`, realised on ~0.8 % of rows at ρ₀ = 0.05**
> **L=16384: `Σ_j A[i,:] ≤ 1 + 1.221e-4`, realised on ~1.2 % of rows at ρ₀ = 0.05**

⚠️⚠️ **`prop:twostep`(iii) / INV-3 can no longer read "never exceeds one unit."** It must read
`Σ_j A_ij ≤ 1 + δ(n_k)`, with `δ` named, its values given for the lengths reported, and the
~1 %-of-rows figure stated. **This is a paper edit, not only a code one** (§58.6), and it is the
single most falsifiable sentence in the draft: a reviewer can refute the absolute wording in ten
lines of code, in a paper whose entire pitch is that its invariants are checked.

⚠️ **E8 comparability breaks exactly where E8 matters.** `frac_cap_binds` and `frac_rows_over_unit`
both read `cap_binds`. Two reviewers appeared to disagree here; they measured different coverage
regimes and both are right:

| | ρ=0.15 | ρ=0.05 | ρ=0.02 | ρ=0.002 |
|---|---|---|---|---|
| n=8192 | 0.0 pp | −0.8 pp | −2.7 pp | **0.3211 → 0.1671 (−48 %)** |
| n=16384 | 0.0 pp | −0.6 pp | −2.9 pp | **0.3491 → 0.1446 (−59 %)** |

Comparable at high coverage; **not comparable at all in the low-coverage regime E8 exists to
detect.** Any E8 number collected before this commit must be re-collected, and the run README must
record the `cap_slack` in force.

⚠️ **Gradients:** the branch discontinuity is unchanged in kind (grad cos-sim 0.499972 either side,
old and new alike — it is the identity → mean-centred-sparsemax Jacobian switch, a property of the
branch). The **forward** jump in row mass is exactly `bind_tol`, so it grows 122×, and the plateau
where the cap exerts no restoring gradient widens by the same factor. End to end this is
negligible (at n=4096, ρ=0.01: 6/256 rows change branch, `‖ΔA‖/‖A‖ = 3.3e-7`,
`‖Δgrad‖/‖grad‖ = 7.7e-5`; at ρ ≥ 0.05, none) — but note the **230× forward→gradient
amplification**, and that runs across this commit are not bit-comparable.

⚠️ **fp64 blindness.** `cap_slack(n, float64)` returns the 1e-6 floor until n ≈ 3.2e17, so every
bit-exactness test is blind to this change; a fp64 test passes for *any* `bind_tol`, so the four
call sites could each carry a different constant undetected. The one fp32 test added is
one-directional. **Nothing tests that the slack is small enough** — a `cap_slack` of 1e-2 passes
everything except a two-decade bracket. The test owed has two halves: sweep *logit entropy* (the
error peaks at 2–3 nats; one `randn*2` draw misses it) at n_k ∈ {8192, 16384} asserting
`cap_binds.sum() == 0` under `force_empty` on all three paths; and the converse — a row with a
genuine excess just above the noise floor must **still** bind.

⚠️ **Re-measure on the GPU.** The slack is now calibrated against an **ATen CPU softmax kernel
artifact**; the CUDA kernel is a different implementation. Re-measure on the H100 the paper's
numbers come from, and say in the appendix which device the measurement is for.

### 58.4 B-1 is fixed where it was found and survives in three other places

The core refactor is right and was verified: `site_map` builds `{layer: [heads]}`,
`keep_dense_head` carries a scalar for one head and a list for several, and every per-head field of
a column is indexed by the same `(dj, hj)` pair that `c["site"]` records. Measured with
`dense_head=[3,7]`: all 17 per-head tensors agree with a full-H reference to **0.0 in fp64**, and
the single-head path is bit-identical to before.

⚠️ **Three residuals, all the same class:**

1. **`chunked.py:183-193` overwrites instead of concatenating** when the two kept heads fall in
   different head blocks, while stamping `head_sub` with the full list. With `HEAD_BLOCK=2` (the
   queue's setting) and `dense_head=[3,5]`, `_kept_head(d,3)` returns index 0 into a tensor holding
   only head 5, and `_kept_head(d,5)` raises `IndexError`. Reachable on the default queue, and
   `MARSEA_DEBUG=1` cannot catch it — `check_invariants` runs per block, before the merge. The same
   range double-remaps `extra["sparse"]`'s head indices.
2. **`run_eval.py:98` is still a layer-keyed dict comprehension** — three lines below a comment
   citing the B-1 review. With two sites in one layer the second wins, and the coreference
   relation is handed to the value columns, corrupting **the A7 headline precision of every dense
   baseline** on E2/E3/E3pad/E3depth/E4.
3. **`_kept_head` returns 0** when the head was not kept, instead of raising — a silent wrong
   answer in exactly B-1's failure class, in the function written to prevent it.

Plus a latent one: narrowing `extra["sparse"]` to the kept heads starves `init_from_sparse`
(measured `[137,0,0,0,0,0,0,0]` against `[145,130,155,137,136,152,127,115]`). Not live only because
`teacher_forced_pass` clears the site set before `generate_greedy`; one line moving and the
frozen-prefix decode (D-28 — every task-level number) silently loses 7/8 of its relation.

**B-2** parity is fixed and verified (six matched dense/chunked configurations agree exactly,
including under head blocking; the new non-zero assertion is non-vacuous). ⚠️ **The two-counter
recommendation was not taken**: one `truncation_events` still sums `counts > K` (an eviction, about
cache lossiness) with `kstar ≥ 0.9K` (a near-truncation warning, about hierarchy exactness).
Fixing the parity without splitting the counter *entrenches* the conflation, because both paths now
agree on a number that means two things.

**B-3** is fixed well: `4·eps·√n_k` clears the reconstruction's measured residual by 26–52×, kills
the CUDA `index_add` nondeterminism, and both the tolerance and the ambiguous band reach the
records. ⚠️ The dense path still uses exact `A.sum(-1) == 0` while the sparse path uses the
tolerance, so the same E8 key still means two things depending on the experiment, and no parity
test makes the *mechanism* produce a zero-mass row.

⚠️ **`invariants.py:168-172`'s new sparse INV-3 is a tautology.** `cap_binds` is defined from the
same `rowsum` the check tests, with `tol3 ≥ slack`, so it holds by construction; doubling `a1`/`A`
on the relation leaves INV-3 and INV-3-recon green while INV-5 and INV-7 go red — the identical
outcome to the version it replaced. The honest split is the right idea; the salvaged half needs an
independent quantity or belongs in `_not_checked` with the other.

### 58.5 ⚠️⚠️ The queue cannot produce the paper's tables

Four defects, none about the mechanism, all fatal to the run.

1. **`run_evalsuite.sh:56` launches 56 concurrent jobs on 8 GPUs.** The barrier is tested once per
   *arm*, and this round's E3depth fix made it **seven** launches per arm. 7 is coprime with 8, so
   `i % 8 == 0` is true exactly once in the entire double loop — first barrier at 56 in flight.
   ⭐ **A regression this round's own fix introduced**: at six jobs the cadence hit 24.
   `run_s2.sh` already has the right pattern (barrier inside `launch()`); it was never ported.
2. **`run_eval.py:118` stems output on `{arm}_{kind}_{experiment}`** — no seed, no set. Nine
   E3-family jobs per arm collapse to one parquet; **all seven E9 ablations collapse to one
   table**. The `.done` markers carry the seed, so a restart marks everything complete. Discovered
   only after ~500 GPU-hours, with nothing on disk to recover.
3. **The `paired_E` host cache is sized at ~268 GB per process.** `run_evalsuite.sh` runs
   E3/E3pad/E3depth with no `--n`, and the glob resolves to exactly n = 1000 at 16K — the figure
   the code's own comment warns about.
4. **A pre-registered E9 arm sits behind `|| echo`.** `preflight.sh:40-43` profiles dense training
   at 2048/4096 and calls it "NOT a gate … the queue uses chunked anyway"; `run_s2.sh:105-116`
   trains three arms `--mode dense --L 8192`, one of them `e9_uniform_quota`. Two files in the repo
   disagree about the same 27 GB and nothing gates it.

⚠️ **Cost.** The queue as written implies **≈1,000 H100-h against `tab:compute`'s ≈336** — a 3×
overrun, ~$2,500–3,300 against D-25's budgeted $1,500–1,950, and ~6.5 days of flawless execution
against 11 days to the deadline. Two cuts fix both: **E3depth to `marsea` + `B0` seed 0 only** (a
*control* currently costing as much as the load-bearing experiment, six times over; saves ≈150 h)
and **`--n 400` on the three 16K jobs** (saves ≈250 h, and blocker 3 forces it anyway). That lands
at ≈610 h and ~4.5 days, and brings `tab:compute` back inside its own band.

⚠️ **Nothing aggregates across seeds and nothing builds the E7/E8 tables.** `rec["e8"]` is written
per row and `aggregate_ruler` never reads it, so the `|E_.j|`/`|E_i.|` distributions, realised
`k*`, `k*` by `|E_.j|`, the `τ_i R̃_i / c̄_i` distribution and the cap-binding fraction are in the
parquet and in no table. E3's coverage rate `|E_.j|/n` is computed as `E_size` and dropped. A
`scripts/collect.py` is owed before §5 can be written.

### 58.6 ⚠️ Paper claims that nothing in the queue produces — Luke's call

Each is a sentence promising a measurement. A reviewer who checks three of them stops believing the
rest. **None of these is edited unilaterally** (record §9 item 10, D-2), but the list is now exact:

| Claim | Status |
|---|---|
| **Llama-3.2-3B cross-family check** (Setup; D-22) | no arm, no backbone loop anywhere |
| **"We ablate both emission policies"** (Sec. 2.6) | no arm, no flag; `frozen_prefix` is chosen by arm type, never swept |
| **"a hard variant preserves them … we ablate both"** (App. F) | nothing |
| **"E9 ablates the tying of τᴷ to τQ"** (App. F) | nothing |
| **E9 over layer set and block size** (App. H) | no arm (`r`, `no_nu`, `tau_j_global` exist behind `EXTENDED_E9=0`, consistent with D-24) |
| **E6 on IHEval** — the ranked half of the motivating scene | no acquisition step; the queue prints "E6 skipped"; per-task checkers not vendored |
| **E7's measured PR trace against Fig. 1** | no τ_j sweep on a trained checkpoint exists; the only sweep rows are synthetic numpy that App. D says are "not evidence about real data" |
| **Efficiency vs measured relation coverage** (App. H) | no FLOPs, no parameter count, no coverage-indexed table |
| **Padding-length invariance** (App. H) | no such check |
| **E3's padding control "at every point"** | one matched point (`ruler.py:67`, K=1); reword or extend the grid |
| **B1/B4 in a headline table with a std column** | D-24 gives them one seed; needs a footnote |
| **"moderate context (2–16K)"** | only 8K and 16K exist |
| **E5's dense-arm headline precision on the paired relation** | `run_evalsuite.sh` passes no `$paired` for MuSiQue/Hotpot — one-line code fix, not a paper edit |

⭐ **Plus the one that is a correctness fix rather than a framing choice: `prop:twostep`(iii) /
INV-3** (§58.3).

### 58.7 The critical path, as of 2026-09-14

**Before the queue launches** — all shell/argparse, ~1 hour:

1. `run_evalsuite.sh` — barrier into `run()`.
2. `run_eval.py` — `--tag` in the output stem; seed/arm/set/ckpt/git in every row.
3. `--n 400` on E3/E3pad/E3depth.
4. Gate the dense-8K training probe, or move the three dense E9 arms.
5. Cut E3depth to `marsea` + `B0` seed 0.
6. Hard-assert MuSiQue and Hotpot present before S2 (`mix.py` currently fails open, and the arms
   would silently train on the wrong mixture).
7. **Regenerate `runs/detector.json`; run `preflight.sh` on the H100** — the engineer's two
   standing items, still correct. Measure `cap_slack` on the H100 in the same pass.

**During training:** the three B-1 residuals and the sparse-store narrowing; `cap_slack` →
`8·eps·log₂ n_k` with the two-sided fp32 test and a `min(…, 1e-3)` ceiling; split
`truncation_events`; make the sparse INV-3 independent or declare it; `--head_block` on
`run_eval.py`/`run_e6.py` plus a gated dense teacher-forced 8K profile and `--sites 2` at 16K;
`scripts/collect.py`; `$paired` on E5; `--mode chunked` on the E9 eval loop; eval-side provenance;
an S0-licence gate on `run_evalsuite.sh`.

**Paper, independent of code:** §58.3's INV-3 restatement first; then §58.6's list.

### 58.8 ⭐ The standing lesson of §57 and §58 together

**Every defect that survived this round is an instrumentation defect, not a mechanism defect.** The
mechanism has been right since 30372ae — harness match, invariants, dense/chunked/decode agreement
at 1e-15. What keeps breaking is the *measuring apparatus*: a quantity computed one way on the
dense path and re-derived differently on the sparse path; a diagnostic sliced to one head and read
at another; a counter summing two incommensurable events; a queue that cannot name its own outputs.

⚠️ **And §22.2's lesson recurs for the fourth time.** The tests could not catch any of these,
because each test exercises the path that produced the number. `MARSEA_DEBUG=1` runs per block and
cannot see a merge bug; a fp64 bit-exactness test passes for any `bind_tol`; the sparse INV-3
checks a quantity against a threshold defined from that same quantity. **A verification that
assumes the thing under test cannot falsify it.** The parity harness proposed in §57 — one
configuration deliberately chosen to hit the rare branches (a zero-mass row, an over-`K_ret`
column, two sites in one layer, two head blocks), asserting every E8 key, every cache field and
every column record across both paths — remains the one structural defence that would have caught
all of them, and it still has not been built.

---

## §59. ⭐⭐ The unit cap, solved in general (2026-09-14 to 2026-09-15)

**Four commits — e16a843, 7814665, 9bacef9, and the round that precedes them — spent on one
threshold. The resolution is a result, not a workaround, and it belongs in the paper.** Reviews:
`claude/marsea_review_e16a843.md`, `_7814665.md`, `_9bacef9.md`. Paper patches:
`claude/marsea_paper_edits_cap.md`.

### 59.1 The problem, stated correctly at last

The unit cap of `eq:rowstep1` tested `Σ_j Atil_ij > 1 + slack`. **The fp32 row sum of a softmax is
not one**, and the error is a property of the **reduction kernel**, not of fp32 arithmetic:

> **δ_max(n_k) = n_k · eps / (2L)**, with `L` the reduction's lane count.

⭐ **The mechanism, measured and verified to within 1 % at every length from 2K to 32K.** ATen's CPU
softmax accumulates its denominator in `L` vector lanes; the lane holding the row max carries an
accumulator ≈ 1.0, so every term below half an ulp of 1.0 added into *that* lane rounds away.
Exactly `1/L` of the sub-half-ulp tail is lost, the denominator comes out too small, and the whole
row is scaled up coherently (`A_j·d/e_j` constant across `j` to seven digits — a denominator-only
error). The worst case is a **near-one-hot row with a long flat tail at gap ≈ 16.64 nats**, entropy
0.002–0.25 nats.

| kernel | δ at 16K |
|---|---|
| CUDA, pairwise | ~1e-6 |
| x86, 16 lanes | 6.0e-5 |
| x86, 8 lanes | 1.2e-4 |
| scalar (`L = 1`) | up to 1e-3 |

**No constant covers that range**, and the target machines (Nebius) have unknown CPUs. Three rounds
of constants were each right on exactly one device: `1e-6` (30372ae), `8·eps·√n_k` (e982f83),
`8·eps·log₂ n_k` (e16a843 — **my recommendation, and it was wrong**: my worst-case search peaked at
entropy 2–3 nats and never entered the near-one-hot regime, so I concluded the growth was
logarithmic when it is **linear in n_k**; headroom at the deployed constant was 0.41× at 8K and
0.22× at 16K).

### 59.2 ⭐ The resolution: change the tested quantity, not the constant

The engineer's move, and it is the right one. Since `Atil == A_sm` **bitwise** off the relation (the
straight-through gate is exactly `0.0` there), the row's excess over its own softmax mass is

```
excess_i = Σ_j Atil_ij − Σ_j A_sm_ij = Σ_{j∈E_i.} (Atil_ij − A_sm_ij)      exactly
```

— a difference of **in-relation terms that reads no row total**, so the kernel's error cannot enter.
The cap binds iff `excess_i > CAP_TOL = 1e-6`, and a binding row is projected onto `Σ_j A_sm_ij`
(the fp64 sum of the fp32 entries), not onto the literal 1.

**Verified independently, three ways.** (a) The error in `excess` is *relative*, not absolute: far
from the boundary it reaches 1.1e-4 (it inherits the kernel's coherent row scaling), but on rows
whose true excess is near zero it is **3–6e-9, i.e. 160–330× under CAP_TOL** — the perturbation is a
common factor on both terms of `cbar_j·p − A_sm` and cancels with the difference. (b) `Atil == A_sm`
off `E` was checked over ~1.1e8 entries across dense / chunked pass-1 / the pass-2 rebuild / decode,
both gate types, head-blocked and not, `E` empty, `E = vis`, `force_empty`, pad rows: **zero
differing entries**. (c) A row with no relation entry has excess **exactly 0.0**, so a forced-empty
relation can never bind — which is what T0 and sanity 5(a) rest on, and it is now *true* rather than
*assumed*.

### 59.3 ⚠️ The regression the change introduced, and its two-part fix

Removing the old test also removed a guarantee nobody noticed it was providing: **`binds ⟹
Σ Atil > 1`**. Without it, `proj_le_masked(Atil, 1, binds=True)` solves the *equality* problem on a
row whose fp32 total is *below* one and scales it **up**. Verified by direct construction — a row
summing to `1 − 3e-5` with eight exact zeros:

```
binds=True, target 1 :  sum 0.99997 -> 1.00000  (+3.0e-5)  theta = -2.5e-6  zeros resurrected: 8
```

⭐ **θ going negative is the tell**: `θ ≥ 0` is exactly what makes INV-6 (Stage-1 zeros permanent)
true at Stage 2 step 1. Through the real normaliser at `τ_j = 200, ρ = 0.6`: **1110 entries with
`Atil == 0` got `a1 > 0`**. Off-relation entries of `a1` *are* `A` (INV-4), so the output moved,
device-dependently — the one thing the design existed to eliminate.

⭐ **My proposed fix was incomplete and the engineer caught it.** I proposed `s = sw − excess`;
`sw` is the fp32 prefix sum and **sparsemax's own internal total is a third kernel-dependent
quantity**, so the target alone does not guarantee `θ > 0` on a razor row. The complete fix is two
parts: project onto the fp64 `sm_mass`, **and** have `proj_le_masked` refuse to bind when its own
arithmetic finds nothing above the target (`θ ≤ 0`). Verified on the same construction: mass down by
exactly the excess, `θ = +3.6e-7`, no zeros lifted, `a1 ≤ Atil`, lands on `sm_mass` to 6e-8; and the
projection removes exactly the excess across 2e-6 → 1e-2.

**The guard costs nothing measurable**: it fired on **0 of 46,246** intended rows; the largest excess
that can escape it is **8.7e-8 = 0.087 × CAP_TOL**, and it is `n_k`-independent (the window is half
an ulp of the fp32 target, not the prefix sum). A crafted cumsum attack failed — torch's CPU `cumsum`
carries a higher-precision accumulator.

⚠️ **The one open question is the guard's window on a GPU.** On a device whose sorted prefix scan is
a genuine fp32 sequential reduction it becomes the scan error, ≈ `n_k·eps/4` ≈ **2.4e-4 at 8K, 240×
CAP_TOL** — and `check_unit_cap.py` cannot see it, because its live-decision assert is one-sided.
Bisect the window on the device and gate it. It is one function and it is the last item on the fix.

### 59.4 What it cost, what it did not, and what it bought

- **Nothing needs re-running.** The regression was **latent on this kernel**: `torch.softmax` here
  always gives `sm_mass − 1 > 0`, so the constant-1 target never scaled a row up in practice. Old vs
  new on identical inputs: identical binding sets, `A` within 1.19e-7. **The fix is device-safety,
  not a numbers correction.**
- **Two undeclared forward-pass changes came in behind it** and both are improvements worth keeping:
  `_softmax_about_max` on the chunked path (`exp((S−m)−r)` rather than `exp(S−lse)` — flat at ~1e-6
  in `n_k` against `½·ulp(|lse|)`, a 6× to 306× improvement), and the projection target. Both mean
  chunked numbers are not bit-comparable across the commits; the appendix must say which formula
  produced them.
- ⭐ **The paper's bound is now device-independent, which it has never been.** `prop:twostep`(iii)
  becomes `Σ_j A_ij ≤ Σ_j A^sm_ij ≤ 1 + n_k·ε/2`: the mechanism never gives a row more than the
  softmax gave it, and the kernel's error is quarantined into a named, bounded, device-*reported*
  term. My earlier "`≤ 1 + δ(n_k)` is vacuous" objection is answered — the content is the
  **converse**, *Stage 1 is the identity on every row the relation did not push over its own softmax
  mass*, and that is true on any kernel. Patches in `claude/marsea_paper_edits_cap.md`,
  applied by `patch_iclr_cap.py` beside it — eight exact-match substitutions, every `old`
  string asserted to occur exactly once **before** anything is written (§0.5), each `new`
  string re-verified unique afterwards, `eq:excess` collision-checked, `.bak` on write.
  Self-tested on a fixture; `--dry-run` against the real `.tex` is the confirmation, since
  the draft lives only in the project and was not editable from the review session without
  the §19 re-emission hazard.

### 59.5 ⭐ Two adjudications against my own reviewers, recorded because the reverse error is costlier

1. **D-2, and I was wrong.** I recommended replacing `torch.nonzero` with
   `((Atil−A_sm)*E).sum(-1, dtype=float64)`. That form **materialises an fp64 copy** — ATen casts in
   `make_reduction`; measured +128.9 MiB on a 64 MiB input, 1.50 GiB at 8K/hb2 against 0.18 GiB for
   the `nonzero` form. The engineer refuted it, took my two correct points (`nonzero` forces a
   device→host sync in the forward and its memory is ρ-dependent), and shipped a **key-chunked masked
   fp64 reduction**: 0.50 GiB, ρ-independent, sync-free, exact. Better than either.
2. **A reviewer's top "blocking" finding was overstated.** They reported S0's layer swap as silently
   corrupting every Phase-B arm. The mechanism is real (S0 mutates `patched_layers` by design, and
   `train.py` asserted only `step` and `phase`), but a Phase-A checkpoint carries **LoRA only** —
   layer-agnostic — and the arm's modules are created fresh at Phase B *by specification*. The
   residue is a provenance/labelling error plus a ~1e-5 bf16 difference in the warm start, not a
   corruption. Add the assert for provenance, not repair.

⚠️ **Standing lesson, and it is §22.2's for the fifth time.** My `log₂` recommendation came from a
search that could not reach the failing regime, and the engineer's own gate script — written to my
spec — found the counterexample I had missed. **A search is a hypothesis about where the worst case
lives, and it inherits every blind spot of the person who wrote it.** The defence that worked was
not a better search but an *analytic* model of the kernel (`n_k·eps/(2L)`) that predicted where to
look, and then a redesign that made the question moot.

### 59.6 State at 9bacef9

Provenance 55/55, tree clean. **132 passed / 14 skipped** (146 collected); `verify_all.py` **0
violations over 1,058,450 evaluations in 63 checks** — every denominator byte-identical for the
fourth round. `check_unit_cap.py` passes at 2048 / 8192 / 16384 and now asserts the §C properties
(`a1 ≤ Atil`, no zero lifted, binding rows land on `sm_mass`, excess exact, `binding_rows > 0` so
they cannot pass vacuously). `cap_slack`, its ceiling and its formula are deleted.

**Three one-line blockers remain before launch**, all verified by execution and all in
`claude/marsea_decisions_register.md`'s Open table: `preflight.sh` exits 1 on the condition it has
just declared non-fatal; `run_evalsuite.sh` still drops E9 arms silently on the documented recovery
path; `collect.py`'s `per_seed` is short *and* mislabelled.

⚠️ **The calendar is now the binding constraint.** Abstract 18 Sep, paper 25 Sep; nothing past E1
has been run; preflight → Phase A → S0 → Phase B → the 98-job eval queue is **three to four days of
wall clock at 8 GPUs** before `collect.py` emits a single table.

---

## §60. The two drafts converged, the cap edits applied, the record merged (2026-09-15)

**Occasion.** Luke asked for an archive of every current document, then for everything to be
brought up to the latest. Packing the archive required a real `marsea_iclr.tex` rather than a
retyped one, which meant reading it off his machine — and that is what surfaced the drift.

### 60.1 ⚠️ The two copies of the ICLR draft had diverged, and §0.3a's rule caught it again

| copy | date | state |
|---|---|---|
| project `claude/marsea_iclr.tex` | 2026-09-10 | current |
| `~/Dev/marsea/documents/marsea_iclr.tex` | 2026-09-08 | **two decisions behind** |

The working copy was missing exactly the two paper edits of §52 and §53 — **D-31**'s sink
sentence at the end of Sec. 2.3, and **D-9a**'s rewrite of App. H's *E7, the column ground truth*
(it still carried the merged target set and `m_j = 2m`). Everything else probed for was present in
both, and the three verification rows §48 added on 2026-09-08 were in the working copy, which
dates it precisely: current through §48, missing §52 and §53.

⚠️ **Two independent sources agree on that diff and neither alone would have been enough.** The
record is the change log — §52 says *"Paper Sec. 2.3 sentence added"*, §53 says *"Paper App. H E7
paragraph rewritten"* and nothing else in the window touches the paper — and a direct probe of the
working copy found those two gaps and no others. **The record earned its keep here**: a probe
tells you what is missing from what you thought to look for, and the change log tells you what to
look for.

⚠️⚠️ **And the trap was live.** `patch_iclr_cap.py`'s eight `old` strings all match exactly once in
the *stale* copy, so the cap patches would have applied there cleanly and produced a draft with the
numerical fix and without D-31 and D-9a. **§0.3a's standing rule — a fix applied to one draft does
not certify the other, in either direction — held for a third time, now between two copies of the
*same* draft rather than between the two papers.** Where §19 was a stale `cp` inside one machine,
this was a stale copy across two stores, and the mechanism that caught it was the same: check the
artifact, not the intention.

### 60.2 What was done, in order, each step asserting before it wrote

1. **`portforward_d9a_d31.py`** — two exact-match patches carrying the 2026-09-10 text, written so
   that both `old` strings must occur exactly once before anything is written, and post-checked for
   the presence of D-31 and both column kinds and the *absence* of the merged target set and
   `m_j = 2m`. ⭐ **Both replacement passages were then verified byte-identical to the project's own
   text** by retrieving them and diffing, so nothing in the merged draft is transcribed prose.
2. **`patch_iclr_cap.py`** — the eight unit-cap patches of §59.4, dry-run first, all eight matching
   exactly once.
3. **`verify_all.py`** — the `Prop. A(iii)` block re-targeted from the literal `1.0` to the row's
   own softmax mass, at four sites plus the printed label, so the harness prints what
   `prop:twostep`(iii) now claims. ⭐ **Re-run and diffed against the pre-edit output: exactly one
   line differs, the label.** Every count, every denominator and the 69.3 % sub-unit figure are
   unchanged, which is the point — in fp64 numpy the two targets are the same number, and the edit
   is about what the harness *says*, not what it computes.
4. **Build:** `pdflatex ×3` — **42 pp, 0 errors, 0 undefined refs, 0 undefined citations, 0
   multiply-defined labels, 0 overfull boxes.** All ten edits verified in the **rendered PDF**
   (§0.5), not only in the source: the sink sentence on p7, the two column kinds on p41, the
   finite-precision remark on p9, the device-statistic paragraph on p35.
5. **Converged**: the merged draft written to the project *and* to `~/Dev/marsea/documents/`, with
   the built PDF and the updated harness, so the two stores hold the same bytes.

⚠️ **The page count is 41 → 42.** Two of the additions are main-text (D-31's sentence in Sec. 2.3,
the finite-precision remark in Sec. 3.2); the other two are appendix. Under §30.0 the 9-page limit
is deferred until the experiments land, so this is not a problem *now* — but the final compression
pass owes those two paragraphs a home, and §27.5's diagnostic (build once with a float deleted)
is the method, not shaving.

### 60.3 The repo's records were a week stale, and that is the more dangerous of the two findings

`~/Dev/marsea/documents/records/` held the implementation spec at **v4.6 (2026-09-08)** against the
current **v4.11**, and a 6 KB decisions register against the current 16 KB. Five spec versions —
including the whole `cap_slack` arc and its retirement — were invisible to anyone reading the repo.
**That is the copy an engineer or a fresh session reads**, and it would have described a deleted
constant as live. All current records are now written there alongside the paper.

### 60.4 ⭐ This record is one file again

`claude/marsea_master_record_part2.md` existed only because this file comes back from
`project_read` inline, so amending it meant re-emitting ~90 KB — the §19 hazard. With the repo
folder connected the record is a file on disk, so the workaround is retired: §§54–59 are merged
in above, §§50–53 were forward-ported from the project copy and **verified byte-identical against
it** by the same retrieve-and-diff method as the paper passages, and part 2 is now a superseded
pointer. ⚠️ **Do not write to part 2.** At 267 KB this file may now cross the threshold at which
`project_read` returns a path rather than inline text — if it does, every future amendment is
surgical, which is the outcome §0.4 has wanted since 2026-08-25.

### 60.5 State

| | |
|---|---|
| `claude/marsea_iclr.tex` | **42 pp**, 0 err / 0 undef / 0 multiply-def / 0 overfull; D-31, D-9a and all eight cap patches applied; identical in the project and in `~/Dev/marsea/documents/` |
| `verify_all.py` | **0 violations over 1,058,450 evaluations in 63 checks**; `Prop. A(iii)` now stated and tested against the row's own softmax mass |
| records | spec **v4.11**, register 2026-09-15, this record §1–§60 — in the project and in the repo |
| still open | the three one-line launch blockers (§59.6); T17(d), the θ-window on the target device; evaluation throughput, unmeasured for a fifth round; **E6 / IHEval**, the oldest open decision |

### 60.6 ⭐ The near-miss is now designed out, not just recorded

Documenting a trap does not disarm it. `patch_iclr_cap.py` was shipped to three places and its
eight `old` strings still matched the stale draft, so anyone who ran it on a backup, an old branch
or an untouched tarball would reproduce the near-miss exactly. **It now carries a prerequisite
guard**: before it checks a single `old` string it requires the three §0.3b probes — D-31's sink
sentence and both of D-9a's column kinds — and on a copy missing any of them it writes nothing,
names what is missing, and prints the `portforward_d9a_d31.py` command to run first. Tested on all
three states of the draft: **stale → refused; forward-ported but unpatched → applies; already
patched → refused.**

⭐ **The general rule this earns, and it belongs beside §0.5's "assert every match up front".**
*Asserting that your `old` strings match is not the same as asserting you are editing the right
file.* Exact-match patching is a guard against editing the wrong **place**; it is no guard at all
against editing the wrong **copy**, and the two failures look identical from inside the script —
both print eight clean matches. A patch script against a document with more than one copy in
circulation should assert an identity probe as well as its match counts.

---

## §61. `e6ca44f` — the θ-window measured and gated; a duplicate harness found (2026-09-15)

Fifth review round, same day as §60. Full review: `claude/marsea_review_e6ca44f.md`.
**Provenance: 56/56 sha256 match; no code file dirty; 146 passed / 14 skipped here with
`MARSEA_DEBUG=1`; harness 0 violations over 1,058,450 evaluations in 63 checks.**

### 61.1 ⭐ The last open mechanism question is closed, and it moved the paper

The θ-guard window (§59.3) is **measured on the device and gated**, not assumed:
`check_unit_cap.py` bisects it and exits 3 if it exceeds `cap_tolerance(n_k)`. Reproduced here on
CPU — 8.8e-8 at 8K, 1.11e-7 at 16K, agreeing with the engineer's ~1.2e-7. ⚠️ **The CUDA figures
(1.06e-6 at 8K, 1.95e-6 at 16K, both above `CAP_TOL`) are theirs alone; this container has no GPU.**

⚠️ **The measurement falsified one clause of the remark §60.2 had just applied.** Patch 2 said
"(iii), (v) and (vi) hold on any kernel". The binding *decision* is kernel-free and (v) and (vi)
hold *exactly* — an empty relation's excess is bitwise `0.0`, verified — but (iii) is a theorem
about the program, and the implementation meets it only up to the window. Amended by
`patch_iclr_window.py`: the remark now separates the decision from the realized total, and App. F
gains the **window as a second device statistic beside δ**, with the `n_k·ε/4` sequential worst
case named. Build after: **42 pp, 0 err / 0 undef / 0 multiply-def / 0 overfull.**

⭐ This is the second time the cap arc has ended somewhere better than it started: App. F now
reports two numbers measuring the distance between the program and the arithmetic that runs it.

### 61.2 ⚠️⚠️ Two copies of `verify_all.py`, drifted in both directions — and half of it was mine

`scripts/verify_all.py` (779 lines) carries the machine-readable verdict and exit code
`preflight.sh` gates on; `documents/verify_all.py` (761 lines) carries §60.2's re-target of
`Prop. A(iii)` to the row's own softmax mass. **Neither had both**, and the copy that actually
runs still printed the label `tab:verification` no longer uses — so the reproducibility statement
was false again, for the copy that runs.

**I caused half of it**, editing the paper's copy without noticing a second existed, *one turn
after writing §60.6 about this exact hazard.* Merged and verified (0 violations, 63 checks, exit
0). ⚠️ **The merge is not the fix**: one of the two paths must stop being a file — a symlink, or a
packaging step nobody edits. This is §19 with a different filename, and the project has now paid
for it three times: §19, the two ICLR drafts (§60.1), and this.

⭐ **Standing addition to §0.5:** *before editing any file in this project, check whether a second
copy of it exists.* Exact-match patching guards the place (§60.6); the §0.3b probes guard the
copy of the draft; neither helps with a file you did not know was duplicated. `find . -name
<basename>` costs nothing.

### 61.3 Where I was wrong again, and the probe that showed it

My §E last round asked for INV-3 to be tightened to fp64 level. The engineer refused on T6:
Stage 2's own fp32 arithmetic overshoots the quota. **They are right.** I then tested the opposite
— that `tol_sum` might be too *small* at scale — and my first probe said so (9.9e-5 at
`n_k = 1024`), but it drew rows of mass ~100. At physical scale the overshoot is flat at ~8e-7
across every realistic `(|E_i·|, τ_i)`, ~150× inside INV-3's allowance, because it scales with the
row's **mass**, not its length. The split — tight INV-3a on `a1`, INV-3 on `A` keeping Stage 2's
allowance — is correct.

Second refuted recommendation of this arc after `8·eps·log₂ n_k`, and ⚠️ **`cap_tolerance` is that
same log form again**: `8·eps·log₂ n_k` = 1.34e-5 at 16K against the engineer's own stated
sequential worst case `n_k·ε/4` = 4.9e-4, a factor of 36. Benign *because it is gated rather than
assumed*, but the justification should be "4× the window measured on this device", not a law with
nothing behind it.

### 61.4 ⭐ The best change in the drop was not on their list

`INV-5` now recomputes the binding criterion with the **masked** form while `row_masses` uses the
**two-sum** form, so the invariant is an independent check rather than a restatement of the
mechanism's own arithmetic. That is §22.2's standing lesson applied unprompted. The two-sum form
itself I verified **bitwise identical** to the masked one at 2K/8K/16K — stronger than the 2e-12
the docstring claims.

### 61.5 Open before the launch

1. Reconcile `verify_all.py` to one file (§61.2).
2. Write the window gate's failure branch into `RUNBOOK_nebius.md`: if `window > tol_cap`, set
   `tol_cap := 4 × window_measured`, record it in App. F, re-run the gate. Undecided, it is a stop
   three days before the abstract.
3. The paper-edit docs are now **behind** the paper, not ahead of it. Anything still owed is a
   patch against the current `.tex`, checked with the §0.3b probes.

⚠️ **Five review copies were deleted from the repo root** between 17:08 and 18:23 UTC by neither
me nor the engineer. **Nothing is lost** — all five are in the project, in the archive under
`04_reviews/`, and restorable in one step. Luke's call whether they go back.

---

## §62. `4f754ed` — the four items landed; an unbounded tolerance override (2026-09-15)

Sixth round, same day. Full review: `claude/marsea_review_4f754ed.md`.

### 62.1 ⚠️ The wrong tarball was attached, and D-32 caught it in one command

The upload was **byte-identical** (md5 `db1aed28…`) to the previous round's: `e6ca44f`, 71 HANDOFF
lines, against the claimed `4f754ed` with 73. ⭐ **The protocol earned its cost again** — a review of
the wrong tree reports every fix as missing, which is how §56 lost three rounds.

⭐ **And the recovery is the new capability worth recording**: with Luke's repo folder connected,
the review was done from `~/Dev/marsea` itself — nine files staged out of the working tree — rather
than from any bundle. **That is better provenance than a tarball, not worse**: the tree, not a copy
of it. ⚠️ The trade is that the "no code file dirty" check then rests on the HANDOFF status block
rather than on an independent `git status`.

### 62.2 Everything asked for landed, and as recommended

One harness (`scripts/verify_all.py`, **sha256-identical to the file I delivered**;
`documents/verify_all.py` a real symlink, device-confirmed, harness running through it at 0
violations / 63 checks / exit 0); the manifest covers the tree; the window gate's failure branch is
`MARSEA_TOL_CAP = 4 × measured window`, recorded by `unit_cap_record` and stated at RUNBOOK step 4b;
the log law is documented as a number to clear rather than a derivation. 149 passed / 8 skipped
offline here. `handoff.sh` bundles only `marsea scripts tests`, so the symlink never enters a
handover tarball — ⚠️ but whatever builds the paper's **supplementary** bundle must dereference it.

### 62.3 ⚠️⚠️ The override has no ceiling — `cap_slack` wearing a new hat

`cap_tolerance` takes `max(default, MARSEA_TOL_CAP)`. The floor is right; there is no upper bound.
Measured: `MARSEA_TOL_CAP=1` gives `tol_cap = 1.0` — **a whole row of mass** — so INV-3a and INV-5b
admit anything, silently, for a 30-hour queue; `1e-5x` raises `ValueError` inside an invariant check
mid-run rather than at startup; and because `window_ok` compares against the *overridden* tolerance,
**the gate becomes self-certifying** and a stale override survives a reschedule onto a different GPU.

Fix: refuse an override above `n_k·ε/4` (nothing above that is a real scan error) and above
`8 × window_measured_now` (justified by a different device), and parse once at startup naming the
variable. At the dev GPU's window the recommended value is 8e-6, under both bars, so the intended
path costs nothing.

⭐ **Why this is worth pressing on.** It is the same failure shape as `cap_slack`: a number bounded
by nothing physical, correct on the device it was set on. The arc's whole lesson was to bound the
quantity by measurement or by the arithmetic — this override is bounded by neither, and it is the
last place in the pipeline where a human types a tolerance in.

### 62.4 State

Blockers: none. The mechanism question: closed and gated. The paper: **42 pp, 0 err / 0 undef /
0 multiply-def / 0 overfull**, cap patches and window amendment applied, identical in the project
and in `~/Dev/marsea/documents/`; it is now two amendments ahead of every paper-edit doc in the
repo. ⚠️ **The calendar is the only thing left**: abstract in two days, nothing beyond E1 run, and
preflight → Phase A → S0 → Phase B → the eval queue is three to four days at 8 GPUs.

---

## §63. `7e05ad6` — the override bounded above; the eval path is not validated (2026-09-15)

Seventh round. Full review: `claude/marsea_review_7e05ad6.md`.

### 63.1 Provenance clean, and the bundle is the repo

Hash, line count and filename all agree for the first time; md5 distinct from all seven previous
uploads; **60/60 manifest entries verify and the manifest now covers the tarball exactly in both
directions** (last round's two unlisted PDFs are in). `~/Dev/marsea/HANDOFF.txt` reads the same
hash. **151 passed / 8 skipped** offline here; the 19 review regression tests pass.

### 63.2 The override is bounded by the arithmetic and by the measurement

Parsed once at import (a typo now stops the process naming `MARSEA_TOL_CAP`, not a bare
`ValueError` 30 hours in); refused above `L·ε/4`, the largest error a sequential fp32 prefix scan
can make at the run's length; refused above `8 ×` the window `check_unit_cap.py` has just measured,
so a stale value carried onto a different GPU no longer makes the gate self-certifying. **I ran
every bar**, including the gate end-to-end on CPU with a stale `5e-5`: exit 3 with the refusal.

⭐ **A concern withdrawn.** Moving a parse to import time usually costs the test that covered the
environment path, and the engineer's own note read like that regression. It is not: the new tests
drive real subprocesses with real `env=`, so that path is better covered than before. *Worth
remembering as a review habit — read the new tests before charging a coverage loss.*

### 63.3 ⚠️ NEW — validation is on training and the gate, not on evaluation

`unit_cap_record(L: int = 0)` and **`run_eval.py:214` calls it with no argument**, so `if L:` is
false: no `validate_tol_cap_override` on the eval path, and `tol_cap_at_L` /
`tol_cap_ceiling_at_L` record as **`None` in every eval table**. That removes exactly the
mitigation the last two rounds rested on — *"the value in force is recorded in every README and
eval table"*. Blast radius is limited (`MARSEA_DEBUG` is set only by `preflight.sh` and
`run_tests.sh`, so invariants do not run in the 98-job queue), but an operator debugging a job
turns them on. **Fix is one argument**: the length is in the set name (`E3_L16384_K*`).

### 63.4 ⚠️ Three shipped tests cannot pass from the shipped bundle

`test_review_4f754ed.py` asserts on `RUNBOOK_nebius.md` and `test_review_e6ca44f.py` on
`documents/verify_all.py`, both excluded from the handover tarball by design. A reviewer running
the shipped tests on the shipped bundle sees three reds that are green in a clone — noise of
exactly the kind D-32 exists to remove. Skip when the path is absent, or ship the two paths.

⭐ **A style note the round earned:** two assertions match *source text* and *runbook prose* rather
than behaviour. Both tests also check behaviour, so nothing is untested — but the prose assertion
is what failed for me against a runbook one commit old, reporting a documentation drift as a
mechanism failure.

### 63.5 ⚠️ The mechanism is finished; the calendar is not

Nothing in the mechanism is open. Seven review rounds have made it unimpeachable and **none has
produced a number for the paper**. Abstract due in two days, nothing beyond E1 run, and
preflight → Phase A → S0 → Phase B → the eval queue is three to four days at eight GPUs. The
honest fallback — submit the abstract on the theory and E1, both finished, and decide the
experiments question deliberately — is better than letting a 3 a.m. gate failure decide it.
