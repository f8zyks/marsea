# MarSea — paper edits: the unit cap in finite precision

*For `claude/marsea_iclr.tex`. Written 2026-09-15, against the tree at `9bacef9`. These supersede
item 18 of `PAPER_EDITS_e982f83.md` (which I have still not seen — if it conflicts, this file is
the later analysis and the numbers here are the measured ones).*

⚠️ **Apply these with an asserting patch script**, per record §0.5: assert every `old` string
occurs **exactly once** before writing anything, then grep-verify each `new` string afterwards.
A script that writes before it can fail has cost this project twice.

---

## Why the paper has to change at all

Three rounds were spent on a slack constant because `prop:twostep`(iii) was stated against the
literal constant **1**, and the fp32 row sum of a softmax is not 1 — it is `1 + δ` with `δ` a
property of the **reduction kernel**, not of fp32 arithmetic:

| kernel | worst-case `δ` at 16K | mechanism |
|---|---|---|
| CUDA (pairwise) | ~1e-6 | pairwise reduction |
| x86, 16 lanes | 6.0e-5 | `n_k·eps/(2L)`: the lane holding the row max sits at 1.0 and drops every tail term under half an ulp |
| x86, 8 lanes | 1.2e-4 | same, `L = 8` |
| scalar | up to 1e-3 | `n_k·eps/2` |

Measured, verified to within 1 % of `n_k·eps/(2L)` at every length from 2K to 32K. **No constant
covers that range**, so a bound written against `1` makes the cap's binding set device-dependent
and is falsifiable by a reviewer with ten lines of code.

The resolution is not a tolerance but a **change of tested quantity**. Since `Ã = A^sm` bitwise off
the relation (part (iv)), the row's excess over its own softmax mass is

```
Σ_j Ã_ij − Σ_j A^sm_ij  =  Σ_{j∈E_i.} (Ã_ij − A^sm_ij)      exactly
```

— a difference of in-relation terms that **reads no row total**. The implementation binds on that,
and projects a binding row onto `Σ_j A^sm_ij`. So the bound becomes `Σ_j A_ij ≤ Σ_j A^sm_ij`,
which is **one unit in exact arithmetic and true on any kernel**, and an empty relation is the
identity bitwise rather than to a tolerance.

⭐ **This is a strengthening, not a weakening.** The paper's pitch is that its invariants are
checked; it can now state a guarantee that survives the floating-point kernel it is checked on,
which very few mechanism papers do. Patch 7 is the one that says so, and I would keep it.

---

## Patch 1 — `prop:twostep`(iii), the statement

**old** (occurs once, in the proposition body)

```latex
\textbf{(iii)} $\sum_j A_{ij}\le 1$ for every query, with equality only when step 1's cap binds,
so a query may still spend less than a unit;
```

**new**

```latex
\textbf{(iii)} $\sum_j A_{ij}\le \sum_j A^{\mathrm{sm}}_{ij}$ for every query --- the mechanism
never gives a row more than the unmodified layer gave it, which is one unit in exact arithmetic
--- with equality only when step 1's cap binds, so a query may still spend less;
```

---

## Patch 2 — the remark that makes it a result rather than a caveat

**old** (occurs once: the closing line of `prop:twostep`, immediately before `\end{proposition}`)

```latex
\textbf{(vi)} with $\mathcal{E}=\emptyset$ the output equals $A^{\mathrm{sm}}$ entrywise.
\end{proposition}
```

**new**

```latex
\textbf{(vi)} with $\mathcal{E}=\emptyset$ the output equals $A^{\mathrm{sm}}$ entrywise.
\end{proposition}

\paragraph{Why (iii) is stated against the row's own mass, and what that buys in finite
precision.} In exact arithmetic $\sum_j A^{\mathrm{sm}}_{ij}=1$ and (iii) is the unit bound. We
write it against the row's own mass because the two are not the same number in floating point,
and the difference is a property of the \emph{reduction kernel} rather than of the mechanism: an
fp32 softmax row sums to $1+\delta$ with $|\delta|\le n_k\varepsilon/2$, the worst case being a
row with one dominant entry and a long tail lying just under half an ulp of it, so that a
$L$-lane reduction discards $1/L$ of that tail and $\delta$ reaches $n_k\varepsilon/(2L)$. That is
$\approx10^{-6}$ for a pairwise reduction and $6\times10^{-5}$ at $n_k=16{,}384$ for a $16$-lane
one. \textbf{Testing whether a row exceeds \emph{one} therefore makes the cap's binding set
device-dependent, and no fixed tolerance is correct on every kernel.} Testing whether the
\emph{relation} added mass does not: by (iv), $\tilde A=A^{\mathrm{sm}}$ off $\mathcal{E}$, so
\begin{equation}
\textstyle\sum_j \tilde A_{ij} - \sum_j A^{\mathrm{sm}}_{ij}
\;=\; \sum_{j\in\mathcal{E}_{i\cdot}}\big(\tilde A_{ij}-A^{\mathrm{sm}}_{ij}\big)
\label{eq:excess}
\end{equation}
exactly --- a difference of in-relation terms that reads no row total. The implementation binds
\eqref{eq:rowstep1} on \eqref{eq:excess} and projects a binding row onto $\sum_jA^{\mathrm{sm}}_{ij}$,
so (iii), (v) and (vi) hold on any kernel and an empty relation reproduces standard attention
\emph{bitwise} rather than to a tolerance. App.~\ref{app:numerics} reports the kernel's own
$\delta$ as a device statistic beside the checks rather than inside them.
```

⚠️ **Check the label** `eq:excess` does not collide; `grep -c 'label{eq:excess}'` must be 1 after.

---

## Patch 3 — §2.2, the step-1 sentence

**old** (occurs once, in `sec:method-fanin`)

```latex
This is the row as standard attention produced it, modified only where a column relation
re-shaped it, held beneath one unit: every row of $A^{\mathrm{sm}}$ sums to one, so the cap is the
identity on any row no key has re-shaped, and binds only where fan-out concentration has pushed
a row over a unit.
```

**new**

```latex
This is the row as standard attention produced it, modified only where a column relation
re-shaped it, held beneath the unit that row already carried: every row of $A^{\mathrm{sm}}$ sums
to one, so the cap is the identity on any row no key has re-shaped, and binds exactly where a
fan-out concentration has \emph{added} mass to the row --- a condition on the relation's own
entries, which is what makes it well posed in finite precision (Prop.~\ref{prop:twostep} and the
remark after it).
```

---

## Patch 4 — the proof of `prop:twostep`(iii)

**old** (occurs once, inside `\begin{proof}[Proof of Prop.~\ref{prop:twostep}]`)

```latex
(iii) Step 1 of Stage 2 is a
projection onto $\{a\ge0:\sum_ja_j\le1\}$, so $\sum_ja^{(1)}_{ij}\le1$;
```

**new**

```latex
(iii) Step 1 of Stage 2 is a
projection onto $\{a\ge0:\sum_ja_j\le\sum_jA^{\mathrm{sm}}_{ij}\}$, which is the unit cap since
the row-softmax row sums to one, so $\sum_ja^{(1)}_{ij}\le\sum_jA^{\mathrm{sm}}_{ij}$;
```

---

## Patch 5 — `tab:verification`, the row that states the bound

**old** (occurs once)

```latex
Prop.~\ref{prop:twostep}(iii), sub-unit rows & $\sum_j A_{ij}\le1$; $69.3\%$ spend less & 0 / 80741 \\
```

**new**

```latex
Prop.~\ref{prop:twostep}(iii), sub-unit rows & $\sum_j A_{ij}\le\sum_jA^{\mathrm{sm}}_{ij}$; $69.3\%$ spend less & 0 / 80741 \\
```

⚠️ **The harness must print the same thing.** `verify_all.py`'s label for that row has to match, or
the reproducibility statement's "reproduces every figure in Tables~\ref{tab:verification} and
\ref{tab:transferchecks}" becomes false. Denominator is unchanged (80,741).

---

## Patch 6 — App. D, Limitations

**old** (occurs once)

```latex
\textbf{Row mass is bounded above by one but not below.}
$\sum_jA_{ij}\le1$ (Prop.~\ref{prop:twostep}(iii)), so no query ever absorbs more than the unit
softmax spends
```

**new**

```latex
\textbf{Row mass is bounded above by the row's own softmax mass, and not below.}
$\sum_jA_{ij}\le\sum_jA^{\mathrm{sm}}_{ij}$ (Prop.~\ref{prop:twostep}(iii)), which is one unit in
exact arithmetic, so no query ever absorbs more than the unit softmax spends
```

---

## Patch 7 — App. F (numerics), the device statistic ⭐

This is the one that turns the whole episode into a contribution rather than a footnote. Put it at
the end of the `\paragraph{Protocol.}` block.

**old** (occurs once, the closing sentence of the Protocol paragraph)

```latex
it; total runtime is under $25$ seconds on one CPU core and the only dependency is
\texttt{numpy}. The last block checks App.~\ref{app:selective} and is reported there.
```

**new**

```latex
it; total runtime is under $25$ seconds on one CPU core and the only dependency is
\texttt{numpy}. The last block checks App.~\ref{app:selective} and is reported there.

\paragraph{One check is about the device rather than the mechanism, and we separate it.} The unit
cap of \eqref{eq:rowstep1} is the one place where a floating-point reduction could have entered a
\emph{decision} rather than a value. As the remark after Prop.~\ref{prop:twostep} sets out, the
implementation binds on the relation's own mass excess \eqref{eq:excess} and projects onto
$\sum_jA^{\mathrm{sm}}_{ij}$, so the binding set is a function of in-relation terms alone and the
softmax reduction's error never reaches it; the excess is accumulated in double precision from
single-precision entries cast before subtracting, giving it no error of its own, and a row with an
empty relation has excess exactly $0.0$, so an empty relation cannot bind the cap on any kernel.
\textbf{We therefore report the kernel's own row-sum error $\delta$ as a statistic of the machine,
beside the checks and not inside them}, and a released implementation should do the same: the
guarantee is device-independent and the measurement of $\delta$ is what demonstrates that it had
to be made so. On the reference CPU, $\delta$ reaches $3.0\times10^{-5}$ at $n_k=8192$ and
$6.0\times10^{-5}$ at $16{,}384$, against $\approx10^{-6}$ for the pairwise reduction of the
deployment GPU --- a spread no fixed tolerance spans, which is why the criterion reads no row
total.
```

---

## Patch 8 — App. E (selective exclusivity), cost (c)

**old** (occurs once)

```latex
\eqref{eq:rowstep1} caps the \emph{whole} row at one unit before any restriction is applied, and
step 2 conserves the relation's quota within it, so the row total never exceeds one and the
complement keeps its step-1 value.
```

**new**

```latex
\eqref{eq:rowstep1} caps the \emph{whole} row at the unit it already carries, before any
restriction is applied, and step 2 conserves the relation's quota within it, so the row total
never exceeds its own softmax mass and the complement keeps its step-1 value.
```

---

## What is NOT in these patches, and is still yours to decide

Unchanged from my last three reviews; none of it is a correctness fix and none of it is edited
unilaterally (D-2):

| Claim | Status |
|---|---|
| Llama-3.2-3B cross-family check (Setup) | no arm exists |
| "We ablate both emission policies" (§2.6) | no arm; `frozen_prefix` is chosen by arm type, never swept |
| the hard-variant ablation (App. D) | nothing |
| "E9 ablates the tying of $\tauK$ to $\tauQ$" (App. D) | no `tie_tau` anywhere |
| E9 over layer set and block size (App. H) | no arm |
| **E6 on IHEval** | no acquisition step; the queue prints "NOT RUN". **The oldest open decision in the project** — acquire it or say in the paper it was not run |
| E7's measured PR trace against Fig. 1 | no $\tau_j$ sweep on a trained checkpoint exists |
| efficiency as a function of measured coverage (App. H) | no FLOPs, no parameter count |
| E3's padding control "at every point" | one matched config; reword or extend the grid |
| B1/B4 in a headline table with a std column | D-24 gives them one seed; needs a footnote |
| "moderate context ($2$--$16$K tokens)" (Setup) | only 8K and 16K exist |

**Forced by `--n 400`** (review at e16a843, §H), and these *are* new:

- E3's **per-stratum exact-set accuracy** now carries a 95 % interval of $\pm0.14$ at 80 examples
  per stratum per seed. Report it **pooled over $n$**, with the per-stratum version in the
  appendix and its interval stated.
- **"Flat" needs an equivalence band.** At a per-stratum seed sd of 0.015–0.040 the 95 % bound on
  total drift across the four doublings is $\pm0.021$ to $\pm0.057$. Supportable: *"drifts by less
  than 5 points across a 16$\times$ sweep in $n$ while the comparator falls from 0.5 to 0.03."*
  Not supportable: unqualified "flat". Pre-commit to the band.
- **The depth panel is one seed and one baseline** (MarSea against B0 only, not the baseline
  family). Say so in the caption; no error bar.
- **MarSea vs B3 and vs B2 are under-powered at three seeds** — state it rather than report a
  silent null.

**Also owed, and cheap:** which softmax formula produced the chunked numbers (the chunked path
computes `exp((S−m)−r)`, the dense path `torch.softmax`; they differ by ~1e-6 relative), and the
`cap` rule in force, which the run README already records.

---

## The patch script

⭐ **`patch_iclr_cap.py` ships beside this file and applies all eight.** Put it in the directory
where `marsea_iclr.tex` builds and run:

```
python3 patch_iclr_cap.py --dry-run          # checks all 8 `old` strings, writes nothing
python3 patch_iclr_cap.py                    # applies, backs up to marsea_iclr.tex.bak
```

It follows §0.5 literally: it **asserts every `old` string occurs exactly once before writing
anything**, refuses if any patch is already applied, refuses if `eq:excess` would collide, and
after writing re-reads the file and verifies every `new` string is unique. Exit code 0 only when
all of that passes. Both failure paths leave the `.tex` untouched, and a post-check failure prints
the `mv` that restores the backup.

⚠️⚠️ **It also refuses to run on the wrong copy of the draft, and that guard is the important
one.** All eight `old` strings below match the **2026-09-08** draft too — the one missing D-31 and
D-9a — so patching that file succeeds and silently yields a draft with the numerical fix and
without two standing decisions. That very nearly happened on 2026-09-15 (record §60.1). Before it
checks a single `old` string the script now requires the three §0.3b identity probes (D-31's sink
sentence, and both of D-9a's column kinds); on a copy missing any of them it writes nothing and
prints the `portforward_d9a_d31.py` command to run first. *Asserting that your `old` strings match
is not the same as asserting you are editing the right file.*

Verified here on a fixture containing all eight `old` strings: 8/8 matched once, all eight `new`
strings unique afterwards, `eq:excess` unique, and a second run correctly refused. It has **not**
been run against the real `marsea_iclr.tex`, which lives only in the project — the `old` strings
were read out of it, so `--dry-run` is the confirmation and it costs a second.

---

## Order

Patches 1–6 are the correctness fix and should go in together; 7 is the one that earns something
from the episode; 8 is consistency. Then `verify_all.py`'s row label (Patch 5's warning), then a
clean build with the usual sweep — 0 errors, 0 undefined refs, 0 undefined citations, 0 overfull,
0 multiply-defined, and `grep -c` on each new string.
