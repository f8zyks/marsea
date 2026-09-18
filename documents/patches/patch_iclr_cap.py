#!/usr/bin/env python3
"""
patch_iclr_cap.py -- apply the unit-cap paper edits to claude/marsea_iclr.tex.

Written 2026-09-15 against the tree at 9bacef9.  Companion to
`claude/marsea_paper_edits_cap.md`, which explains *why* each patch is needed.

Protocol (master record Sec. 0.5: "a patch script must write before it can fail
-- or assert every match up front"):

  1. read the file once;
  2. assert every `old` string occurs EXACTLY ONCE -- if any does not, print the
     offending patch names and exit WITHOUT writing anything;
  3. apply all substitutions in memory;
  4. write `marsea_iclr.tex.bak` then `marsea_iclr.tex`;
  5. grep-verify every `new` string occurs exactly once, and that the new label
     `eq:excess` is unique.

Usage:
    python3 patch_iclr_cap.py [path/to/marsea_iclr.tex]   # default: ./marsea_iclr.tex
    python3 patch_iclr_cap.py --dry-run [path]            # check matches only

After a successful run:
    - change verify_all.py's label for the Prop. twostep(iii) row to match
      Patch 5 (it currently prints the `<= 1` form);
    - rebuild: pdflatex x3 (the .bbl is present and citations are unchanged);
    - check 0 errors / 0 undefined refs / 0 undefined citations / 0 overfull /
      0 multiply-defined, and read the changed pages in the RENDERED PDF.
"""

import sys
import os
import shutil

# ---------------------------------------------------------------------------
# The patches.  Each is (name, old, new).  `old` must occur exactly once.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Prerequisites: markers that identify the base as the 2026-09-10-or-later draft.
# Each is text the pre-D-9a / pre-D-31 draft does NOT contain. See main().
# ---------------------------------------------------------------------------

PREREQUISITES = [
    ("D-31, the attention-sink exclusion (Sec. 2.3)",
     "One position is excluded from every relation by construction"),
    ("D-9a, the coreference column (App. H)",
     r"\emph{coreference column}"),
    ("D-9a, the value column (App. H)",
     r"\emph{value column}"),
]

PATCHES = []

PATCHES.append((
    "1 prop:twostep(iii) statement",
    r"""\textbf{(iii)} $\sum_j A_{ij}\le 1$ for every query, with equality only when step 1's cap binds,
so a query may still spend less than a unit;""",
    r"""\textbf{(iii)} $\sum_j A_{ij}\le \sum_j A^{\mathrm{sm}}_{ij}$ for every query --- the mechanism
never gives a row more than the unmodified layer gave it, which is one unit in exact arithmetic
--- with equality only when step 1's cap binds, so a query may still spend less;""",
))

PATCHES.append((
    "2 remark after prop:twostep",
    r"""\textbf{(vi)} with $\mathcal{E}=\emptyset$ the output equals $A^{\mathrm{sm}}$ entrywise.
\end{proposition}""",
    r"""\textbf{(vi)} with $\mathcal{E}=\emptyset$ the output equals $A^{\mathrm{sm}}$ entrywise.
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
$\delta$ as a device statistic beside the checks rather than inside them.""",
))

PATCHES.append((
    "3 sec:method-fanin step-1 sentence",
    r"""This is the row as standard attention produced it, modified only where a column relation
re-shaped it, held beneath one unit: every row of $A^{\mathrm{sm}}$ sums to one, so the cap is the
identity on any row no key has re-shaped, and binds only where fan-out concentration has pushed
a row over a unit.""",
    r"""This is the row as standard attention produced it, modified only where a column relation
re-shaped it, held beneath the unit that row already carried: every row of $A^{\mathrm{sm}}$ sums
to one, so the cap is the identity on any row no key has re-shaped, and binds exactly where a
fan-out concentration has \emph{added} mass to the row --- a condition on the relation's own
entries, which is what makes it well posed in finite precision (Prop.~\ref{prop:twostep} and the
remark after it).""",
))

PATCHES.append((
    "4 proof of prop:twostep(iii)",
    r"""(iii) Step 1 of Stage 2 is a
projection onto $\{a\ge0:\sum_ja_j\le1\}$, so $\sum_ja^{(1)}_{ij}\le1$;""",
    r"""(iii) Step 1 of Stage 2 is a
projection onto $\{a\ge0:\sum_ja_j\le\sum_jA^{\mathrm{sm}}_{ij}\}$, which is the unit cap since
the row-softmax row sums to one, so $\sum_ja^{(1)}_{ij}\le\sum_jA^{\mathrm{sm}}_{ij}$;""",
))

PATCHES.append((
    "5 tab:verification row",
    r"""Prop.~\ref{prop:twostep}(iii), sub-unit rows & $\sum_j A_{ij}\le1$; $69.3\%$ spend less & 0 / 80741 \\""",
    r"""Prop.~\ref{prop:twostep}(iii), sub-unit rows & $\sum_j A_{ij}\le\sum_jA^{\mathrm{sm}}_{ij}$; $69.3\%$ spend less & 0 / 80741 \\""",
))

PATCHES.append((
    "6 App. D limitations",
    r"""\textbf{Row mass is bounded above by one but not below.}
$\sum_jA_{ij}\le1$ (Prop.~\ref{prop:twostep}(iii)), so no query ever absorbs more than the unit
softmax spends""",
    r"""\textbf{Row mass is bounded above by the row's own softmax mass, and not below.}
$\sum_jA_{ij}\le\sum_jA^{\mathrm{sm}}_{ij}$ (Prop.~\ref{prop:twostep}(iii)), which is one unit in
exact arithmetic, so no query ever absorbs more than the unit softmax spends""",
))

PATCHES.append((
    "7 App. F numerics, the device statistic",
    r"""it; total runtime is under $25$ seconds on one CPU core and the only dependency is
\texttt{numpy}. The last block checks App.~\ref{app:selective} and is reported there.""",
    r"""it; total runtime is under $25$ seconds on one CPU core and the only dependency is
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
total.""",
))

PATCHES.append((
    "8 App. E cost (c)",
    r"""\eqref{eq:rowstep1} caps the \emph{whole} row at one unit before any restriction is applied, and
step 2 conserves the relation's quota within it, so the row total never exceeds one and the
complement keeps its step-1 value.""",
    r"""\eqref{eq:rowstep1} caps the \emph{whole} row at the unit it already carries, before any
restriction is applied, and step 2 conserves the relation's quota within it, so the row total
never exceeds its own softmax mass and the complement keeps its step-1 value.""",
))


def main(argv):
    dry = "--dry-run" in argv
    args = [a for a in argv[1:] if not a.startswith("--")]
    path = args[0] if args else "marsea_iclr.tex"

    if not os.path.isfile(path):
        print("ERROR: no such file: %s" % path)
        return 2

    with open(path, encoding="utf-8") as fh:
        src = fh.read()

    # ---- phase 0a: PREREQUISITE guard (record §60.1) -------------------------
    # The eight `old` strings below also match the 2026-09-08 draft, which is
    # missing D-31 and D-9a. Patching THAT file succeeds and yields a draft with
    # the numerical fix and without two standing decisions -- the near-miss of
    # 2026-09-15. So check the base is the current draft before touching it.
    missing = [label for label, probe in PREREQUISITES if probe not in src]
    if missing:
        print("REFUSING TO WRITE -- this is not the current draft. Missing:")
        for label in missing:
            print("    %s" % label)
        print("""
These patches WOULD apply cleanly here, and that is the trap (record §60.1): you
would get the unit-cap fix on a draft that predates D-31 and D-9a.

Forward-port first, then come back:

    python3 portforward_d9a_d31.py --dry-run %s
    python3 portforward_d9a_d31.py           %s
""" % (path, path))
        return 1

    # ---- phase 0b: double-apply guard, checked next so the message is right --
    already = [name for name, _old, new in PATCHES if new in src]
    if already:
        print("REFUSING TO WRITE -- these patches are applied already:")
        for name in already:
            print("    %s" % name)
        print("\nNothing to do. (If you meant to re-apply from scratch, restore "
              "the .bak first.)")
        return 1

    # ---- phase 1: assert every old string occurs exactly once ----------------
    bad = []
    for name, old, _new in PATCHES:
        n = src.count(old)
        if n != 1:
            bad.append((name, n))
        print("  %-42s old occurrences: %d" % (name, n))
    if bad:
        print("\nREFUSING TO WRITE -- %d patch(es) did not match exactly once:" % len(bad))
        for name, n in bad:
            print("    %s  (count=%d)" % (name, n))
        print("\nThe .tex has drifted from the version these patches were written "
              "against (9bacef9, 2026-09-15). Re-derive the offending `old` "
              "strings from the current file before re-running.")
        return 1

    if src.count(r"\label{eq:excess}") != 0:
        print("\nREFUSING TO WRITE -- label eq:excess already exists in the file.")
        return 1

    if dry:
        print("\nDry run: all %d patches match exactly once. Nothing written." % len(PATCHES))
        return 0

    # ---- phase 2: apply -----------------------------------------------------
    out = src
    for _name, old, new in PATCHES:
        out = out.replace(old, new, 1)

    shutil.copyfile(path, path + ".bak")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(out)

    # ---- phase 3: verify ----------------------------------------------------
    with open(path, encoding="utf-8") as fh:
        chk = fh.read()
    errs = []
    for name, old, new in PATCHES:
        if chk.count(new) != 1:
            errs.append("%s: new string occurs %d times" % (name, chk.count(new)))
        # Patches 2 and 7 APPEND: their `new` begins with their `old`, so the
        # old text is legitimately still present. Only demand its removal for
        # patches that genuinely replace.
        if old not in new and chk.count(old) != 0:
            errs.append("%s: old string still present" % name)
    if chk.count(r"\label{eq:excess}") != 1:
        errs.append("label eq:excess occurs %d times" % chk.count(r"\label{eq:excess}"))

    print("\nWrote %s (backup at %s.bak)" % (path, path))
    if errs:
        print("POST-CHECK FAILED:")
        for e in errs:
            print("    " + e)
        print("Restore with: mv %s.bak %s" % (path, path))
        return 1

    print("POST-CHECK OK: %d patches applied, each new string unique, "
          "eq:excess unique." % len(PATCHES))
    print("""
Next, and none of it is optional:
  1. verify_all.py -- change the Prop. twostep(iii) row label from the
     '<= 1' form to '<= sum_j A^sm_ij'. The reproducibility statement claims the
     harness reproduces every figure in tab:verification; it becomes false
     otherwise. The denominator (80,741) is unchanged.
  2. pdflatex x3 (the .bbl is present, citations unchanged).
  3. Confirm 0 errors, 0 undefined refs, 0 undefined citations, 0 overfull
     boxes, 0 multiply-defined labels -- Patch 2 and Patch 7 each add a
     paragraph, so check the 9-page main-text limit still holds (Patch 2 is in
     Sec. 3; Patch 7 is in an appendix and cannot cost main-text pages).
  4. Read the changed pages in the RENDERED PDF, not only in the .tex.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
