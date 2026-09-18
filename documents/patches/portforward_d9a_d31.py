#!/usr/bin/env python3
"""
portforward_d9a_d31.py -- bring a 2026-09-08 marsea_iclr.tex up to the 2026-09-10 draft.

WHY THIS EXISTS
---------------
Two copies of the ICLR draft drifted apart:

  * the Claude project's `claude/marsea_iclr.tex`, 2026-09-10 -- current;
  * `~/Dev/marsea/documents/marsea_iclr.tex`, 2026-09-08 -- two decisions behind.

The master record is the change log and names exactly two paper edits in that window,
and nothing else:

  * record §52 (D-31, 2026-09-09): "Paper Sec. 2.3 sentence added" -- position 0, the
    attention sink, excluded from every relation.
  * record §53 (D-9a, 2026-09-10): "Paper App. H E7 paragraph rewritten (two column
    kinds; why they are not merged; S0 described as the full-head sweep with the sink
    excluded; the column test runs at the head that routes it, in a patched layer)."

Independently confirmed by probing the 2026-09-08 file: the §47 change was offered and
never applied (absent from both, correctly), and all three verification rows added by
record §48 on 2026-09-08 ARE present -- so the local file is current through §48 and the
gap is exactly D-31 and D-9a.

The replacement text below is the 2026-09-10 project copy's, verbatim.

Run this BEFORE patch_iclr_cap.py:

    python3 portforward_d9a_d31.py --dry-run marsea_iclr.tex
    python3 portforward_d9a_d31.py           marsea_iclr.tex
    python3 patch_iclr_cap.py     --dry-run  marsea_iclr.tex
    python3 patch_iclr_cap.py                marsea_iclr.tex

Same protocol as patch_iclr_cap.py (record §0.5): every `old` string is asserted to occur
exactly once BEFORE anything is written; a file that already carries the new text is
refused; the original is kept at <file>.bak-preport; and every `new` string is verified
unique afterwards.
"""

import sys
import os
import shutil

PATCHES = []

# ---------------------------------------------------------------------------
# D-31 (record §52) -- Sec. 2.3, the attention sink is in no relation.
# The 2026-09-08 file ends the subsection at the cost sentence.
# ---------------------------------------------------------------------------
PATCHES.append((
    "D-31 sink sentence (Sec. 2.3)",
    r"""it. Cost is unchanged, since $e_{ij}$ is a rank-$r$ inner product computed alongside the scores.
""",
    r"""it. Cost is unchanged, since $e_{ij}$ is a rank-$r$ inner product computed alongside the scores.
One position is excluded from every relation by construction: the first token of a decoder is
the attention sink, not a candidate for anything, and a query whose only visible key re-shapes
its quota away from it would be left with an empty row --- so $e_{i0}=e_{0j}=0$, and the sink's
row and column stay at standard attention.
""",
))

# ---------------------------------------------------------------------------
# D-9a (record §53) -- App. H, "E7, the column ground truth", rewritten.
# The old paragraph merges the coreference and value targets into one T_j and
# reports m_j = 2m on RULER; S0's full-head sweep showed answer positions route
# to values, not to the first mention, so the merged target set would score the
# coreference column against a target the backbone never routes to it.
# ---------------------------------------------------------------------------
PATCHES.append((
    "D-9a E7 column ground truth (App. H)",
    r"""\paragraph{E7, the column ground truth.} No benchmark annotates $T_j$, the set of queries that
\emph{should} attend to a key, so E7 derives it by exact string match from annotation the
benchmarks do ship, on both instruments alike. A key is the first token of the \emph{first
mention} of a bridge string; $T_j$ is the first token of every \emph{later} mention of the same
string together with the answer positions that emit its value. On RULER's multi-value task the
$m$ needles share one key phrase, so its first mention has $2m$ later mentions (the other
needles, the question, the $m$ answer positions) and $m_j=2m$ moves with the sweep; on MuSiQue the
bridge strings are the decomposition's intermediate answers, giving a leaf passage $m_j=2$ and a
bridging one $3$--$5$, and because we assemble the prompt the gold passages are placed in hop
order so that every later mention is visible to the earlier one. Reversing that order changes
$m_j$ deterministically, which is a free check that the annotation measures what we say. Two
things are kept separate from it. First, a target the relation excluded is a \emph{relation
recall} miss, logged on its own and never charged to $\tau_j$; the interval is computed on
$T_j\cap\mathcal{E}_{\cdot j}$ against the relation's complement, and a column whose complement
is empty has $\delta_j=+\infty$ and a lower endpoint of $0$. Second, the construction is ground
truth about the \emph{task} and a hypothesis about \emph{routing}, so S0 first checks on the
unmodified decoder that the retrieval head actually sends the later mentions to the earlier one
--- the mass those queries put on the key exceeds half the row's maximum on at least $80\%$ of
examples --- and a construction that fails the check is reported as not measurable on this
backbone rather than used. Dense arms have no relation, so their column precision is reported on
two domains, the full visible column ($m_j/n$) and MarSea's paired relation
($m_j/|\mathcal{E}_{\cdot j}|$, the identity of Prop.~\ref{prop:fullsupportceiling}); the second
is the headline.
""",
    r"""\paragraph{E7, the column ground truth.} No benchmark annotates $T_j$, the set of queries that
\emph{should} attend to a key, so E7 derives it by exact string match from annotation the
benchmarks do ship, on both instruments alike, and it uses two kinds of column because the
backbone routes them through different heads. The \emph{coreference column}: a key is the first
token of the \emph{first mention} of a bridge string, and $T_j$ is the first token of every
\emph{later mention} of the same string. On RULER's multi-value task the $m$ needles share one
key phrase, so its first mention has $m+1$ later mentions (the other $m-1$ needles, the
question, and the answer prefix) and $m_j=m+1$ moves with the sweep; on MuSiQue the bridge
strings are the decomposition's intermediate answers, giving a leaf passage $m_j=2$ and a
bridging one $3$--$5$, and because we assemble the prompt the gold passages are placed in hop
order so that every later mention is visible to the earlier one. Reversing that order changes
$m_j$ deterministically, which is a free check that the annotation measures what we say. The
\emph{value column}: a key is a needle's value span and $T_j$ is the answer positions that emit
that value, so $m_j$ is the number of emitting tokens, typically one; this is the column the
answer rows are scored on, and it is the $m_j\le1$ regime of Sec.~\ref{sec:method-fanout}. The
two are not merged into one target set: S0 found that answer positions attend to values and not
to the key phrase's first mention, so a merged $T_j$ would score the coreference column against
a target the backbone never routes to it. Two things are kept separate from both. First, a
target the relation excluded is a \emph{relation recall} miss, logged on its own and never
charged to $\tau_j$; the interval is computed on $T_j\cap\mathcal{E}_{\cdot j}$ against the
relation's complement, and a column whose complement is empty has $\delta_j=+\infty$ and a lower
endpoint of $0$. Second, the construction is ground truth about the \emph{task} and a hypothesis
about \emph{routing}, so S0 first checks on the unmodified decoder, sweeping every head, that
some head sends the later mentions to the earlier one --- the mass those queries put on the key
span exceeds half the row's maximum over candidate keys, the sink excluded, on at least $80\%$
of examples --- and a construction that fails the check is reported as not measurable on this
backbone rather than used; the column test runs at the head that routes it, which must be in a
patched layer. Dense arms have no relation, so their column precision is reported on
two domains, the full visible column ($m_j/n$) and MarSea's paired relation
($m_j/|\mathcal{E}_{\cdot j}|$, the identity of Prop.~\ref{prop:fullsupportceiling}); the second
is the headline.
""",
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

    already = [name for name, _o, new in PATCHES if new in src]
    if already:
        print("REFUSING TO WRITE -- already forward-ported:")
        for name in already:
            print("    %s" % name)
        print("\nThis file is already at the 2026-09-10 draft. Nothing to do.")
        return 1

    bad = []
    for name, old, _new in PATCHES:
        n = src.count(old)
        print("  %-38s old occurrences: %d" % (name, n))
        if n != 1:
            bad.append((name, n))
    if bad:
        print("\nREFUSING TO WRITE -- %d patch(es) did not match exactly once:" % len(bad))
        for name, n in bad:
            print("    %s  (count=%d)" % (name, n))
        print("\nThis file is not the 2026-09-08 draft these ports were derived from.")
        return 1

    if dry:
        print("\nDry run: both ports match exactly once. Nothing written.")
        return 0

    out = src
    for _name, old, new in PATCHES:
        out = out.replace(old, new, 1)

    shutil.copyfile(path, path + ".bak-preport")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(out)

    with open(path, encoding="utf-8") as fh:
        chk = fh.read()
    errs = []
    for name, old, new in PATCHES:
        if chk.count(new) != 1:
            errs.append("%s: new text occurs %d times" % (name, chk.count(new)))
        if old not in new and chk.count(old) != 0:
            errs.append("%s: old text still present" % name)
    # the two claims these ports exist to put in the paper
    for probe, label in [
        ("One position is excluded from every relation by construction", "D-31 sink sentence"),
        (r"\emph{coreference column}", "D-9a coreference column"),
        (r"\emph{value column}", "D-9a value column"),
    ]:
        if chk.count(probe) != 1:
            errs.append("%s: probe %r occurs %d times" % (label, probe, chk.count(probe)))
    for gone, label in [
        ("together with the answer positions that emit its value", "merged T_j"),
        ("$m_j=2m$ moves with the sweep", "m_j = 2m"),
    ]:
        if chk.count(gone) != 0:
            errs.append("%s: superseded text still present (%r)" % (label, gone))

    print("\nWrote %s (backup at %s.bak-preport)" % (path, path))
    if errs:
        print("POST-CHECK FAILED:")
        for e in errs:
            print("    " + e)
        print("Restore with: mv %s.bak-preport %s" % (path, path))
        return 1

    print("POST-CHECK OK: both ports applied; D-31 and D-9a present; the merged target")
    print("set and the m_j = 2m claim are gone.")
    print("\nThis file should now equal the 2026-09-10 project draft. Next: patch_iclr_cap.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
