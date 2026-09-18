#!/usr/bin/env python3
"""
patch_iclr_window.py -- the guard-window amendment to the unit-cap remark.

WHY
---
`patch_iclr_cap.py` (record §59.4, applied §60.2) left the finite-precision remark saying
that "(iii), (v) and (vi) hold on any kernel". Code drop **e6ca44f** measured the one thing
that sentence had not accounted for: the theta guard inside `proj_le_masked` has a WINDOW.
A row whose excess over its softmax mass is positive but smaller than the fp32 prefix scan's
own error is left uncapped, so the *realized* row total can sit that far above
`sum_j A^sm_ij`. Measured: ~1e-7 on the reference CPU (I reproduce 8.8e-8 at 8K, 1.1e-7 at
16K), 1.06e-6 at 8K and 1.95e-6 at 16K on the deployment GPU -- above CAP_TOL.

So the honest split is:
  * the binding DECISION is kernel-free -- that part of the remark is right and is the point
    of the excess criterion;
  * (v) zeros never resurrected and (vi) empty relation == standard attention hold EXACTLY
    (excess is bitwise 0.0 on an empty relation -- verified);
  * (iii) is a theorem about the exact program. The implementation meets it up to the window.

Prop. twostep(iii) itself does NOT change: it is a statement about the program, proved in
App. E, and it is true. Only the finite-precision remark and App. F's device statistics move.

Protocol as ever (record §0.5 and §60.6): identity probes first, then every `old` string
asserted to occur exactly once before anything is written, then each `new` verified unique.

    python3 patch_iclr_window.py --dry-run marsea_iclr.tex
    python3 patch_iclr_window.py           marsea_iclr.tex
"""

import sys
import os
import shutil

# The base must already carry D-31, D-9a AND the cap patches (§60.6's lesson: assert you are
# editing the right FILE, not just the right place).
PREREQUISITES = [
    ("D-31, the attention-sink exclusion (Sec. 2.3)",
     "One position is excluded from every relation by construction"),
    ("D-9a, the coreference column (App. H)", r"\emph{coreference column}"),
    ("the cap patches (patch_iclr_cap.py)", r"\label{eq:excess}"),
]

PATCHES = []

PATCHES.append((
    "A  the finite-precision remark, decision vs realized total",
    r"""exactly --- a difference of in-relation terms that reads no row total. The implementation binds
\eqref{eq:rowstep1} on \eqref{eq:excess} and projects a binding row onto $\sum_jA^{\mathrm{sm}}_{ij}$,
so (iii), (v) and (vi) hold on any kernel and an empty relation reproduces standard attention
\emph{bitwise} rather than to a tolerance. App.~\ref{app:numerics} reports the kernel's own
$\delta$ as a device statistic beside the checks rather than inside them.""",
    r"""exactly --- a difference of in-relation terms that reads no row total. The implementation binds
\eqref{eq:rowstep1} on \eqref{eq:excess} and projects a binding row onto $\sum_jA^{\mathrm{sm}}_{ij}$,
so the binding \emph{decision} is a function of in-relation terms alone on every kernel, (v) and
(vi) hold \emph{exactly} --- an empty relation has excess bitwise $0.0$ and so reproduces standard
attention bitwise rather than to a tolerance. \textbf{What finite precision still costs is a
window, not a constant.} A row whose excess is positive but smaller than the projection's own
arithmetic can see is left uncapped, so the \emph{realized} row total may exceed
$\sum_jA^{\mathrm{sm}}_{ij}$ by that much: $10^{-7}$ on our reference CPU and $2\times10^{-6}$ at
$n_k=16{,}384$ on the deployment GPU, both measured. (iii) is a statement about the program and is
exact; the implementation meets it up to that window, which App.~\ref{app:numerics} reports and
which the release gates against the tolerance its invariants allow. App.~\ref{app:numerics}
likewise reports the kernel's own $\delta$ as a device statistic beside the checks rather than
inside them.""",
))

PATCHES.append((
    "B  App. F device statistics, the window beside delta",
    r"""to be made so. On the reference CPU, $\delta$ reaches $3.0\times10^{-5}$ at $n_k=8192$ and
$6.0\times10^{-5}$ at $16{,}384$, against $\approx10^{-6}$ for the pairwise reduction of the
deployment GPU --- a spread no fixed tolerance spans, which is why the criterion reads no row
total.""",
    r"""to be made so. On the reference CPU, $\delta$ reaches $3.0\times10^{-5}$ at $n_k=8192$ and
$6.0\times10^{-5}$ at $16{,}384$, against $\approx10^{-6}$ for the pairwise reduction of the
deployment GPU --- a spread no fixed tolerance spans, which is why the criterion reads no row
total.

\textbf{A second device statistic, and it is the one that bounds what the implementation
delivers.} The projection that enforces the cap is itself computed in single precision, so it
cannot act on an excess smaller than its own arithmetic resolves; a row inside that
\emph{window} is left uncapped and keeps its excess. The window is a property of the device's
prefix reduction exactly as $\delta$ is a property of its softmax reduction: $1\times10^{-7}$ on
the reference CPU, $1.1\times10^{-6}$ at $n_k=8192$ and $2.0\times10^{-6}$ at $16{,}384$ on the
deployment GPU, and as much as $n_k\varepsilon/4$ on a strictly sequential fp32 scan. We measure
it by bisection on the target device before any run and refuse to start if it exceeds what the
invariants allow, so the statement the release makes is
$\sum_jA_{ij}\le\sum_jA^{\mathrm{sm}}_{ij}+w$ with $w$ measured and reported --- never that the
binding set is device-independent inside the window. \emph{Reporting both numbers is the point}:
a mechanism whose guarantee is checked in floating point owes the reader the size of the gap
between the program and the arithmetic that runs it.""",
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

    missing = [label for label, probe in PREREQUISITES if probe not in src]
    if missing:
        print("REFUSING TO WRITE -- this is not the current, cap-patched draft. Missing:")
        for label in missing:
            print("    %s" % label)
        print("\nRun portforward_d9a_d31.py and patch_iclr_cap.py first (record §0.3b, §60.6).")
        return 1

    already = [name for name, _o, new in PATCHES if new in src]
    if already:
        print("REFUSING TO WRITE -- already applied:")
        for name in already:
            print("    %s" % name)
        return 1

    bad = []
    for name, old, _new in PATCHES:
        n = src.count(old)
        print("  %-48s old occurrences: %d" % (name, n))
        if n != 1:
            bad.append((name, n))
    if bad:
        print("\nREFUSING TO WRITE -- %d patch(es) did not match exactly once." % len(bad))
        return 1

    if dry:
        print("\nDry run: both amendments match exactly once. Nothing written.")
        return 0

    out = src
    for _n, old, new in PATCHES:
        out = out.replace(old, new, 1)
    shutil.copyfile(path, path + ".bak-window")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(out)

    chk = open(path, encoding="utf-8").read()
    errs = [("%s: new occurs %d times" % (n, chk.count(new)))
            for n, _o, new in PATCHES if chk.count(new) != 1]
    if "hold on any kernel" in chk:
        errs.append("the superseded 'hold on any kernel' clause is still present")
    print("\nWrote %s (backup at %s.bak-window)" % (path, path))
    if errs:
        for e in errs:
            print("    " + e)
        print("Restore with: mv %s.bak-window %s" % (path, path))
        return 1
    print("POST-CHECK OK. Rebuild (pdflatex x3) and check the page count: patch B adds a "
          "paragraph to an appendix, patch A lengthens a Sec. 3 remark.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
