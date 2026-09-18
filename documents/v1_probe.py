"""
E1, the replication probe, rebuilt for the two-step selective form.

WHAT CHANGED AND WHY.  In the two-step form the fan-out capacity is *inherited*:
step 1 fixes the column at standard attention's mass and step 2 re-shapes only the
exclusive set, conserving its quota.  So fan-out MASS is no longer the bounded
quantity and a mass sweep is the wrong instrument -- an earlier version of this
figure measured mass and would have shown MarSea tracking softmax.  What the
mechanism bounds is fan-out DEGREE: how many queries key j actually reaches
inside its relation.  Both panels measure that instead.

Panel (a): the exclusive set GROWS.  |E_.j| candidates compete for key j and only
m_j of them are true.  Query-local schemes with non-negative weights must reach
all of them (Cor. 2, cardinality half): softmax's support is |E_.j| and its
precision m_j/|E_.j|, an identity, not a tendency (Prop. 14).  MarSea's support is
m_j at every |E_.j|, because tau_j from Thm. 2's interval terminates the tail.
This is the claim the paper lives on and it involves no n.

Panel (b): does the statistic the exclusivity head reads carry the answer?
tau_j must land so that k*(tau_j) = m_j, and nu_j = ||p_.j||_2^-2 is a smooth
estimate of exactly that, so tau^K reads it.  Whether nu_j tracks the true m_j is
NOT an identity -- it depends on tau_j and on the score column -- so it is
measurable.  Under a per-key tau_j inside its interval it tracks; under one global
tau it saturates, which is Cor. 3 in training-free form and the reason a
uniform-exclusivity null cannot supply the head with a usable input.

numpy + matplotlib.  Deterministic.
        python3 v1_probe.py            # writes v1_probe.pdf, prints the table
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = 12
MJ    = 4                       # the true relation size, held fixed
SIZES = np.array([4, 8, 16, 32, 64, 128, 256])      # |E_.j|, the candidate set
TAU_G = 1.2                     # the single global exclusivity used as the null in (b)


def sparsemax(z):
    z = np.asarray(z, float); u = np.sort(z)[::-1]; c = np.cumsum(u) - 1.0
    k = np.arange(1, len(z) + 1); cond = u - c / k > 0; r = k[cond][-1]
    return np.maximum(z - c[cond][-1] / r, 0.0)


def interval(s, T, m):
    """Thm. 2's sharp endpoints on the index set s is drawn over."""
    smin = min(s[i] for i in T)
    W = float(sum(s[i] - smin for i in T))
    out = [s[i] for i in range(len(s)) if i not in T]
    delta = smin - (max(out) if out else -1.0)
    lo = 1.0 / (W + m * delta)
    hi = (1.0 / W) if W > 0 else np.inf
    return lo, hi, delta


# ------------------------------------------------- panel (a): degree vs |E_.j|
rows = {k: np.zeros((SEEDS, len(SIZES))) for k in
        ["MarSea, support size", "MarSea, precision",
         "softmax, support size", "softmax, precision"]}
for sd in range(SEEDS):
    rng = np.random.default_rng(400 + sd)
    for a, n in enumerate(SIZES):
        s = rng.uniform(0, 1, size=n) * 0.9         # plausible distractors
        s[:MJ] = 1.0 + rng.random(MJ)               # the m_j true members
        T = set(range(MJ))
        lo, hi, _ = interval(s, T, MJ)
        tau = lo + 0.5 * (min(hi, lo * 20.0) - lo)  # tau_j inside Thm. 2's interval
        S = set(np.nonzero(sparsemax(tau * s))[0].tolist())
        rows["MarSea, support size"][sd, a] = len(S)
        rows["MarSea, precision"][sd, a] = len(S & T) / len(S)
        rows["softmax, support size"][sd, a] = n     # full support, every temperature
        rows["softmax, precision"][sd, a] = MJ / n

print(f"panel (a): m_j = {MJ} fixed, the candidate set |E_.j| grows "
      f"({SEEDS} seeds)\n")
print(f"{'|E_.j|':>8}{'MarSea |S|':>12}{'MarSea P_j':>12}{'softmax |S|':>14}{'softmax P_j':>13}")
for a, n in enumerate(SIZES):
    print(f"{n:>8}{rows['MarSea, support size'].mean(0)[a]:>12.2f}"
          f"{rows['MarSea, precision'].mean(0)[a]:>12.3f}"
          f"{rows['softmax, support size'].mean(0)[a]:>14.0f}"
          f"{rows['softmax, precision'].mean(0)[a]:>13.3f}")
slope = np.polyfit(SIZES, rows["MarSea, support size"].mean(0), 1)[0]
print(f"\n   MarSea support-size slope in |E_.j|: {slope:+.4f}   (softmax's is +1.0000)")

# --------------------------- panel (b): does nu_j carry the relation's size?
mjs = np.array([1, 2, 4, 8, 16, 32])
NQ = int(mjs.sum())
KEYS = ["per-key $\\tau_j$ (in its interval)", f"one global $\\tau={TAU_G}$"]
panel_b = {k: np.zeros((SEEDS, len(mjs))) for k in KEYS}
n_bad = 0                                   # columns where the draw violated Assumption 2
for sd in range(SEEDS):
    rng = np.random.default_rng(2000 + sd)
    S = 0.6 * rng.normal(size=(NQ, len(mjs)))
    o = 0
    for b, mj in enumerate(mjs):
        S[o:o + mj, b] = 2.0 + 0.25 * rng.random(mj); o += mj
    o = 0
    for b, mj in enumerate(mjs):
        T = set(range(o, o + mj)); o += mj
        col = S[:, b]
        lo, hi, delta = interval(col, T, mj)
        if delta <= 0:                      # Assumption 2 fails on this column: no valid interval,
            n_bad += 1                      # and lo can be NEGATIVE.  Redraw rather than report it.
            col = S[:, b] = np.where(np.isin(np.arange(NQ), list(T)), col, np.minimum(col, 1.9))
            lo, hi, delta = interval(col, T, mj)
        for k, tau in [(KEYS[0], lo + 0.5 * (min(hi, lo * 20) - lo)), (KEYS[1], TAU_G)]:
            p = sparsemax(tau * col)
            panel_b[k][sd, b] = 1.0 / np.sum(p ** 2)      # nu_j = ||p||_2^-2
print(f"\npanel (b), n_q = {NQ}, true m_j = {list(mjs)}  "
      f"({n_bad} of {SEEDS*len(mjs)} columns violated Assn. 2 on the first draw and were re-clipped):")
for k in KEYS:
    print(f"   nu_j  {k:<34s} " + "  ".join(f"{x:6.2f}" for x in panel_b[k].mean(0)))

# ----------------------------------------------------------------- the figure
ACC, OR, NEUT = "#2a78d6", "#eb6834", "#52514e"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Termes", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2, "pdf.fonttype": 42,
})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.2, 2.05))

ax.plot(SIZES, rows["softmax, support size"].mean(0), "-", color=OR, lw=1.3,
        marker="o", ms=3.0, mec="white", mew=0.4, label="softmax / MESH / any dense scheme")
ax.plot(SIZES, rows["MarSea, support size"].mean(0), "-", color=ACC, lw=1.5,
        marker="o", ms=3.0, mec="white", mew=0.4, label="MarSea")
ax.axhline(MJ, color=NEUT, ls=":", lw=0.8, zorder=1)
ax.text(SIZES[-1], MJ * 1.35, f"true $m_j={MJ}$", fontsize=5.8, color=NEUT, ha="right")
ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
ax.set_xticks(SIZES); ax.set_xticklabels([str(v) for v in SIZES])
ax.set_xlabel("candidates in the relation $|\\mathcal{E}_{\\cdot j}|$", fontsize=7, labelpad=1.5)
ax.set_ylabel("fan-out degree $|\\mathrm{supp}(A_{\\cdot j})|$", fontsize=7, labelpad=1.5)
ax.set_title("(a) degree; the comparator's precision is an identity", fontsize=7, pad=3)
ax.legend(fontsize=5.8, frameon=False, loc="upper left", handlelength=1.6,
          labelspacing=0.3, borderpad=0.1)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.tick_params(labelsize=6.4, pad=1.5)
axt = ax.twinx()
axt.plot(SIZES, rows["softmax, precision"].mean(0), "--", color=OR, lw=0.9, alpha=0.75)
axt.plot(SIZES, rows["MarSea, precision"].mean(0), "--", color=ACC, lw=0.9, alpha=0.75)
axt.set_ylim(-0.03, 1.08); axt.set_ylabel("precision $P_j$ (dashed)", fontsize=6.4, labelpad=1.5)
axt.tick_params(labelsize=6.0, pad=1.0); axt.spines["top"].set_visible(False)

w = 0.34; xs = np.arange(len(mjs))
for a, k in enumerate(KEYS):
    v = panel_b[k]
    bx.bar(xs + (a - 0.5) * w, v.mean(0), w * 0.9, color=(ACC if a == 0 else NEUT),
           linewidth=0, zorder=3, label=k, yerr=v.std(0),
           error_kw=dict(lw=0.6, ecolor="#0b0b0b", capsize=1.2))
bx.plot(xs, mjs, "k:", lw=0.9, zorder=4)
bx.text(len(mjs) - 1.15, mjs[-1] * 1.03, "true $m_j$", fontsize=6.4, ha="right", va="bottom")
bx.set_xticks(xs); bx.set_xticklabels([str(v) for v in mjs])
bx.set_xlabel("the key's true relation size $m_j$", fontsize=7, labelpad=1.5)
bx.set_ylabel("effective cardinality $\\nu_j$", fontsize=7, labelpad=1.5)
bx.set_title("(b) does $\\nu_j$ carry $m_j$?", fontsize=7, pad=3)
bx.legend(fontsize=5.6, frameon=False, loc="upper left", handlelength=1.2,
          labelspacing=0.28, borderpad=0.1)
for sp in ("top", "right"): bx.spines[sp].set_visible(False)
bx.tick_params(labelsize=6.4, pad=1.5)

fig.tight_layout(pad=0.2, w_pad=1.4)
fig.savefig("v1_probe.pdf", bbox_inches="tight")
print("\n-> v1_probe.pdf")
