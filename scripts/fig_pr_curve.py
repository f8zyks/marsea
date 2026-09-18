"""
Figure 2: the precision-recall trace of the fan-out program as the exclusivity
tau_j sweeps, against the point a full-support mechanism occupies.

Nothing here is drawn by hand.  The trace is obtained by running the Stage-1
program on one planted column at 400 temperatures and reading off the realized
support, and the comparison point by running a column-softmax on the same column
at the same temperatures.  The assertions below check the three regimes of
eq. (21) exactly, not merely that P = 1 or R = 1.
Requires numpy and matplotlib.  Deterministic.
        python3 pr_curve.py            # writes pr_curve.pdf
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ the program
def sparsemax(z):
    n = len(z); zs = np.sort(z)[::-1]; cs = np.cumsum(zs)
    k = np.arange(1, n + 1)
    kk = k[(1 + k * zs) > cs][-1]
    return np.maximum(z - (cs[kk - 1] - 1.0) / kk, 0.0)

# ------------------------------------------------------------------ one column
rng = np.random.default_rng(0)
nq, m = 64, 4
s = np.empty(nq)
s[:m] = np.linspace(1.0, 0.4, m)            # targets, with within-set spread
s[m:] = -0.3 - rng.random(nq - m)           # non-targets, strictly below
T = set(range(m))
delta = s[:m].min() - s[m:].max()
W = float(np.sum(s[:m] - s[:m].min()))
lo, hi = 1.0 / (W + m * delta), 1.0 / W     # Thm. 2's sharp endpoints

taus = np.exp(np.linspace(np.log(lo) - 2.6, np.log(hi) + 2.2, 400))
P, R, K = [], [], []
Pb, Rb = [], []                              # the column-softmax comparison point
for tau in taus:
    S = set(np.nonzero(sparsemax(tau * s))[0].tolist())
    P.append(len(S & T) / len(S)); R.append(len(S & T) / m); K.append(len(S))
    q = np.exp(tau * s - (tau * s).max()); q /= q.sum()
    Sb = set(np.nonzero(q)[0].tolist())
    Pb.append(len(Sb & T) / len(Sb)); Rb.append(len(Sb & T) / m)
P, R, K = np.array(P), np.array(R), np.array(K, float)
Pb, Rb = np.array(Pb), np.array(Rb)
inside = (taus >= lo) & (taus < hi)
below, above = taus < lo, taus >= hi
assert np.all(np.abs(P[inside] - 1) < 1e-12) and np.all(np.abs(R[inside] - 1) < 1e-12)
assert np.all(R[below] == 1.0) and np.allclose(P[below], m / K[below])   # (1, m/k*)
assert np.all(P[above] == 1.0) and np.allclose(R[above], K[above] / m)   # (k*/m, 1)
assert np.all((np.abs(P - 1) < 1e-12) | (np.abs(R - 1) < 1e-12))         # no interior point
assert np.allclose(Rb, 1.0) and np.allclose(Pb, m / nq)   # softmax: one point, every tau
b_R, b_P = float(Rb[0]), float(Pb[0])

# ------------------------------------------------------------------ the figure
ACCENT, NEUTRAL, INK = "#2a78d6", "#52514e", "#0b0b0b"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Termes", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 6.2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2, "pdf.fonttype": 42,
})
fig, ax = plt.subplots(figsize=(1.62, 1.46))
ax.plot(R, P, color=ACCENT, lw=1.3, solid_capstyle="round", zorder=3)
ax.plot([1], [1], "o", color=ACCENT, ms=3.6, mec="white", mew=0.7, zorder=4)
ax.plot([b_R], [b_P], "s", color=NEUTRAL, ms=3.4, mec="white", mew=1.1, zorder=6)

ax.annotate("$P_j\\!=\\!R_j\\!=\\!1$", xy=(1, 1), xytext=(0.60, 0.885),
            color=ACCENT, fontsize=5.8, ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.5,
                            shrinkA=2, shrinkB=4))
ax.annotate("full support,\nevery $\\tau_j$", xy=(b_R, b_P), xytext=(0.58, 0.155),
            color=NEUTRAL, fontsize=5.8, ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color=NEUTRAL, lw=0.5,
                            shrinkA=2, shrinkB=4))
ax.text(0.615, 1.055, "$\\tau_j$ large", color=INK, fontsize=5.8,
        ha="center", va="bottom")
ax.text(1.045, 0.52, "$\\tau_j$ small", color=INK, fontsize=5.8,
        ha="center", va="center", rotation=90)

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.set_xlim(0.0, 1.14); ax.set_ylim(-0.04, 1.17)
ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
ax.set_xticklabels(["0", "0.5", "1"]); ax.set_yticklabels(["0", "0.5", "1"])
ax.set_xlabel("support recall $R_j$", fontsize=6.2, labelpad=1.0)
ax.set_ylabel("support precision $P_j$", fontsize=6.2, labelpad=1.0)
ax.spines["left"].set_bounds(0, 1); ax.spines["bottom"].set_bounds(0, 1)
ax.tick_params(labelsize=5.8, pad=1.0)
fig.tight_layout(pad=0.12)
fig.savefig("pr_curve.pdf")
print(f"nq={nq} m={m} delta={delta:.3f} W={W:.3f} "
      f"interval=[{lo:.4f},{hi:.4f}) softmax point=({b_R:.4f},{b_P:.4f}) -> pr_curve.pdf")
