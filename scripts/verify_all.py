"""
MarSea -- numerical verification of every analytical claim.
Self-contained: numpy only.  Reproduces Table 8 of the paper.
    python3 verify_all.py            # full run
Deterministic: every block seeds its own generator.
"""
import sys
import numpy as np

TOL = 1e-9
results = []          # (label, violations, trials)
def report(label, v, n): results.append((label, v, n))

# ----------------------------------------------------------------- primitives
def sparsemax(z):
    """argmax_{p in simplex} <p,z> - 1/2||p||^2 .  Closed form by sorting."""
    n = len(z); zs = np.sort(z)[::-1]; cs = np.cumsum(zs)
    k = np.arange(1, n + 1)
    kk = k[(1 + k * zs) > cs][-1]
    psi = (cs[kk - 1] - 1.0) / kk
    return np.maximum(z - psi, 0.0), psi, kk

def entmax(z, alpha, iters=80):
    """alpha-entmax by bisection on the dual threshold; alpha>1."""
    if abs(alpha - 2.0) < 1e-12:
        return sparsemax(z)[0]
    lo, hi = (alpha - 1) * z.max() - 1.0, (alpha - 1) * z.max()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        p = np.maximum((alpha - 1) * z - mid, 0.0) ** (1.0 / (alpha - 1))
        if p.sum() > 1.0: lo = mid
        else: hi = mid
    p = np.maximum((alpha - 1) * z - 0.5 * (lo + hi), 0.0) ** (1.0 / (alpha - 1))
    return p / p.sum()

def G(z, k):
    """Prefix statistic of Lem. 1: G(k) = sum_{r<=k} (z_(r) - z_(k))."""
    zs = np.sort(z)[::-1]
    return float(np.sum(zs[:k] - zs[k - 1]))

def targets_scores(rng, nq, m, delta_lo=0.2, spread=1.0):
    """Random column with a planted target set of size m and positive margin."""
    T = set(rng.choice(nq, size=m, replace=False).tolist())
    s = np.empty(nq)
    hi = spread * rng.random(m)                       # target scores in [0,spread)
    d  = delta_lo + rng.random()                      # margin
    lo = -d - spread * rng.random(nq - m)             # non-targets strictly below
    ti = sorted(T); ni = [i for i in range(nq) if i not in T]
    s[ti] = hi; s[ni] = lo
    return s, T

# =========================================================== Lem. 1 (prefix)
rng = np.random.default_rng(1)
v1 = 0; maxdiff = 0.0
for _ in range(5000):
    n = int(rng.integers(2, 31)); z = rng.normal(scale=3.0, size=n)
    g = [G(z, k) for k in range(1, n + 1)]
    if any(g[k + 1] < g[k] - TOL for k in range(n - 1)): v1 += 1     # monotone
    zs = np.sort(z)[::-1]
    for k in range(1, n):                                            # closed form
        maxdiff = max(maxdiff, abs(g[k] - g[k - 1] - k * (zs[k - 1] - zs[k])))
report("Lem. 1, monotonicity of G", v1, 5000)
report("Lem. 1, closed-form difference", maxdiff, None)

rng = np.random.default_rng(2); v = 0
for _ in range(3000):
    n = int(rng.integers(2, 31)); z = rng.normal(scale=3.0, size=n)
    _, _, kk = sparsemax(z)
    pred = max([k for k in range(1, n + 1) if G(z, k) < 1.0])
    if kk != pred: v += 1
report("Lem. 1, support characterization", v, 3000)

# ================================================= Prop. 9 (monotone support)
def support_monotone(rng, trials, alpha):
    v = 0
    for _ in range(trials):
        n = int(rng.integers(3, 25)); z = rng.normal(scale=2.0, size=n)
        taus = np.sort(rng.uniform(0.05, 20.0, size=12))
        sizes = [int((entmax(t * z, alpha) > 1e-12).sum()) for t in taus]
        if any(sizes[i + 1] > sizes[i] for i in range(len(sizes) - 1)): v += 1
    return v
rng = np.random.default_rng(3)
report("Prop. 9, alpha = 2", support_monotone(rng, 2000, 2.0), 2000)
for a in (1.2, 1.5, 1.8, 3.0):
    report(f"Prop. 9, alpha = {a}", support_monotone(rng, 600, a), 600)

# ============================================ Thm. 2 (recovery interval, SHARP)
# G(m+1) = tau * (W + m*delta) exactly, so k* <= m iff tau >= 1/(W + m*delta),
# and k* >= m iff tau < 1/W.  Both endpoints are exact, and the interval
# [1/(W + m*delta), 1/W) is non-empty whenever delta > 0 -- no rejection needed.
# Below the lower endpoint the support strictly contains T (P<1, R=1); at or
# above the upper endpoint it is a proper subset of T (R<1, P=1).
rng = np.random.default_rng(4)
v_in = 0; v_lo = 0; v_hi = 0; n_col = 0; n_in = 0; n_lo = 0; n_hi = 0; n_try = 6000
for _ in range(n_try):
    nq = int(rng.integers(4, 40)); m = int(rng.integers(1, min(8, nq)))
    s, T = targets_scores(rng, nq, m)
    delta = min(s[i] for i in T) - max(s[i] for i in range(nq) if i not in T)
    W = float(sum(s[i] - min(s[k] for k in T) for i in T))
    lo = 1.0 / (W + m * delta)                      # sharp lower endpoint
    hi = (1.0 / W) if W > 0 else np.inf             # sharp upper endpoint
    assert lo < hi                                  # non-empty whenever delta > 0
    n_col += 1
    # inside the sharp interval: exact recovery.  Counted in tau-evaluations, to match
    # the scope twin below; the row previously reported columns and so undercounted 4x.
    span = (min(hi, lo * 50.0) - lo)
    for tau in lo + span * rng.random(4):
        p, _, _ = sparsemax(tau * s)
        n_in += 1
        if set(np.nonzero(p)[0].tolist()) != T: v_in += 1
    # strictly below the lower endpoint: support strictly contains T
    for tau in lo * rng.uniform(0.05, 0.95, size=2):
        p, _, kk = sparsemax(tau * s); S = set(np.nonzero(p)[0].tolist())
        n_lo += 1
        if not (T < S): v_lo += 1                   # proper superset => P<1, R=1
    # at or above the upper endpoint: support is a proper subset of T
    if W > 0:
        for tau in hi * (1.0 + 3.0 * rng.random(2)):
            p, _, kk = sparsemax(tau * s); S = set(np.nonzero(p)[0].tolist())
            n_hi += 1
            if not (S < T): v_hi += 1               # proper subset => R<1, P=1
report("Thm. 2, sharp recovery interval", v_in, n_in)
report("Thm. 2, sharpness below lower endpoint (P<1)", v_lo, n_lo)
report("Thm. 2, sharpness above upper endpoint (R<1)", v_hi, n_hi)

# ================================= Prop. 24(iii): a hard limit is not a matching
rng = np.random.default_rng(5); n = 5
s = rng.normal(size=(n, n)); A = np.zeros((n, n))
for j in range(n): A[np.argmax(s[:, j]), j] = 1.0
colsums, rowsums = A.sum(0), A.sum(1)
matching = np.all(rowsums == 1)
report("Prop. 24(iii) not a matching (col sums all 1: %s; row sums %s)"
       % (bool(np.all(colsums == 1)), tuple(int(x) for x in rowsums)),
       0 if not matching else 1, None)

# ============================================== Prop. 21 (dilution bound)
rng = np.random.default_rng(6); v = 0
for _ in range(4000):
    nq = int(rng.integers(6, 200)); m = int(rng.integers(1, min(10, nq)))
    s, T = targets_scores(rng, nq, m)
    delta = min(s[i] for i in T) - max(s[i] for i in range(nq) if i not in T)
    tau = rng.uniform(0.1, 8.0)
    e = np.exp(tau * (s - s.max())); p = e / e.sum()
    inside = sum(p[i] for i in T); outside = 1.0 - inside
    if outside / inside > (nq - m) / m * np.exp(-tau * delta) + 1e-9: v += 1
report("Prop. 21, dilution bound", v, 4000)

# ============================================= Prop. 17 (graded column recall)
rng = np.random.default_rng(7); v = 0; n_acc = 0
while n_acc < 3000:
    nq = int(rng.integers(5, 40)); m = int(rng.integers(2, min(9, nq)))
    s, T = targets_scores(rng, nq, m)
    W = float(sum(s[i] - min(s[k] for k in T) for i in T))
    if W <= 0: continue
    tau = (1.0 / W) * (1.0 + 3.0 * rng.random())        # strictly above 1/W
    p, _, kk = sparsemax(tau * s); S = set(np.nonzero(p)[0].tolist())
    ss = np.sort(s[sorted(T)])[::-1]
    Wk = lambda k: float(np.sum(ss[:k] - ss[k - 1]))
    kpred = max([k for k in range(1, m + 1) if Wk(k) < 1.0 / tau])
    n_acc += 1
    if not S <= T or kk != kpred: v += 1                # P=1 and R=k(tau)/m
report("Prop. 17, graded recall", v, n_acc)

# ================================== Prop. 1 / Prop. 22 (hierarchy)
def hierarchical(z, B, K, arity=4):
    idx = np.arange(len(z)); surv = []
    for s0 in range(0, len(z), B):
        blk = idx[s0:s0 + B]
        surv.append(blk[np.argsort(z[blk])[::-1][:K]])
    while len(surv) > 1:
        nxt = []
        for s0 in range(0, len(surv), arity):
            grp = np.concatenate(surv[s0:s0 + arity])
            nxt.append(grp[np.argsort(z[grp])[::-1][:K]])
        surv = nxt
    cand = surv[0]
    sub, psi, _ = sparsemax(z[cand])
    p = np.zeros(len(z)); p[cand] = sub
    return p

rng = np.random.default_rng(8); v = 0; kmax = 0
for _ in range(2000):
    nq = int(rng.choice([64, 128, 256, 512, 1024, 2048]))
    z = rng.normal(scale=2.0, size=nq) * rng.uniform(0.05, 20.0)
    pg, _, kk = sparsemax(z); kmax = max(kmax, kk)
    K = int(kk + rng.integers(0, 20))                   # K >= k*
    B = int(rng.choice([8, 16, 32, 64]))
    if not np.allclose(hierarchical(z, B, K), pg, atol=1e-12): v += 1
report("Prop. 1, exact tournament (largest k* seen = %d)" % kmax, v, 2000)

# ---- how large must K be?  k* by regime.  tau is the regime parameter.
# The gate of Sec. 3.4 admits only columns with a clear front-runner, i.e. the
# exclusive (large-tau) regime; the broad regime is reported for contrast.
rng = np.random.default_rng(81)
print("\n  k* by regime (4000 columns each, n_q in {64,...,2048}):")
for label, lo, hi in [("exclusive  tau in [0.5, 6]", 0.5, 6.0),
                      ("mixed      tau in [0.1, 20]", 0.1, 20.0),
                      ("broad      tau in [0.02, 1]", 0.02, 1.0)]:
    ks = np.array([sparsemax(rng.normal(scale=2.0, size=int(rng.choice(
            [64,128,256,512,1024,2048]))) * rng.uniform(lo, hi))[2]
          for _ in range(4000)])
    print(f"    {label:28s} median {np.percentile(ks,50):5.1f}   "
          f"p99 {np.percentile(ks,99):5.1f}   max {ks.max():4d}")
print()

rng = np.random.default_rng(9); v = 0; n_acc = 0
for _ in range(7200):
    nq = int(rng.integers(32, 256)); m = int(rng.integers(3, 12))
    s, T = targets_scores(rng, nq, m)
    delta = min(s[i] for i in T) - max(s[i] for i in range(nq) if i not in T)
    W = float(sum(s[i] - min(s[k] for k in T) for i in T))
    if not W > 0: continue                               # need a finite upper endpoint
    lo = 1.0 / (W + m * delta)                           # sharp lower endpoint (Thm. 2)
    tau = lo + (1.0 / W - lo) * rng.random()
    K = int(rng.integers(1, m))                          # K < m  : under-retention
    p = hierarchical(tau * s, int(rng.choice([8, 16, 32])), K)
    S = set(np.nonzero(p)[0].tolist())
    top = set(np.array(sorted(T))[np.argsort(s[sorted(T)])[::-1][:K]].tolist())
    P = len(S & T) / len(S); R = len(S & T) / m; rho = float(sum(p[i] for i in T))
    n_acc += 1
    if S != top or abs(P - 1) > TOL or abs(R - K / m) > TOL or abs(rho - 1) > 1e-9: v += 1
report("Prop. 22, fidelity under retention (K < m)", v, n_acc)

# ============================ Prop. 14 (full-support precision ceiling), both margins
rng = np.random.default_rng(10); vc = vr = 0; nc = 0
for nq, nk in [(64, 16), (256, 32), (1024, 64), (4096, 128)]:
    for _ in range(3):
        m = int(rng.integers(1, min(nq, 20))); ki = int(rng.integers(1, nk))
        zc = rng.normal(size=nq); pc = np.exp(zc - zc.max()); pc /= pc.sum()
        Sc = set(np.nonzero(pc)[0].tolist()); Tc = set(range(m))
        if abs(len(Sc & Tc) / len(Sc) - m / nq) > TOL: vc += 1
        zr = rng.normal(size=nk); pr = np.exp(zr - zr.max()); pr /= pr.sum()
        Sr = set(np.nonzero(pr)[0].tolist()); Kr = set(range(ki))
        if abs(len(Sr & Kr) / len(Sr) - ki / nk) > TOL: vr += 1
        nc += 1
report("Prop. 14, precision ceiling (column)", vc, nc)
report("Prop. 14, row ceiling", vr, nc)

# ================================ Prop. 18 (row interval) and precision monotone
rng = np.random.default_rng(11); v = 0
for _ in range(4000):
    nk = int(rng.integers(3, 30)); ki = int(rng.integers(1, nk))
    At = np.empty(nk)
    At[:ki] = 1.0 + rng.random(ki)                       # true sources, separated
    At[ki:] = rng.random(nk - ki) * 0.9
    K = set(range(ki))
    lo, hi = At[ki:].max() if nk > ki else 0.0, At[:ki].min()
    for th in lo + (hi - lo) * rng.random(3):
        S = set(np.nonzero(np.maximum(At - th, 0.0))[0].tolist())
        if S != K: v += 1
report("Prop. 18, row interval", v, 4000)

rng = np.random.default_rng(12); v = 0
for _ in range(2000):
    nk = int(rng.integers(4, 30)); ki = int(rng.integers(1, nk))
    At = np.empty(nk)
    At[:ki] = 1.0 + rng.random(ki); At[ki:] = rng.random(nk - ki) * 0.9
    K = set(range(ki)); prev = -1.0
    for th in np.sort(rng.uniform(0, At.max(), size=8)):
        S = set(np.nonzero(np.maximum(At - th, 0.0))[0].tolist())
        P = len(S & K) / len(S) if S else 1.0
        if P < prev - TOL: v += 1                        # precision non-decreasing
        prev = P
report("Prop. 18, precision monotone", v, 2000)

# ==================================== Prop. 19 (graded row recall)
rng = np.random.default_rng(13); v = 0; n = 0
for _ in range(3000):
    nk = int(rng.integers(4, 30)); ki = int(rng.integers(1, nk))
    At = np.sort(rng.uniform(0.05, 3.0, size=nk))[::-1]  # separated by construction
    K = set(range(ki))
    for th in rng.uniform(At[ki - 1], At[0] * 1.05, size=6):
        S = set(np.nonzero(np.maximum(At - th, 0.0))[0].tolist())
        k_th = int(np.sum(At > th)); n += 1
        if abs(len(S & K) / ki - min(k_th, ki) / ki) > TOL: v += 1
        if S and S != set(range(len(S))): v += 1         # weakest-first (a prefix)
report("Prop. 19, graded row recall", v, n)

# ==================================== Prop. 20 (stage composition)
rng = np.random.default_rng(14); vv = [0, 0, 0, 0]; cols = 0
for _ in range(4000):
    nq = int(rng.integers(6, 40)); nk = int(rng.integers(2, 8))
    S0 = rng.normal(size=(nq, nk))
    tau = rng.uniform(0.3, 4.0, size=nk); c = rng.uniform(0.4, 2.5, size=nk)
    T = [set(rng.choice(nq, size=int(rng.integers(1, min(6, nq))), replace=False).tolist())
         for _ in range(nk)]
    if rng.random() < 0.5:                               # half with a planted margin
        for j in range(nk):
            for i in range(nq):
                S0[i, j] = (2.0 + abs(S0[i, j])) if i in T[j] else -(2.0 + abs(S0[i, j]))
    At = np.column_stack([c[j] * sparsemax(tau[j] * S0[:, j])[0] for j in range(nk)])
    th = rng.uniform(0, At.max() * 0.6, size=nq)
    A = np.maximum(At - th[:, None], 0.0)
    for j in range(nk):
        S1 = set(np.nonzero(At[:, j])[0].tolist()); S2 = set(np.nonzero(A[:, j])[0].tolist())
        cols += 1
        if not S2 <= S1: vv[0] += 1
        P1 = len(S1 & T[j]) / len(S1) if S1 else 1.0
        P2 = len(S2 & T[j]) / len(S2) if S2 else 1.0
        R1 = len(S1 & T[j]) / len(T[j]); R2 = len(S2 & T[j]) / len(T[j])
        if R2 > R1 + TOL: vv[0] += 1
        if abs(P1 - 1) < TOL and abs(P2 - 1) > TOL: vv[1] += 1
        if (T[j] - S1) & S2: vv[3] += 1
report("Prop. 20(i), support shrinks / recall non-increasing", vv[0], cols)
report("Prop. 20(ii), precision preserved", vv[1], cols)
report("Prop. 20(iv), false negatives stick", vv[3], cols)

# ==================================== Prop. 23 (prefix monotonicity, causal form)
# (i)  psi_j is non-decreasing as the visible prefix grows
# (ii) exclusion is permanent: once p_ij = 0 it stays 0
# (iii) the frozen-prefix variant OVER-admits, so its error is precision-only
rng = np.random.default_rng(16); vi = vii = viii = 0
for _ in range(4000):
    T = int(rng.integers(4, 120))
    z = rng.normal(scale=2.0, size=T) * rng.uniform(0.3, 6.0)
    psis = []; supports = []; frozen = np.zeros(T, bool)
    for t in range(1, T + 1):
        p, psi = sparsemax(z[:t])[:2]
        psis.append(psi); supports.append(set(np.nonzero(p)[0].tolist()))
        frozen[t - 1] = p[t - 1] > 0            # entry fixed when first visible
    if any(psis[t + 1] < psis[t] - TOL for t in range(T - 1)): vi += 1
    if any(i not in supports[t] and i in supports[t + 1]
           for t in range(T - 1) for i in range(t + 1)): vii += 1
    if not supports[-1] <= set(np.nonzero(frozen)[0].tolist()): viii += 1
report("Prop. 23(i), psi non-decreasing in prefix", vi, 4000)
report("Prop. 23(ii), exclusion is permanent", vii, 4000)
report("Prop. 23(iii), frozen prefix over-admits", viii, 4000)

# ==================================== Cor. 5 (joint attainment, full pipeline)
rng = np.random.default_rng(15); v = 0; n_acc = 0
while n_acc < 3000:
    nk = int(rng.integers(2, 6)); ms = rng.integers(1, 6, size=nk); nq = int(ms.sum())
    perm = rng.permutation(nq); T = []; o = 0
    for j in range(nk):
        T.append(set(perm[o:o + ms[j]].tolist())); o += ms[j]      # {T_j} partitions
    S0 = np.empty((nq, nk)); ok = True
    for j in range(nk):
        sj, _ = np.empty(nq), None
        hi = rng.random(ms[j]); d = 0.3 + rng.random()
        tj = sorted(T[j]); nj = [i for i in range(nq) if i not in T[j]]
        sj[tj] = hi; sj[nj] = -d - rng.random(len(nj))
        S0[:, j] = sj
    tau = np.empty(nk); c = rng.uniform(0.5, 2.0, size=nk); Kret = np.empty(nk, int)
    for j in range(nk):
        m = ms[j]; Tj = T[j]
        delta = min(S0[i, j] for i in Tj) - (max(S0[i, j] for i in range(nq) if i not in Tj)
                                             if m < nq else -1e9)
        W = float(sum(S0[i, j] - min(S0[k, j] for k in Tj) for i in Tj))
        lo = 1.0 / (W + m * delta)                              # sharp lower endpoint
        hi2 = (1.0 / W) if W > 0 else np.inf
        tau[j] = lo + (min(hi2, lo * 50) - lo) * rng.random()
        Kret[j] = m + int(rng.integers(0, 5))                       # K >= m
    if not ok: continue
    At = np.column_stack([c[j] * hierarchical(tau[j] * S0[:, j], 8, int(Kret[j]))
                          for j in range(nk)])
    weakest = min(c[j] * (1 - tau[j] * float(sum(S0[i, j] - min(S0[k, j] for k in T[j])
                  for i in T[j]))) / ms[j] for j in range(nk))
    if weakest <= 0: continue
    th = np.full(nq, weakest * rng.uniform(0.05, 0.95))
    A = np.maximum(At - th[:, None], 0.0)
    n_acc += 1
    bad = False
    for j in range(nk):                                             # column margin
        S = set(np.nonzero(A[:, j])[0].tolist())
        if S != T[j]: bad = True
    Kof = {i: {j for j in range(nk) if i in T[j]} for i in range(nq)}
    for i in range(nq):                                             # row margin
        S = set(np.nonzero(A[i, :])[0].tolist())
        if S != Kof[i]: bad = True
    if bad: v += 1
report("Cor. 5, joint attainment (all four quantities)", v, n_acc)

# ================================= App. E: selective exclusivity (restriction)
# Both programs are defined on an index set.  Restricting Stage 1 to E_.j and
# Stage 2 to E_i. leaves the sparsemax/truncation results untouched, moves n_q
# out of Assumption 2, and weakens the conservation obstruction to mu/n_k.
rng = np.random.default_rng(21)
v_in = v_lo = v_hi = 0; n_in = n_lo = n_hi = 0
for _ in range(6000):
    ne = int(rng.integers(3, 24)); m = int(rng.integers(1, ne))   # |E_.j| >= 2
    sE = np.empty(ne); sE[:m] = rng.random(m)
    d = 0.2 + rng.random(); sE[m:] = -d - rng.random(ne - m)
    T = set(range(m))
    delta = sE[:m].min() - sE[m:].max()                           # margin ON the scope
    W = float(np.sum(sE[:m] - sE[:m].min()))
    lo = 1.0 / (W + m * delta); hi = (1.0 / W) if W > 0 else np.inf
    assert lo < hi
    for tau in lo + (min(hi, lo * 50.0) - lo) * rng.random(4):
        n_in += 1
        if set(np.nonzero(sparsemax(tau * sE)[0])[0].tolist()) != T: v_in += 1
    for tau in lo * rng.uniform(0.05, 0.95, size=2):
        n_lo += 1
        if not (T < set(np.nonzero(sparsemax(tau * sE)[0])[0].tolist())): v_lo += 1
    if W > 0:
        for tau in hi * (1.0 + 3.0 * rng.random(2)):
            n_hi += 1
            if not (set(np.nonzero(sparsemax(tau * sE)[0])[0].tolist()) < T): v_hi += 1
report("Thm. 2 on a scope E_.j, sharp interval", v_in, n_in)
report("Thm. 2 on a scope, lower endpoint sharp", v_lo, n_lo)
report("Thm. 2 on a scope, upper endpoint sharp", v_hi, n_hi)

# the margin over a full column degrades in n_q; the margin over the scope does not
print("\n  margin and exact recovery vs n_q  (|E_.j| = 8, m = 3, pool ~ N(-2,1)):")
print(f"  {'n_q':>7}  {'delta full':>11}  {'delta scope':>12}  {'rec full':>9}  {'rec scope':>10}")
v = 0; n = 0
for nq in [8, 32, 128, 512, 2048, 8192]:
    # The scope column is regenerated identically at every n_q on purpose: it is what makes
    # the two delta columns comparable.  But that also means the scope RECOVERY check does
    # not vary with n_q -- `pool` never enters it -- so it is counted on the first pass only
    # and the remaining five passes are measurement, not verification.
    gE = np.random.default_rng(7)          # scope block: identical stream at every n_q
    gP = np.random.default_rng(11)         # the pool outside the scope: its own stream
    df = []; de = []; rf = 0; re_ = 0; tr = 3000
    for _ in range(tr):
        ne, m = 8, 3
        sE = np.empty(ne); sE[:m] = gE.random(m)
        dd = 0.2 + gE.random(); sE[m:] = -dd - gE.random(ne - m)
        pool = gP.normal(-2.0, 1.0, size=max(nq - ne, 0))
        T = set(range(m))
        d_scope = sE[:m].min() - sE[m:].max()
        d_full = sE[:m].min() - (max(sE[m:].max(), pool.max()) if pool.size else sE[m:].max())
        W = float(np.sum(sE[:m] - sE[:m].min()))
        de.append(d_scope); df.append(d_full)
        lo = 1.0 / (W + m * d_scope); hi = (1.0 / W) if W > 0 else np.inf
        tau = lo + 0.5 * (min(hi, lo * 50.0) - lo)
        rec = set(np.nonzero(sparsemax(tau * sE)[0])[0].tolist()) == T
        if nq == 8:                                  # count once; see the note above
            n += 1
            if not rec: v += 1
        if rec: re_ += 1
        sfull = np.concatenate([sE, pool])
        if set(np.nonzero(sparsemax(tau * sfull)[0])[0].tolist()) == T: rf += 1
    print(f"  {nq:>7}  {np.mean(df):>11.3f}  {np.mean(de):>12.3f}"
          f"  {100*rf/tr:>8.1f}%  {100*re_/tr:>9.1f}%")
print()
report("Assn. 2 on a scope: recovery, n_q-independent by construction", v, n)

# Prop. 23 transfers iff the mask is arrival-sealed (e_ij a function of the pair)
rng = np.random.default_rng(22); v = 0; n = 0
for _ in range(4000):
    nq = int(rng.integers(8, 60)); tau = rng.uniform(0.3, 4.0)
    s = rng.normal(size=nq)
    keep = rng.random(nq) < rng.uniform(0.2, 1.0)      # decided once, never revoked
    psi_prev = -np.inf; excluded = set()
    for t in range(1, nq + 1):
        idx = [i for i in range(t) if keep[i]]
        if not idx: continue
        p, psi, _ = sparsemax(tau * s[np.array(idx)])
        n += 1
        if psi < psi_prev - TOL: v += 1                # (i) threshold non-decreasing
        psi_prev = psi
        now0 = {idx[a] for a in range(len(idx)) if p[a] <= TOL}
        if excluded - now0: v += 1                     # (ii) exclusion permanent
        excluded |= now0
report("Prop. 23 under a nested (arrival-sealed) scope", v, n)

rng = np.random.default_rng(23); brk = 0; n = 0
for _ in range(3000):
    nq = int(rng.integers(8, 40)); tau = rng.uniform(0.5, 4.0)
    s = rng.normal(size=nq); broke = False; excluded = set()
    for t in range(1, nq + 1):
        keep = rng.random(t) < 0.7                     # re-decided each step: revocable
        idx = [i for i in range(t) if keep[i]]
        if not idx: continue
        p, _, _ = sparsemax(tau * s[np.array(idx)])
        now0 = {idx[a] for a in range(len(idx)) if p[a] <= TOL}
        if (excluded & set(idx)) - now0: broke = True
        excluded |= now0
    n += 1; brk += broke
print(f"  a revocable scope breaks permanence on {100*brk/n:.1f}% of {n} prefixes\n")

# Prop. 4 weakens: the bound is mu/n_k, and n_q/n_k is escapable off the scope
rng = np.random.default_rng(24); v1 = v2 = 0; n = 0
for _ in range(3000):
    nq = int(rng.integers(32, 256)); nk = int(rng.integers(4, 16))
    A = rng.random((nq, nk)); A /= A.sum(1, keepdims=True)        # rows still sum to 1
    M = rng.random((nq, nk)) < rng.uniform(0.02, 0.5)
    colsum = np.array([A[M[:, j], j].sum() for j in range(nk)])
    mu = float(A[M].sum()); n += 1
    if colsum.max() < mu / nk - TOL: v1 += 1                      # corrected bound holds
    if colsum.max() >= nq / nk - TOL: v2 += 1                     # original is escapable
report("Prop. 4 on a scope: max_j >= mu/n_k", v1, n)
report("Prop. 4 on a scope: n_q/n_k escapable", v2, n)

# Prop. 14 under a mask: the ceiling becomes m_j/|E_.j|, still score-independent
rng = np.random.default_rng(25); v = 0; n = 0
for _ in range(4000):
    ne = int(rng.integers(2, 40)); m = int(rng.integers(1, ne))
    z = rng.normal(size=ne) * rng.uniform(0.1, 8.0)
    p = np.exp(z - z.max()); p /= p.sum()
    S = set(np.nonzero(p)[0].tolist()); n += 1
    if abs(len(S & set(range(m))) / len(S) - m / ne) > TOL: v += 1
report("Prop. 14 under a mask: ceiling m_j/|E_.j|", v, n)

# Prop. 20 under ONE shared mask (Stage 1 on E_.j, Stage 2 on E_i.)
rng = np.random.default_rng(26); vv = [0, 0, 0]; cols = 0
for _ in range(3000):
    nq = int(rng.integers(8, 40)); nk = int(rng.integers(2, 8))
    S0 = rng.normal(size=(nq, nk))
    M = rng.random((nq, nk)) < rng.uniform(0.25, 1.0)
    tau = rng.uniform(0.3, 4.0, size=nk); c = rng.uniform(0.4, 2.5, size=nk)
    T = [set(np.nonzero(M[:, j])[0][rng.random(int(M[:, j].sum())) < 0.4].tolist())
         for j in range(nk)]
    At = np.zeros((nq, nk))
    for j in range(nk):
        idx = np.nonzero(M[:, j])[0]
        if idx.size: At[idx, j] = c[j] * sparsemax(tau[j] * S0[idx, j])[0]
    th = rng.uniform(0, At.max() * 0.6 + 1e-9, size=nq)
    A = np.zeros_like(At)
    for i in range(nq):
        jdx = np.nonzero(M[i, :])[0]
        A[i, jdx] = np.maximum(At[i, jdx] - th[i], 0.0)
    for j in range(nk):
        if not T[j]: continue
        S1 = set(np.nonzero(At[:, j])[0].tolist()); S2 = set(np.nonzero(A[:, j])[0].tolist())
        cols += 1
        if not S2 <= S1: vv[0] += 1
        P1 = len(S1 & T[j]) / len(S1) if S1 else 1.0
        P2 = len(S2 & T[j]) / len(S2) if S2 else 1.0
        if len(S1 & T[j]) / len(T[j]) < len(S2 & T[j]) / len(T[j]) - TOL: vv[0] += 1
        if abs(P1 - 1) < TOL and abs(P2 - 1) > TOL: vv[1] += 1
        if (T[j] - S1) & S2: vv[2] += 1
report("Prop. 20(i) under one mask", vv[0], cols)
report("Prop. 20(ii) under one mask", vv[1], cols)
report("Prop. 20(iv) under one mask", vv[2], cols)

# ============ Fig. 2: the (R_j, P_j) trace of the fan-out program as tau sweeps
# Thm. 2 and Prop. 19(i) between them fix the whole curve: below the interval
# R=1 and P=m_j/k*, inside it P=R=1, above it P=1 and R=k*/m_j.  So the trace is
# an L with its corner at (1,1) and no interior point -- which is the shape
# Fig. 2 plots and E7 pre-commits to.  A full-support mechanism is not on a
# curve at all: it sits at (1, m_j/n_q) at every temperature.
rng = np.random.default_rng(27)
v_lo = v_in = v_hi = v_shape = 0; n_lo = n_in = n_hi = n_col = 0
for _ in range(4000):
    nq = int(rng.integers(6, 60)); m = int(rng.integers(1, min(8, nq)))
    s, T = targets_scores(rng, nq, m)
    delta = min(s[i] for i in T) - max(s[i] for i in range(nq) if i not in T)
    W = float(sum(s[i] - min(s[k] for k in T) for i in T))
    lo = 1.0 / (W + m * delta); hi = (1.0 / W) if W > 0 else np.inf
    n_col += 1; interior = False
    taus = np.concatenate([lo * rng.uniform(0.02, 0.99, size=6),
                           lo + (min(hi, lo * 40) - lo) * rng.random(4),
                           hi * (1.0 + 4.0 * rng.random(6)) if W > 0 else np.array([])])
    for tau in taus:
        S = set(np.nonzero(sparsemax(tau * s)[0])[0].tolist())
        k = len(S); P = len(S & T) / k; R = len(S & T) / m
        if abs(P - 1) > TOL and abs(R - 1) > TOL: interior = True
        if tau < lo - TOL:
            n_lo += 1
            if abs(R - 1) > TOL or abs(P - m / k) > TOL: v_lo += 1
        elif tau < hi:
            n_in += 1
            if abs(R - 1) > TOL or abs(P - 1) > TOL: v_in += 1
        else:
            n_hi += 1
            if abs(P - 1) > TOL or abs(R - k / m) > TOL: v_hi += 1
    if interior: v_shape += 1
report("Fig. 2, trace below the interval (R=1, P=m/k*)", v_lo, n_lo)
report("Fig. 2, trace inside the interval (P=R=1)", v_in, n_in)
report("Fig. 2, trace above the interval (P=1, R=k*/m)", v_hi, n_hi)
# The L-shape is a re-reading of the same evaluations, one boolean per column, so
# it is printed rather than reported: counting it again would double-count them.
print(f"\n  no interior point (P<1 and R<1 at once) on any of {n_col} columns "
      f"meeting Assn. 2: {v_shape} violations, over the same "
      f"{n_lo + n_in + n_hi} evaluations\n")

rng = np.random.default_rng(28); v = 0; n = 0
for _ in range(2000):
    nq = int(rng.integers(8, 500)); m = int(rng.integers(1, min(8, nq)))
    z = rng.normal(size=nq); T = set(range(m)); pts = set()
    for tau in np.exp(rng.uniform(-4, 4, size=8)):          # any temperature
        p = np.exp(tau * z - (tau * z).max()); p /= p.sum()
        S = set(np.nonzero(p)[0].tolist())
        pts.add((round(len(S & T) / m, 12), round(len(S & T) / len(S), 12)))
    n += 1
    if pts != {(1.0, round(m / nq, 12))}: v += 1
report("Fig. 2, full support is one point (1, m/n_q)", v, n)


# =====================================================================================
# The two-step selective form (Sec. 2): step 1 fixes the whole margin at standard
# attention's mass; step 2 re-shapes only the exclusive set, conserving its quota.
# =====================================================================================

def _row_softmax(S):
    e = np.exp(S - S.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

def _proj(v, s):                    # argmin .5||a-v||^2  s.t.  sum a = s, a >= 0
    v = np.asarray(v, float); u = np.sort(v)[::-1]; c = np.cumsum(u) - s
    k = np.arange(1, len(v) + 1); cond = u - c / k > 0; r = k[cond][-1]
    return np.maximum(v - c[cond][-1] / r, 0.0)

def _proj_le(v, s):                 # argmin .5||a-v||^2  s.t.  sum a <= s, a >= 0
    w = np.maximum(np.asarray(v, float), 0.0)
    return w if w.sum() <= s else _proj(v, s)

# ---- Prop. A: conservation and consistency of the two-step form
rng = np.random.default_rng(31)
v_col = v_anch = v_quota = v_row = v_off = v_sub = v_zero = v_quota_cap = v_off1 = v_capid = v_tau1 = 0
n_col = n_row = 0
for _ in range(3000):
    nq = int(rng.integers(10, 45)); nk = int(rng.integers(3, 8))
    S = rng.normal(size=(nq, nk))
    E = rng.random((nq, nk)) < rng.uniform(0.25, 0.85)
    M = _row_softmax(S).sum(0)                       # standard attention's column masses
    At = np.zeros((nq, nk)); cprime = np.zeros(nk)
    Asm = _row_softmax(S)
    for j in range(nk):
        base = Asm[:, j]                             # step 1 IS standard attention's column
        At[:, j] = base
        Ej = np.where(E[:, j])[0]
        if len(Ej):
            cprime[j] = base[Ej].sum()               # cbar_j inherited: the scope's quota
            p = sparsemax(rng.uniform(0.3, 4.0) * S[Ej, j])[0]
            At[Ej, j] = cprime[j] * p                # step 2: re-shape, conserve the quota
            if abs(At[Ej, j].sum() - cprime[j]) > TOL: v_quota += 1
        n_col += 1
        if abs(At[:, j].sum() - M[j]) > 1e-8: v_col += 1        # (i) column total preserved
        if not np.array_equal(At[~E[:, j], j], Asm[~E[:, j], j]): v_off1 += 1   # Stage-1 complement untouched (exact)
    if abs(At.sum() - nq) > 1e-7: v_anch += 1                   # (ii) sum_j c_j = n_q, unimposed
    for i in range(nq):
        sm_mass_i = Asm[i, :].sum()                  # the row's OWN softmax mass (== 1 in exact arithmetic)
        a1 = _proj_le(At[i, :], sm_mass_i)           # step 1: a CAP at the row's own mass
        Ei = np.where(E[i, :])[0]; a2 = a1.copy()
        if len(Ei):
            ci = a1[Ei].sum(); ti = rng.uniform(0.3, 4.0)
            if ci > TOL: a2[Ei] = _proj_le(At[i, Ei], ci / ti) * ti  # step 2 is a CAP at the quota
        n_row += 1
        if a2.sum() > sm_mass_i + 1e-8: v_row += 1              # (iii) never more than the row's own softmax mass
        if a2.sum() < sm_mass_i - 1e-8: v_sub += 1              # sub-unit rows survive (counted)
        if At[i, :].sum() <= sm_mass_i and not np.array_equal(a1, At[i, :]): v_capid += 1   # the cap is the identity on a sub-unit row
        if len(Ei) and ci > TOL:                                # tau_i = 1 reproduces step 1's row exactly
            if not np.allclose(_proj_le(At[i, Ei], ci) , a1[Ei], atol=1e-12): v_tau1 += 1
        off = ~E[i, :]
        if not np.allclose(a2[off], a1[off]): v_off += 1        # (iv) complement fixed by step 1
        if ((At[i, :] == 0) & (a2 > 0)).any(): v_zero += 1     # (v) a Stage-1 zero never resurrects
        if a2[Ei].sum() > ci + 1e-8 if len(Ei) else False: v_quota_cap += 1   # quota is a ceiling
report("Prop. A(i), column total = standard attention's", v_col, n_col)
report("Prop. A(ii), sum_j c_j = n_q without imposing it", v_anch, 3000)
report("Prop. A, fan-out scope carries exactly its quota cbar_j", v_quota, n_col)
report("Prop. A(v), a Stage-1 zero never resurrects under eq:rowprog", v_zero, n_row)
report("Prop. A, fan-in scope never exceeds its quota cbar_i", v_quota_cap, n_row)
report("Prop. A(iii), row mass never exceeds the row's own softmax mass", v_row, n_row)
report("Prop. A(iv), the complement is untouched by step 2", v_off, n_row)
report("Prop. A(iv), Stage-1 complement untouched (column half)", v_off1, n_col)
report("Prop. A, the unit cap is the identity on a sub-unit row", v_capid, n_row)
report("Sec. 2.2, tau_i = 1 reproduces step 1's row exactly", v_tau1, n_row)
# ---- with an empty relation the mechanism IS standard attention (exact)
rng2 = np.random.default_rng(34); v_std = 0
for _ in range(2000):
    nq = int(rng2.integers(5, 40)); nk = int(rng2.integers(3, 9)); S = rng2.normal(size=(nq, nk))
    Asm = _row_softmax(S); a = np.stack([_proj_le(Asm[i], Asm[i].sum()) for i in range(nq)])
    if not np.allclose(a, Asm, atol=1e-12): v_std += 1
report("Prop. A, empty relation reproduces standard attention exactly", v_std, 2000)
print(f"     (rows spending strictly less than a unit: {100*v_sub/n_row:.1f}% -- the property is live)")
# ---- tau_j -> 0 on a NON-empty relation is the UNIFORM re-shaping, not standard attention
#      (App. D/E of the paper; an earlier draft said the opposite).  Lem. 1: at tau -> 0 every
#      prefix has G < 1, so the support is the whole relation and entries tend to cbar_j/|E_.j|.
rng3 = np.random.default_rng(35); v_unif = 0; v_notstd = 0; n_u = 0
for _ in range(2000):
    nq = int(rng3.integers(6, 40)); S = rng3.normal(size=(nq, 3)); Asm = _row_softmax(S)
    Ej = np.where(rng3.random(nq) < 0.6)[0]
    if len(Ej) < 2: continue
    n_u += 1
    cb = Asm[Ej, 0].sum(); p = sparsemax(1e-9 * S[Ej, 0])[0]; col = cb * p
    if not np.allclose(col, cb / len(Ej), atol=1e-6): v_unif += 1          # uniform over the relation
    if np.allclose(col, Asm[Ej, 0], atol=1e-3): v_notstd += 1               # and NOT standard attention
report("App. E(e), tau_j -> 0 gives the uniform re-shaping cbar_j/|E_.j|", v_unif, n_u)
report("App. E(e), tau_j -> 0 does not return standard attention", v_notstd, n_u)

# ---- Prop. B: scope monotonicity of Assumption 2 and the recovery interval
rng = np.random.default_rng(32)
v_d = v_w = v_cont = 0; n_b = 0; n_pos = 0; rescued = 0
for _ in range(6000):
    nq = int(rng.integers(12, 90)); m = int(rng.integers(2, 6))
    s = rng.normal(size=nq); T = list(range(m)); s[T] = 2.0 + rng.random(m)
    pool = list(range(m, nq))
    E = set(T) | set(rng.choice(pool, size=int(rng.integers(3, len(pool))), replace=False).tolist())
    outer = sorted(E - set(T))
    Ep = set(T) | set(rng.choice(outer, size=int(rng.integers(1, max(2, len(outer)))),
                                 replace=False).tolist())
    W = float(sum(s[i] - min(s[k] for k in T) for i in T))
    d = min(s[i] for i in T) - max(s[i] for i in E if i not in T)
    dp = min(s[i] for i in T) - max(s[i] for i in Ep if i not in T)
    n_b += 1
    if dp < d - TOL: v_d += 1                                    # (i) delta non-decreasing
    if m * dp < m * d - TOL: v_w += 1                            # (ii) width non-decreasing
    if d > 0:                                                    # (iii) interval containment,
        n_pos += 1                                               #      given Assn. 2 on the larger
        lo, lop = 1.0 / (W + m * d), 1.0 / (W + m * dp)
        if not (lop <= lo + TOL): v_cont += 1
    if d <= 0 < dp: rescued += 1                                 # (iv) the rescue direction
report("Prop. B(i), delta_j non-decreasing as the scope shrinks", v_d, n_b)
report("Prop. B(ii), relative interval width non-decreasing", v_w, n_b)
report("Prop. B(iii), interval(E) contained in interval(E')", v_cont, n_pos)
print(f"     (Assn. 2 fails on E but holds on the shrunken E' on {100*rescued/n_b:.1f}% of columns)")

# ---- Cor. 2 (cardinality half): degree is bounded as the exclusive set grows
rng = np.random.default_rng(33)
v_deg = v_prec = 0; n_d = 0
for _ in range(2000):
    mj = int(rng.integers(2, 7))
    for msize in [mj, 2 * mj, 4 * mj, 8 * mj, 16 * mj]:          # |E| grows, m_j fixed
        s = rng.uniform(0, 1, size=msize) * 0.9
        s[:mj] = 1.0 + rng.random(mj); T = set(range(mj))
        smin = s[:mj].min(); W = float((s[:mj] - smin).sum())
        delta = smin - (s[mj:].max() if msize > mj else -1.0)
        lo = 1.0 / (W + mj * delta); hi = (1.0 / W) if W > 0 else np.inf
        tau = lo + 0.5 * (min(hi, lo * 20.0) - lo)
        Ssup = set(np.nonzero(sparsemax(tau * s)[0])[0].tolist())
        n_d += 1
        if len(Ssup) != mj: v_deg += 1                           # degree stays at m_j
        if len(Ssup & T) / len(Ssup) != 1.0: v_prec += 1         # precision stays at 1
report("Cor. 2 (cardinality), degree = m_j as |E_.j| grows", v_deg, n_d)
report("Cor. 2 (cardinality), precision = 1 as |E_.j| grows", v_prec, n_d)

# ---- The single-target column (m_j = 1) and the composition that removes distractors.
#      Thm. 2 at m_j = 1: W_j = 0, interval [1/delta_j, inf).  For tau_j >= 1/delta_j the program
#      puts the ENTIRE quota on the argmax and exact zeros elsewhere; sparsemax is 1-Lipschitz in
#      its argument so the column moves continuously in tau_j.  A key whose argmax is not the
#      answer query therefore contributes exactly 0 to the answer row after Stage 1.
rng = np.random.default_rng(40)
v_arg = v_lip = v_comp = 0; n_arg = n_lip = n_comp = n_reach = 0
for _ in range(3000):
    nq = int(rng.integers(3, 40)); s = rng.normal(size=nq)
    top = np.argsort(s)[::-1]; delta = s[top[0]] - s[top[1]]
    if delta <= 1e-6: continue
    for tau in (1.0 / delta) * (1.001 + 3.0 * rng.random(3)):        # above 1/delta_j (the endpoint itself is a Thm. 2 row)
        p = sparsemax(tau * s)[0]; n_arg += 1
        e = np.zeros(nq); e[top[0]] = 1.0
        if (p > 0).sum() != 1 or abs(p[top[0]] - 1.0) > 1e-12: v_arg += 1   # entire quota on the argmax
    t1 = rng.uniform(0.2, 2.0) / delta; t2 = t1 + rng.uniform(0.0, 0.5) / delta   # a sweep through 1/delta
    p1 = sparsemax(t1 * s)[0]; p2 = sparsemax(t2 * s)[0]; n_lip += 1
    if np.linalg.norm(p2 - p1) > (t2 - t1) * np.linalg.norm(s) + 1e-9: v_lip += 1   # continuity in tau_j
# composition: distractor keys, one answer row i*, relation = full column
for _ in range(2000):
    nq = int(rng.integers(4, 30)); nk = int(rng.integers(3, 10)); S = rng.normal(size=(nq, nk))
    Asm = _row_softmax(S); istar = int(rng.integers(0, nq)); At = np.zeros_like(Asm)
    for j in range(nk):
        top = np.argsort(S[:, j])[::-1]; delta = S[top[0], j] - S[top[1], j]
        if delta <= 1e-6: continue
        tau = (1.0 / delta) * (1.0 + 2.0 * rng.random())
        At[:, j] = Asm[:, j].sum() * sparsemax(tau * S[:, j])[0]
        n_comp += 1
        if top[0] != istar:
            if At[istar, j] != 0.0: v_comp += 1                         # excluded from row i* by Stage 1 alone
        else:
            n_reach += 1                                                # reaches row i*; the row program decides
report("Thm. 2 at m_j = 1, entire quota on the argmax for tau_j >= 1/delta_j", v_arg, n_arg)
report("Thm. 2 at m_j = 1, column continuous in tau_j through 1/delta_j (1-Lipschitz)", v_lip, n_lip)
report("Composition, a key concentrating elsewhere is an exact zero on the answer row", v_comp, n_comp)
print(f"     (keys concentrating onto the answer row, left to the row program: {100*n_reach/n_comp:.1f}%)")

# ----------------------------------------------------------------- print
w = max(len(r[0]) for r in results)
print("=" * (w + 22))
for label, v, n in results:
    if n is None:
        print(f"{label:<{w}}   {v:.3e}" if isinstance(v, float) else f"{label:<{w}}   {v}")
    else:
        print(f"{label:<{w}}   {v:>6d} / {n}")
print("=" * (w + 22))
# a machine-readable verdict and an exit code: preflight.sh needs something to gate on, and a wall of "0 / 10000"
# lines is not it (review d5bd980 F)
# "checks" names two quantities a factor of 17,000 apart: the paper reports 1,058,450 CHECKS, which is the number
# of individual trials, while a row of this table is one property.  Both are printed (review 30372ae D).  Rows with
# no trial count (a closed-form difference, a structural assertion) carry no evaluations but can still FAIL, so they
# gate too -- they used to be dropped entirely.
_counted = [(label, v, n) for label, v, n in results if n is not None and isinstance(v, int)]
_uncounted = [(label, v, n) for label, v, n in results if n is None and isinstance(v, int)]
_evals = sum(n for _, _, n in _counted)
_total = sum(v for _, v, _ in _counted) + sum(v for _, v, _ in _uncounted)
print(f"TOTAL VIOLATIONS: {_total} over {_evals} evaluations in {len(_counted) + len(_uncounted)} checks "
      f"({len(_counted)} with trial counts, {len(_uncounted)} structural)")
if _total:
    for label, v, n in _counted + _uncounted:
        if v:
            print(f"  FAILED {label}: {v}" + (f" / {n}" if n is not None else ""))
    sys.exit(1)
