"""
compare_harness.py -- step 4: port verify_all.py's Prop. A per-column/per-row loop (lines ~597-632)
to a numpy reference and compare with marsea_cold.MarSeaNormalizer on random cases.
"""
import sys, numpy as np, torch
sys.path.insert(0, "/home/claude/cold")
from marsea_cold import MarSeaNormalizer, State

# ---- verbatim from verify_all.py -----------------------------------------------------------
def sparsemax(z):
    n = len(z); zs = np.sort(z)[::-1]; cs = np.cumsum(zs)
    k = np.arange(1, n + 1)
    kk = k[(1 + k * zs) > cs][-1]
    psi = (cs[kk - 1] - 1.0) / kk
    return np.maximum(z - psi, 0.0), psi, kk

def _row_softmax(S):
    e = np.exp(S - S.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

def _proj(v, s):
    v = np.asarray(v, float); u = np.sort(v)[::-1]; c = np.cumsum(u) - s
    k = np.arange(1, len(v) + 1); cond = u - c / k > 0; r = k[cond][-1]
    return np.maximum(v - c[cond][-1] / r, 0.0)

def _proj_le(v, s):
    w = np.maximum(np.asarray(v, float), 0.0)
    return w if w.sum() <= s else _proj(v, s)

TOL = 1e-9
def harness_two_step(S, E, tau, ti):
    """Prop. A block, with the per-column tau and per-row tau_i passed in instead of sampled."""
    nq, nk = S.shape
    Asm = _row_softmax(S); At = np.zeros((nq, nk)); cprime = np.zeros(nk)
    for j in range(nk):
        base = Asm[:, j]; At[:, j] = base
        Ej = np.where(E[:, j])[0]
        if len(Ej):
            cprime[j] = base[Ej].sum()
            p = sparsemax(tau[j] * S[Ej, j])[0]
            At[Ej, j] = cprime[j] * p
    A = np.zeros((nq, nk)); a1s = np.zeros((nq, nk))
    for i in range(nq):
        a1 = _proj_le(At[i, :], 1.0); a1s[i] = a1
        Ei = np.where(E[i, :])[0]; a2 = a1.copy()
        if len(Ei):
            ci = a1[Ei].sum()
            if ci > TOL: a2[Ei] = _proj_le(At[i, Ei], ci / ti[i]) * ti[i]
        A[i] = a2
    return Asm, At, a1s, A
# ---------------------------------------------------------------------------------------------

rng = np.random.default_rng(31)
norm = MarSeaNormalizer(16, 8)
maxd = {"A_sm": 0.0, "Atil": 0.0, "a1": 0.0, "A": 0.0}
maxd64 = {"A": 0.0}
n_cap = 0; n_zero_quota = 0
for t in range(200):
    nq = int(rng.integers(10, 45)); nk = int(rng.integers(3, 8))
    S = rng.normal(size=(nq, nk))
    E = rng.random((nq, nk)) < rng.uniform(0.25, 0.85)
    tau = rng.uniform(0.3, 4.0, size=nk); ti = rng.uniform(0.3, 4.0, size=nq)
    Asm, At, a1, A = harness_two_step(S, E, tau, ti)
    logits = np.where(E, 1.0, -1.0)
    vis = torch.ones(nq, nk, dtype=torch.bool)
    for dt, store in ((torch.float32, maxd), (torch.float64, maxd64)):
        with torch.no_grad():
            Am, d = norm.normalize(torch.as_tensor(S, dtype=dt)[None, None], vis,
                                   torch.randn(1, 1, nk, 16, dtype=dt), torch.randn(1, 1, nq, 16, dtype=dt), State(),
                                   logits_override=torch.as_tensor(logits, dtype=dt)[None, None],
                                   tau_j_override=torch.as_tensor(tau, dtype=dt)[None, None],
                                   tau_i_override=torch.as_tensor(ti, dtype=dt)[None, None])
        if dt == torch.float32:
            store["A_sm"] = max(store["A_sm"], np.abs(d.A_sm[0, 0].numpy() - Asm).max())
            store["Atil"] = max(store["Atil"], np.abs(d.Atil[0, 0].numpy() - At).max())
            store["a1"] = max(store["a1"], np.abs(d.a1[0, 0].numpy() - a1).max())
        store["A"] = max(store["A"], np.abs(Am[0, 0].numpy() - A).max())
    n_cap += int(d.cap_binds.sum()); n_zero_quota += int(((d.cbar_i == 0) & (d.E.sum(-1) > 0)).sum())
print("200 cases, unmasked (as in the harness).  max |diff| vs harness reference:")
print("  fp32 :", {k: f"{v:.2e}" for k, v in maxd.items()})
print("  fp64 :", {k: f"{v:.2e}" for k, v in maxd64.items()})
print(f"  rows where the unit cap bound: {n_cap};  rows with a non-empty relation and cbar_i == 0: {n_zero_quota}")
