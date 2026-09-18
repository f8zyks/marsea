"""
test_cold.py -- INV-1..INV-13 of spec Sec. 2 plus the extra checks requested, on random inputs.
Run: python3 test_cold.py
"""
import math, sys
import torch
import numpy as np

sys.path.insert(0, "/home/claude/cold")
from marsea_cold import (MarSeaNormalizer, SoftmaxNorm, SoftmaxOneNorm, MESHNorm, KeyOnlyTauNorm,
                         RowEntmaxNorm, MatchedSparsityNorm, State, row_softmax, sorted_prefix_stat,
                         sparsemax_masked, _SparsemaxMasked, proj_le_masked, repeat_kv)

torch.manual_seed(0)
rng = np.random.default_rng(0)
D = 16      # head dim (small for speed)
R = 8
TOL = 1e-6

counts = {}
def check(name, ok_bool_tensor_or_bool):
    ok = bool(ok_bool_tensor_or_bool)
    c = counts.setdefault(name, [0, 0])
    c[1] += 1
    if not ok:
        c[0] += 1

def rand_case(use_finfo_min=False, force_E=None, n_q=None, n_k=None):
    B = int(rng.integers(1, 3)); Hkv = int(rng.integers(1, 3)); g = int(rng.integers(1, 3)); H = Hkv * g
    n_q = n_q or int(rng.integers(1, 41)); n_k = n_k or int(rng.integers(1, 11))
    # offset-causal vis (ambiguity #6) plus random extra masking, every row keeps >= 1 visible key
    i = torch.arange(n_q)[:, None]; j = torch.arange(n_k)[None, :]
    off = n_k - n_q
    vis = (j <= i + off)
    vis = vis & (torch.rand(n_q, n_k) > 0.15)
    for ii in range(n_q):
        if not vis[ii].any():
            vis[ii, max(0, min(n_k - 1, ii + off))] = True
    vis = vis[None, None].expand(B, H, n_q, n_k).clone()
    if rng.random() < 0.3:                          # per-batch padding: drop some keys entirely in one batch elt
        b = int(rng.integers(0, B)); jj = int(rng.integers(0, n_k))
        vis[b, :, :, jj] = False
        for ii in range(n_q):
            if not vis[b, 0, ii].any():
                vis[b, :, ii, jj] = True
    scale = float(rng.uniform(0.3, 4.0))
    S = torch.randn(B, H, n_q, n_k) * scale
    neg = torch.finfo(torch.float32).min if use_finfo_min else float("-inf")
    S = S.masked_fill(~vis, neg)
    K_kv = torch.randn(B, Hkv, n_k, D)
    Q = torch.randn(B, H, n_q, D)
    # random relation with empty rows/cols: logits sign decides E
    if force_E is None:
        dens = float(rng.uniform(0.0, 1.0))
        E = (torch.rand(B, H, n_q, n_k) < dens)
        if rng.random() < 0.3:                      # empty column(s)
            E[..., int(rng.integers(0, n_k))] = False
        if rng.random() < 0.3:                      # empty row(s)
            E[..., int(rng.integers(0, n_q)), :] = False
        E = E & vis
    else:
        E = force_E.expand(B, H, n_q, n_k) & vis
    logits = torch.where(E, torch.rand_like(S) * 2 + 0.01, -(torch.rand_like(S) * 2 + 0.01))
    tau_j = torch.as_tensor(rng.uniform(0.05, 6.0, size=(B, H, n_k)), dtype=torch.float32)
    tau_i = torch.as_tensor(rng.uniform(0.05, 6.0, size=(B, H, n_q)), dtype=torch.float32)
    return dict(S=S, vis=vis, K_kv=K_kv, Q=Q, logits=logits, tau_j=tau_j, tau_i=tau_i, E=E)

norm = MarSeaNormalizer(D, R)

# ---------------------------------------------------------------- INV-1..8, 10..12 on random cases
N_CASES = 400
for t in range(N_CASES):
    cs = rand_case(use_finfo_min=(t % 4 == 0))
    use_head_tau = (t % 3 == 0)                     # a third of cases use the real TauK/TauQ heads
    with torch.no_grad():
        A, d = norm.normalize(cs["S"], cs["vis"], cs["K_kv"], cs["Q"], State(),
                              logits_override=cs["logits"],
                              tau_j_override=None if use_head_tau else cs["tau_j"],
                              tau_i_override=None if use_head_tau else cs["tau_i"])
    vis, E = cs["vis"], d.E
    n_q = A.shape[-2]
    check("finite", torch.isfinite(A).all() and torch.isfinite(d.tau_j).all() and torch.isfinite(d.tau_i).all())
    check("INV-1 col sums of Atil == c", torch.allclose(d.Atil.sum(-2), d.c, atol=1e-5))
    check("INV-2 sum_j c_j == n_q (derived)", torch.allclose(d.c.sum(-1), torch.full_like(d.c.sum(-1), n_q), atol=1e-4))
    check("INV-3 row sums of A <= 1", (A.sum(-1) <= 1 + TOL).all())
    check("INV-4 A == a1 off the relation", torch.equal(A[~E], d.a1[~E]))
    check("INV-5 one E: kstar<=|E_.j|, supp_rel<=|E_i.|", (d.kstar <= E.sum(-2)).all() and (d.supp_rel <= E.sum(-1)).all()
          and (E == (cs["logits"] > 0) & vis).all())
    check("INV-6 Atil==0 => A==0", (A[d.Atil == 0] == 0).all())
    check("INV-7 supp(A[:,j]) subset supp(Atil[:,j])", (((A > 0) & ~(d.Atil > 0)).sum(-2) == 0).all())
    check("INV-8 ~vis => A,Atil,a1,p == 0 and E False", (A[~vis] == 0).all() and (d.Atil[~vis] == 0).all()
          and (d.a1[~vis] == 0).all() and (d.p[~vis] == 0).all() and (~E[~vis]).all())
    check("INV-10 theta >= 0", (d.theta >= 0).all())
    check("INV-12 relation row mass <= cbar_i", ((A * E).sum(-1) <= d.cbar_i + TOL).all())
    # quota exactly carried on non-empty columns (Prop. A)
    ne = E.sum(-2) > 0
    check("fan-out relation carries exactly cbar_j", torch.allclose((d.Atil * E).sum(-2)[ne], d.cbar_j[ne], atol=1e-5))
    check("p is a distribution on each non-empty E_.j", torch.allclose(d.p.sum(-2)[ne], torch.ones_like(d.p.sum(-2)[ne]), atol=1e-5)
          and (d.p.sum(-2)[~ne] == 0).all())
    check("cbar_j = sum_E A_sm", torch.allclose(d.cbar_j, (d.A_sm * E).sum(-2), atol=1e-6))

# ---------------------------------------------------------------- INV-11: E all-False => A == A_sm
for t in range(200):
    cs = rand_case(force_E=torch.zeros(1, 1, 1, 1, dtype=torch.bool), use_finfo_min=(t % 2 == 0))
    with torch.no_grad():
        A, d = norm.normalize(cs["S"], cs["vis"], cs["K_kv"], cs["Q"], State(), logits_override=cs["logits"])
    Asm = row_softmax(cs["S"], cs["vis"])
    check("INV-11 E=all-False => A == row_softmax(S) (1e-6)", torch.allclose(A, Asm, atol=1e-6))
    check("INV-11 E=all-False => E is empty", (~d.E).all())

# ---------------------------------------------------------------- INV-13: tau_j = 1e-9 => uniform on the relation
for t in range(200):
    cs = rand_case()
    with torch.no_grad():
        A, d = norm.normalize(cs["S"], cs["vis"], cs["K_kv"], cs["Q"], State(), logits_override=cs["logits"],
                              tau_j_override=torch.full_like(cs["tau_j"], 1e-9), tau_i_override=cs["tau_i"])
    E = d.E; cnt = E.sum(-2)
    target = (d.cbar_j / cnt.clamp_min(1)).unsqueeze(-2).expand_as(A)
    ok = torch.allclose(d.Atil[E], target[E], atol=1e-6)
    check("INV-13 tau_j=1e-9 => Atil[E_.j,j] == cbar_j/|E_.j|", ok)
    big = (cnt >= 2).unsqueeze(-2).expand_as(A) & E & (cs["vis"].sum(-1, keepdim=True) >= 2)   # rows with 1 visible key have A_sm = 1 = uniform (test artifact)
    if big.any():
        check("INV-13 ... and != A_sm somewhere on |E_.j|>=2 columns", (d.Atil[big] - d.A_sm[big]).abs().max() > 1e-3)

# ---------------------------------------------------------------- INV-9: growing visible prefix, tau_j frozen
for t in range(150):
    n_q = int(rng.integers(2, 30)); n_k = int(rng.integers(1, 8))
    S = torch.randn(1, 1, n_q, n_k) * float(rng.uniform(0.3, 4.0))
    keep = torch.rand(1, 1, n_q, n_k) < float(rng.uniform(0.3, 1.0))      # arrival-sealed relation
    tau_j = torch.as_tensor(rng.uniform(0.1, 6.0, size=(1, 1, n_k)), dtype=torch.float32)
    K_kv = torch.randn(1, 1, n_k, D); Q = torch.randn(1, 1, n_q, D)
    psi_prev = torch.full((n_k,), float("-inf")); excluded = torch.zeros(n_q, n_k, dtype=torch.bool)
    ok_psi = True; ok_perm = True
    for tt in range(n_q):
        vis = torch.zeros(n_q, n_k, dtype=torch.bool); vis[: tt + 1] = True      # keys all visible, queries arrive
        logits = torch.where(keep, torch.ones_like(S), -torch.ones_like(S))
        with torch.no_grad():
            A, d = norm.normalize(S, vis, K_kv, Q, State(), logits_override=logits, tau_j_override=tau_j, tau_i_override=torch.ones(1, 1, n_q))
        nonempty = d.E[0, 0].sum(0) > 0
        psi = d.psi_j[0, 0]
        if ((psi < psi_prev - 1e-6) & nonempty).any(): ok_psi = False
        psi_prev = torch.where(nonempty, psi, psi_prev)
        now0 = (d.p[0, 0] == 0) & d.E[0, 0]
        if (excluded & d.E[0, 0] & ~now0).any(): ok_perm = False
        excluded |= now0
    check("INV-9 psi_j non-decreasing in the prefix (tau_j frozen)", ok_psi)
    check("INV-9 exclusion permanent (tau_j frozen)", ok_perm)

# ---------------------------------------------------------------- B0 == row softmax
b0 = SoftmaxNorm()
for t in range(100):
    cs = rand_case(use_finfo_min=(t % 2 == 0))
    A, _ = b0.normalize(cs["S"], cs["vis"])
    ref = torch.softmax(cs["S"].masked_fill(~cs["vis"], float("-inf")), -1)
    check("B0 == row_softmax(S)", torch.allclose(A, ref, atol=1e-7))

# ---------------------------------------------------------------- primitives: T1 (support == top-k*, G monotone)
for t in range(300):
    L = int(rng.integers(1, 30)); z = torch.randn(4, L) * 3
    mask = torch.rand(4, L) < 0.7; mask[:, 0] = True
    kstar, psi, G, u, mv = sorted_prefix_stat(z, mask)
    p = sparsemax_masked(z, mask)
    Gv = G.clone(); Gv[~mv] = float("inf")
    ok = True
    for b in range(4):
        gg = Gv[b][mv[b]]
        if (gg[1:] < gg[:-1] - 1e-6).any(): ok = False
        supp = int((p[b] > 0).sum())
        if supp != int(kstar[b]): ok = False
        if abs(float(p[b].sum()) - 1) > 1e-5: ok = False
        # support is the top-k* of the sorted order
        zs = z[b].masked_fill(~mask[b], -1e30); order = torch.argsort(zs, descending=True)[: supp]
        if not (p[b][order] > 0).all(): ok = False
    check("T1 support == top-k*, G non-decreasing, sum p == 1", ok)

# ---------------------------------------------------------------- gradcheck of the custom sparsemax Function (fp64)
gc_ok = True
for t in range(20):
    L = int(rng.integers(2, 9)); z = (torch.randn(3, L, dtype=torch.float64) * 2).requires_grad_(True)
    mask = torch.rand(3, L) < 0.8; mask[:, 0] = True
    try:
        gc_ok &= torch.autograd.gradcheck(lambda zz: _SparsemaxMasked.apply(zz, mask), (z,), eps=1e-6, atol=1e-5)
    except Exception as e:
        gc_ok = False
check("T13 gradcheck custom sparsemax", gc_ok)

# ---------------------------------------------------------------- gradients flow into every parameter
torch.manual_seed(1)
cs = rand_case(n_q=24, n_k=8)
norm2 = MarSeaNormalizer(D, R)
with torch.no_grad():
    norm2.relation.calibrate_b0(cs["K_kv"], cs["Q"], cs["vis"], 0.3)
S = cs["S"].clone().requires_grad_(True)
A, d = norm2.normalize(S, cs["vis"], cs["K_kv"], cs["Q"], State())
w = torch.randn_like(A)
loss = (A * w).sum()
loss.backward()
names = {"U_phi": norm2.relation.U.weight, "V_phi": norm2.relation.V.weight, "b0": norm2.relation.b0,
         "TauK.fc1": norm2.tauK.fc1.weight, "TauK.fc2": norm2.tauK.fc2.weight, "TauK.fc2.bias": norm2.tauK.fc2.bias,
         "TauQ.fc1": norm2.tauQ.fc1.weight, "TauQ.fc2": norm2.tauQ.fc2.weight, "TauQ.fc2.bias": norm2.tauQ.fc2.bias,
         "S": S}
grad_report = {}
for k, prm in names.items():
    g = prm.grad
    ok = g is not None and torch.isfinite(g).all().item() and (g.abs().sum().item() > 0)
    grad_report[k] = (None if g is None else float(g.abs().sum()))
    check(f"grad finite & nonzero: {k}", ok)
print("coverage at grad test:", float((d.E & cs['vis']).sum() / cs['vis'].sum()), "|E_.j| hist:", torch.bincount(d.E.sum(-2).flatten()).tolist())

# ---------------------------------------------------------------- finite-difference check of d loss/d tau_j, d tau_i (T13)
fd_ok = True
for t in range(10):
    cs = rand_case(n_q=6, n_k=4)
    tj = cs["tau_j"].double().requires_grad_(True); ti = cs["tau_i"].double().requires_grad_(True)
    def f(tj_, ti_):
        A, _ = norm.normalize(cs["S"].double(), cs["vis"], cs["K_kv"].double(), cs["Q"].double(), State(),
                              logits_override=cs["logits"].double(), tau_j_override=tj_, tau_i_override=ti_)
        return (A * torch.arange(A.numel(), dtype=torch.float64).view_as(A)).sum()
    try:
        fd_ok &= torch.autograd.gradcheck(f, (tj, ti), eps=1e-6, atol=1e-4, rtol=1e-3, nondet_tol=1e-6)
    except Exception as e:
        fd_ok = False
check("T13 finite-difference d loss/d tau_j, d tau_i", fd_ok)

# ---------------------------------------------------------------- baselines run and basic properties
b1 = SoftmaxOneNorm(); b2 = MESHNorm(D); b3 = KeyOnlyTauNorm(D, R); b4a = RowEntmaxNorm(2.0); b4b = RowEntmaxNorm(1.5); b5 = MatchedSparsityNorm()
for t in range(30):
    cs = rand_case()
    with torch.no_grad():
        A1, d1 = b1.normalize(cs["S"], cs["vis"])
        check("B1 rows sum < 1, no zeros on vis", (A1.sum(-1) < 1).all() and (A1[cs["vis"]] > 0).all())
        A4, _ = b4a.normalize(cs["S"], cs["vis"])
        check("B4 alpha=2 rows sum to 1", torch.allclose(A4.sum(-1), torch.ones_like(A4.sum(-1)), atol=1e-5))
        A4b, _ = b4b.normalize(cs["S"], cs["vis"])
        check("B4 alpha=1.5 rows sum to 1 (1e-4)", torch.allclose(A4b.sum(-1), torch.ones_like(A4b.sum(-1)), atol=1e-4))
        A3, d3 = b3.normalize(cs["S"], cs["vis"], cs["K_kv"], cs["Q"], State())
        check("B3 runs, rows <= 1, finite", torch.isfinite(A3).all() and (A3.sum(-1) <= 1 + TOL).all())
        Am, dm = norm.normalize(cs["S"], cs["vis"], cs["K_kv"], cs["Q"], State(), logits_override=cs["logits"],
                                tau_j_override=cs["tau_j"], tau_i_override=cs["tau_i"])
        A5, d5 = b5.normalize(cs["S"], cs["vis"], trace=(dm.E, dm.supp_rel))
        check("B5 matches |supp| on the relation and softmax off it",
              torch.equal(((A5 > 0) & dm.E).sum(-1), dm.supp_rel) and torch.allclose(A5[~dm.E], dm.A_sm[~dm.E]))
        check("B5 rows sum to 1 where the count > 0", torch.allclose(A5.sum(-1)[dm.supp_rel > 0], torch.ones_like(A5.sum(-1)[dm.supp_rel > 0]), atol=1e-5))
    if t < 6:
        with torch.no_grad():
            A2, d2 = b2.normalize(cs["S"], cs["vis"], cs["K_kv"], cs["Q"])
        check("B2 finite, zero off vis", torch.isfinite(A2).all() and (A2[~cs["vis"]] == 0).all())
        rs = d2.extra["row_sum"]; b = d2.extra["b"]
        check("B2 row sums == b_i (1e-3)  [expected to FAIL under causal masks]", torch.allclose(rs, b, atol=1e-3))
        check("B2 column sums == a_j (1e-3)", torch.allclose(d2.extra["col_sum"], d2.extra["a"], atol=1e-3))

# B2 gradient flows into h_a, h_b
cs = rand_case(n_q=8, n_k=5)
A2, _ = b2.normalize(cs["S"], cs["vis"], cs["K_kv"], cs["Q"])
(A2 * torch.randn_like(A2)).sum().backward()
check("B2 grad into h_a, h_b", all(p.grad is not None and torch.isfinite(p.grad).all() for p in b2.parameters()))

# ---------------------------------------------------------------- report
print("\n%-70s %8s %8s" % ("check", "fails", "trials"))
print("-" * 90)
allok = True
for k, (f, n) in counts.items():
    print("%-70s %8d %8d   %s" % (k, f, n, "PASS" if f == 0 else "FAIL"))
    allok &= (f == 0)
print("-" * 90)
print("grad |sum|:", {k: (None if v is None else round(v, 4)) for k, v in grad_report.items()})
print("ALL PASS" if allok else "SOME FAILURES")
