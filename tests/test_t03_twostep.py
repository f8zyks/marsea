"""T3 (prop:twostep: INV-1..5, 7; SAN-1), T6 (prop:stagecomp on the REAL eq:rowprog path), T9
(fullsupportceiling), T10 (fig:prcurve), T12 (tau -> 0 limit), T14 (target detector), T16
(tau_i = 1 identity) -- Sec. 11.  Harness reference: verify_all.py L585-668 (Prop. A block)."""
import numpy as np
import torch
import pytest
from conftest import rand_case, planted_column, interval
from marsea.normalizer import MarSeaNormalizer, State, row_softmax
from marsea.invariants import invariant_report
from marsea.primitives import proj_le_masked, sparsemax_masked
from marsea.baselines import SoftmaxNorm, SoftmaxOneNorm, MESHNorm

D, R = 16, 8


@pytest.fixture(scope="module")
def norm():
    torch.manual_seed(0)
    return MarSeaNormalizer(D, R)


# ----------------------------------------------------------------------------- T3
def test_t3_invariants_random(rng, norm):
    fails = {}
    for t in range(400):
        cs = rand_case(rng, use_finfo_min=(t % 4 == 0))
        use_head = (t % 3 == 0)
        with torch.no_grad():
            A, d = norm.normalize(cs["S"], cs["vis"], cs["K_kv"], cs["Q"], State(), logits_override=cs["logits"],
                                  tau_j_override=None if use_head else cs["tau_j"],
                                  tau_i_override=None if use_head else cs["tau_i"])
        rep = invariant_report(A, d, cs["vis"])
        for k, (ok, worst) in rep.items():
            if not ok:
                fails.setdefault(k, []).append(worst)
        # the relation E is exactly the sign of the logits on vis (one object)
        assert (d.E == ((cs["logits"] > 0) & cs["vis"].expand_as(cs["logits"]))).all()
        # p is a distribution on every non-empty E_.j and 0 elsewhere
        ne = d.E.sum(-2) > 0
        assert torch.allclose(d.p.sum(-2)[ne], torch.ones_like(d.p.sum(-2)[ne]), atol=1e-5)
        assert (d.p.sum(-2)[~ne] == 0).all()
        # cbar_j reads A_sm
        assert torch.allclose(d.cbar_j, (d.A_sm * d.E).sum(-2), atol=1e-6)
        # nu in [1, |E_.j|], 1 on empty columns
        assert (d.nu >= 1 - 1e-5).all() and (d.nu <= d.E.sum(-2).clamp_min(1) + 1e-4).all()
        assert (d.nu[~ne] == 1).all()
    assert not fails, {k: (len(v), max(v)) for k, v in fails.items()}


def test_t3_harness_prop_a_reference(rng):
    """compare_harness.py: the numpy Prop. A loop of verify_all.py L597-639 vs the tensor path (fp64 to 1e-12)."""
    def np_sparsemax(z):
        n = len(z); zs = np.sort(z)[::-1]; cs = np.cumsum(zs); k = np.arange(1, n + 1)
        kk = k[(1 + k * zs) > cs][-1]; psi = (cs[kk - 1] - 1.0) / kk
        return np.maximum(z - psi, 0.0)
    def _row_softmax(S):
        e = np.exp(S - S.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
    def _proj(v, s):
        v = np.asarray(v, float); u = np.sort(v)[::-1]; c = np.cumsum(u) - s
        k = np.arange(1, len(v) + 1); cond = u - c / k > 0; r = k[cond][-1]
        return np.maximum(v - c[cond][-1] / r, 0.0)
    def _proj_le(v, s, bind_tol=0.0):
        w = np.maximum(np.asarray(v, float), 0.0); return w if w.sum() <= s + bind_tol else _proj(v, s)
    TOL = 1e-9
    def harness(S, E, tau, ti):
        nq, nk = S.shape; Asm = _row_softmax(S); At = np.zeros((nq, nk))
        for j in range(nk):
            At[:, j] = Asm[:, j]; Ej = np.where(E[:, j])[0]
            if len(Ej):
                cp = Asm[Ej, j].sum(); At[Ej, j] = cp * np_sparsemax(tau[j] * S[Ej, j])
        A = np.zeros((nq, nk)); a1s = np.zeros((nq, nk))
        for i in range(nq):
            a1 = _proj_le(At[i, :], 1.0, 1e-6); a1s[i] = a1; Ei = np.where(E[i, :])[0]; a2 = a1.copy()
            if len(Ei):
                ci = a1[Ei].sum()
                if ci > TOL: a2[Ei] = _proj_le(At[i, Ei], ci / ti[i]) * ti[i]
                else: a2[Ei] = 0.0                     # spec: u = 0 at a zero quota (C-25; differs by <= 1e-9)
            A[i] = a2
        return Asm, At, a1s, A
    norm = MarSeaNormalizer(D, R)
    worst = 0.0
    for t in range(150):
        nq = int(rng.integers(10, 45)); nk = int(rng.integers(3, 8))
        S = rng.normal(size=(nq, nk)); E = rng.random((nq, nk)) < rng.uniform(0.25, 0.85)
        tau = rng.uniform(0.3, 4.0, size=nk); ti = rng.uniform(0.3, 4.0, size=nq)
        Asm, At, a1, A = harness(S, E, tau, ti)
        with torch.no_grad():
            Am, d = norm.normalize(torch.as_tensor(S, dtype=torch.float64)[None, None], torch.ones(nq, nk, dtype=torch.bool),
                                   torch.randn(1, 1, nk, D, dtype=torch.float64), torch.randn(1, 1, nq, D, dtype=torch.float64), State(),
                                   logits_override=torch.as_tensor(np.where(E, 1.0, -1.0), dtype=torch.float64)[None, None],
                                   tau_j_override=torch.as_tensor(tau, dtype=torch.float64)[None, None],
                                   tau_i_override=torch.as_tensor(ti, dtype=torch.float64)[None, None])
        for ours, ref in ((d.A_sm, Asm), (d.Atil, At), (d.a1, a1), (Am, A)):
            worst = max(worst, float(np.abs(ours[0, 0].numpy() - ref).max()))
    assert worst < 1e-12, worst


# ----------------------------------------------------------------------------- T6
def test_t6_stage1_zero_never_resurrects_on_real_path(rng, norm):
    """Prop. A(v) on the REAL eq:rowprog path (never a hand-drawn floor): Atil == 0 => A == 0; supp(A_.j) subset
    supp(Atil_.j); theta_i >= 0 wherever step 2 runs; and the same under the extreme tau_i in both directions."""
    bad = 0; n_rows = 0; n_theta = 0
    for t in range(300):
        cs = rand_case(rng)
        for ti in (cs["tau_i"], torch.full_like(cs["tau_i"], 0.05), torch.full_like(cs["tau_i"], 50.0)):
            with torch.no_grad():
                A, d = norm.normalize(cs["S"], cs["vis"], cs["K_kv"], cs["Q"], State(), logits_override=cs["logits"],
                                      tau_j_override=cs["tau_j"], tau_i_override=ti)
            n_rows += A.shape[-2]
            if (A[d.Atil == 0] != 0).any(): bad += 1
            if (((A > 0) & ~(d.Atil > 0)).sum(-2) != 0).any(): bad += 1
            if (d.theta < 0).any(): bad += 1
            n_theta += int((d.theta > 0).sum())
    assert bad == 0
    assert n_theta > 0, "theta never positive: the binding regime was not exercised"


# ----------------------------------------------------------------------------- T9
def test_t9_full_support_ceiling_dense_arms(rng):
    """every dense arm sits at exactly m_j/|domain| at EVERY tau / scoring function, both domains; no variance."""
    b0, b1, b2 = SoftmaxNorm(), SoftmaxOneNorm(), MESHNorm(D)
    viol = 0; n = 0
    for _ in range(60):
        nq = int(rng.integers(6, 40)); nk = int(rng.integers(3, 8))
        S = torch.randn(1, 1, nq, nk) * float(rng.uniform(0.1, 8.0))
        vis = torch.ones(1, 1, nq, nk, dtype=torch.bool)
        Erel = torch.rand(1, 1, nq, nk) < 0.6                              # MarSea's paired relation (the matched domain)
        m = int(rng.integers(1, nq)); T = torch.zeros(nq, dtype=torch.bool); T[:m] = True
        K_kv = torch.randn(1, 1, nk, D); Q = torch.randn(1, 1, nq, D)
        for arm in (b0, b1, b2):
            with torch.no_grad():
                A, _ = arm.normalize(S, vis, K_kv, Q)
            for j in range(nk):
                col = A[0, 0, :, j]
                supp = col > 0
                n += 1
                if abs(supp.float().mean().item() * 0 + (supp & T).sum().item() / supp.sum().item() - m / nq) > 1e-9: viol += 1
                sr = supp & Erel[0, 0, :, j]
                if sr.sum() > 0:
                    mrel = int((T & Erel[0, 0, :, j]).sum()); nrel = int(Erel[0, 0, :, j].sum())
                    if abs((sr & T).sum().item() / sr.sum().item() - mrel / nrel) > 1e-9: viol += 1
    assert viol == 0, f"{viol}/{n}"


# ----------------------------------------------------------------------------- T10
def test_t10_pr_trace_is_an_L(rng, norm):
    """sweeping tau_j traces the L: (R=1, P=m/k*) below, (1,1) inside, (P=1, R=k*/m) above; no interior point.
    Run THROUGH the normaliser (E = the column, tau_j_override) so the Stage-1 support is the real one."""
    v_shape = 0; v_reg = 0; n_col = 0
    for _ in range(300):
        nq = int(rng.integers(6, 60)); m = int(rng.integers(1, min(8, nq)))
        s, T = planted_column(rng, nq, m)
        lo, hi, delta, W = interval(s, T)
        S = torch.as_tensor(s, dtype=torch.float32)[None, None, :, None]           # one key column
        vis = torch.ones(1, 1, nq, 1, dtype=torch.bool)
        logits = torch.ones(1, 1, nq, 1)
        taus = np.concatenate([lo * rng.uniform(0.02, 0.99, size=4), lo + (min(hi, lo * 40) - lo) * rng.random(3),
                               hi * (1.0 + 4.0 * rng.random(4)) if W > 0 else np.array([])])
        n_col += 1; interior = False
        for tau in taus:
            with torch.no_grad():
                A, d = norm.normalize(S, vis, torch.randn(1, 1, 1, D), torch.randn(1, 1, nq, D), State(),
                                      logits_override=logits, tau_j_override=torch.tensor([[[float(tau)]]]),
                                      tau_i_override=torch.ones(1, 1, nq))
            Ssup = set(torch.nonzero(d.Atil[0, 0, :, 0] > 0).flatten().tolist())
            k = len(Ssup); P = len(Ssup & T) / k; Rr = len(Ssup & T) / m
            if abs(P - 1) > 1e-9 and abs(Rr - 1) > 1e-9: interior = True
            if tau < lo - 1e-9:
                if abs(Rr - 1) > 1e-9 or abs(P - m / k) > 1e-9: v_reg += 1
            elif tau < hi:
                if abs(Rr - 1) > 1e-9 or abs(P - 1) > 1e-9: v_reg += 1
            else:
                if abs(P - 1) > 1e-9 or abs(Rr - k / m) > 1e-9: v_reg += 1
        if interior: v_shape += 1
    assert v_reg == 0 and v_shape == 0, (v_reg, v_shape, n_col)


# ----------------------------------------------------------------------------- T12
def test_t12_tau_to_zero_is_uniform_reshaping(rng, norm):
    """tau_j = 1e-9 on random non-empty relations: entries == cbar_j/|E_.j| and != A_sm (H9)."""
    n_diff = 0
    for t in range(200):
        cs = rand_case(rng)
        with torch.no_grad():
            A, d = norm.normalize(cs["S"], cs["vis"], cs["K_kv"], cs["Q"], State(), logits_override=cs["logits"],
                                  tau_j_override=torch.full_like(cs["tau_j"], 1e-9), tau_i_override=cs["tau_i"])
        E = d.E; cnt = E.sum(-2)
        target = (d.cbar_j / cnt.clamp_min(1)).unsqueeze(-2).expand_as(A)
        assert torch.allclose(d.Atil[E], target[E], atol=1e-6)
        big = (cnt >= 2).unsqueeze(-2).expand_as(A) & E & (cs["vis"].expand_as(A).sum(-1, keepdim=True) >= 2)
        if big.any() and (d.Atil[big] - d.A_sm[big]).abs().max() > 1e-3:
            n_diff += 1
    assert n_diff > 100, "tau -> 0 should NOT return standard attention on most cases"


# ----------------------------------------------------------------------------- T14
def test_t14_target_is_atil_not_a1(rng, norm):
    """on cap-binding rows A[i,E] == tau_i * proj_le(Atil[i,E], cbar_i/tau_i) and != the a1-target version somewhere."""
    n_bind = 0; n_differ = 0
    for t in range(300):
        cs = rand_case(rng, rel_density=0.9)
        tj = torch.full_like(cs["tau_j"], 8.0)                       # sharp columns push rows over one unit
        with torch.no_grad():
            A, d = norm.normalize(cs["S"], cs["vis"], cs["K_kv"], cs["Q"], State(), logits_override=cs["logits"],
                                  tau_j_override=tj, tau_i_override=cs["tau_i"])
        E = d.E
        rows = torch.nonzero(d.cap_binds)
        for (b, h, i) in rows.tolist():
            if E[b, h, i].sum() == 0: continue
            n_bind += 1
            Ei = E[b, h, i]
            q = (d.cbar_i[b, h, i] / d.tau_i[b, h, i])[None]
            u_ref, _ = proj_le_masked(d.Atil[b, h, i][None].masked_fill(~Ei[None], 0), q, Ei[None])
            assert torch.allclose(A[b, h, i][Ei], (d.tau_i[b, h, i] * u_ref[0])[Ei], atol=1e-6)
            u_alt, _ = proj_le_masked(d.a1[b, h, i][None].masked_fill(~Ei[None], 0), q, Ei[None])
            if (u_alt[0][Ei] - u_ref[0][Ei]).abs().max() > 1e-6: n_differ += 1
    assert n_bind > 20, "the unit cap never bound; the detector was not exercised"
    assert n_differ > 0, "target = Atil and target = a1 never differed: T14 has no bite"


# ----------------------------------------------------------------------------- T16
def test_t16_tau_i_one_is_identity_on_step1_row(rng, norm):
    """tau_i = 1 gives A[i,E] == a1[i,E] exactly (to fp) on every row incl. cap-binding ones (record Sec. 43)."""
    n_bind = 0; worst = 0.0
    for t in range(300):
        cs = rand_case(rng, rel_density=float(rng.uniform(0.3, 1.0)))
        tj = cs["tau_j"] if t % 2 else torch.full_like(cs["tau_j"], 8.0)
        with torch.no_grad():
            A, d = norm.normalize(cs["S"].double(), cs["vis"], cs["K_kv"].double(), cs["Q"].double(), State(),
                                  logits_override=cs["logits"].double(), tau_j_override=tj.double(),
                                  tau_i_override=torch.ones_like(cs["tau_i"]).double())
        n_bind += int(d.cap_binds.sum())
        worst = max(worst, float((A - d.a1).abs().max()))
    assert n_bind > 50
    assert worst < 1e-9, worst


def test_empty_relation_is_standard_attention(rng, norm):
    """INV-9 via the head path too: logits forced very negative => E empty => A == A_sm (1e-6)."""
    for t in range(100):
        cs = rand_case(rng, use_finfo_min=(t % 2 == 0))
        with torch.no_grad():
            A, d = norm.normalize(cs["S"], cs["vis"], cs["K_kv"], cs["Q"], State(),
                                  logits_override=torch.full_like(cs["logits"], -1e4))
        assert (~d.E).all()
        assert torch.allclose(A, row_softmax(cs["S"], cs["vis"]), atol=1e-6)


def test_distractor_split_and_m1_column(rng, norm):
    """Sec. 12.3a / harness L716-751: at tau_j >= 1/delta_j on a full-column relation the whole quota goes to the
    argmax query and a key concentrating elsewhere is an EXACT zero on the answer row after Stage 1."""
    v = 0; n = 0; n_reach = 0
    for _ in range(300):
        nq = int(rng.integers(4, 30)); nk = int(rng.integers(3, 10))
        S = torch.randn(1, 1, nq, nk); vis = torch.ones(1, 1, nq, nk, dtype=torch.bool)
        istar = int(rng.integers(0, nq))
        top2 = S[0, 0].topk(2, dim=0).values; delta = (top2[0] - top2[1])
        if (delta <= 1e-4).any(): continue
        tau = (1.0 / delta) * (1.0 + 2.0 * torch.rand(nk))
        with torch.no_grad():
            A, d = norm.normalize(S, vis, torch.randn(1, 1, nk, D), torch.randn(1, 1, nq, D), State(),
                                  logits_override=torch.ones(1, 1, nq, nk), tau_j_override=tau[None, None],
                                  tau_i_override=torch.ones(1, 1, nq))
        amax = S[0, 0].argmax(0)
        for j in range(nk):
            n += 1
            if int((d.p[0, 0, :, j] > 0).sum()) != 1 or abs(float(d.p[0, 0, amax[j], j]) - 1) > 1e-6: v += 1
            if amax[j] != istar:
                if d.Atil[0, 0, istar, j] != 0: v += 1
            else:
                n_reach += 1
    assert v == 0 and n > 0
