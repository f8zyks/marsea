"""T1 (lem:prefix), T2 (thm:recovery, both endpoints, on a scope too), T13 (gradients) -- Sec. 11.
Ports of verify_all.py blocks L52-70, L92-123, L376-399 to the tensor code path."""
import numpy as np
import torch
import pytest
from conftest import planted_column, interval, rand_case
from marsea.primitives import (sorted_prefix_stat, sparsemax_masked, SparsemaxMasked, entmax_masked,
                               proj_le_masked, proj_eq_masked)


def np_sparsemax(z):
    n = len(z); zs = np.sort(z)[::-1]; cs = np.cumsum(zs)
    k = np.arange(1, n + 1)
    kk = k[(1 + k * zs) > cs][-1]
    psi = (cs[kk - 1] - 1.0) / kk
    return np.maximum(z - psi, 0.0), psi, kk


def G_np(z, k):
    zs = np.sort(z)[::-1]
    return float(np.sum(zs[:k] - zs[k - 1]))


# ----------------------------------------------------------------------------- T1
def test_t1_prefix_statistic_matches_harness(rng):
    """support == top-k* of the sorted order; G non-decreasing; k* = max{k : G(k) < 1}; psi as the harness."""
    viol = 0; n = 0
    for _ in range(3000):
        L = int(rng.integers(2, 31)); z = rng.normal(scale=3.0, size=L)
        zt = torch.as_tensor(z, dtype=torch.float64)[None]
        mask = torch.ones(1, L, dtype=torch.bool)
        kstar, psi, G, u, mv = sorted_prefix_stat(zt, mask)
        p_np, psi_np, kk = np_sparsemax(z)
        p = sparsemax_masked(zt, mask)[0].numpy()
        n += 1
        if int(kstar) != kk or abs(float(psi) - psi_np) > 1e-9: viol += 1
        if np.abs(p - p_np).max() > 1e-9: viol += 1
        g = G[0].numpy()
        if np.any(g[1:] < g[:-1] - 1e-9): viol += 1                 # monotone
        pred = max([k for k in range(1, L + 1) if G_np(z, k) < 1.0])
        if int(kstar) != pred: viol += 1
        if abs(p.sum() - 1) > 1e-9: viol += 1
        order = np.argsort(-z, kind="stable")[:kk]
        if not (p[order] > 0).all(): viol += 1                       # support is the top-k* prefix
    assert viol == 0, f"T1: {viol} violations in {n}"


def test_t1_padded_ragged_batches(rng):
    """the batched form on a padded [n_sets, L_max] tensor with -1e30 pads EXCLUDED from the prefix count."""
    viol = 0
    for _ in range(500):
        n_sets = int(rng.integers(1, 6)); L = int(rng.integers(1, 20))
        z = torch.randn(n_sets, L, dtype=torch.float64) * 3
        mask = torch.rand(n_sets, L) < 0.6
        mask[:, 0] = True
        kstar, psi, G, u, mv = sorted_prefix_stat(z, mask)
        p = sparsemax_masked(z, mask)
        for b in range(n_sets):
            zz = z[b][mask[b]].numpy()
            p_ref, psi_ref, kk = np_sparsemax(zz)
            if int(kstar[b]) != kk or abs(float(psi[b]) - psi_ref) > 1e-9: viol += 1
            if np.abs(p[b][mask[b]].numpy() - p_ref).max() > 1e-9: viol += 1
            if (p[b][~mask[b]] != 0).any(): viol += 1
            gg = G[b][mv[b]].numpy()
            if np.any(gg[1:] < gg[:-1] - 1e-9): viol += 1
    # an entirely masked set gives p = 0, kstar = 0, finite psi
    z = torch.randn(2, 5, dtype=torch.float64); mask = torch.zeros(2, 5, dtype=torch.bool)
    kstar, psi, G, u, mv = sorted_prefix_stat(z, mask)
    p = sparsemax_masked(z, mask)
    assert (p == 0).all() and (kstar == 0).all() and torch.isfinite(psi).all()
    assert viol == 0


def test_t1_ties_h3(rng):
    """H3: k* can fall by more than one at a tie; the stable sort keeps lower indices first."""
    z = torch.tensor([[1.0, 1.0, 1.0, -5.0]], dtype=torch.float64)
    p = sparsemax_masked(z, torch.ones_like(z, dtype=torch.bool))
    assert torch.allclose(p, torch.tensor([[1 / 3, 1 / 3, 1 / 3, 0.0]], dtype=torch.float64))
    z = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    p = sparsemax_masked(z, torch.ones_like(z, dtype=torch.bool))
    assert torch.allclose(p, torch.full((1, 4), 0.25, dtype=torch.float64))


# ----------------------------------------------------------------------------- T2
def test_t2_recovery_interval_sharp(rng):
    """Thm. 2: supp(p) == T iff tau in [1/(W + m delta), 1/W); below: proper superset; at/above: proper subset."""
    v_in = v_lo = v_hi = 0; n_in = n_lo = n_hi = 0
    for _ in range(4000):
        nq = int(rng.integers(4, 40)); m = int(rng.integers(1, min(8, nq)))
        s, T = planted_column(rng, nq, m)
        lo, hi, delta, W = interval(s, T)
        assert lo < hi
        st = torch.as_tensor(s, dtype=torch.float64)[None]; mask = torch.ones(1, nq, dtype=torch.bool)
        span = (min(hi, lo * 50.0) - lo)
        for tau in lo + span * rng.random(4):
            p = sparsemax_masked(tau * st, mask)[0].numpy(); n_in += 1
            if set(np.nonzero(p)[0].tolist()) != T: v_in += 1
        for tau in lo * rng.uniform(0.05, 0.95, size=2):
            p = sparsemax_masked(tau * st, mask)[0].numpy(); n_lo += 1
            if not (T < set(np.nonzero(p)[0].tolist())): v_lo += 1
        if W > 0:
            for tau in hi * (1.0 + 3.0 * rng.random(2)):
                p = sparsemax_masked(tau * st, mask)[0].numpy(); n_hi += 1
                if not (set(np.nonzero(p)[0].tolist()) < T): v_hi += 1
    assert v_in == 0 and v_lo == 0 and v_hi == 0, (v_in, n_in, v_lo, n_lo, v_hi, n_hi)


def test_t2_on_a_scope(rng):
    """App. E: the same interval on E_.j with delta over the scope; the pool outside never enters."""
    v = 0; n = 0
    for _ in range(3000):
        ne = int(rng.integers(3, 24)); m = int(rng.integers(1, ne))
        sE = np.empty(ne); sE[:m] = rng.random(m)
        d = 0.2 + rng.random(); sE[m:] = -d - rng.random(ne - m)
        T = set(range(m))
        lo, hi, delta, W = interval(sE, T)
        nq = ne + int(rng.integers(0, 40))
        # embed the scope in a longer column with a pool of large distractors OUTSIDE the scope
        z = np.concatenate([sE, rng.normal(3.0, 1.0, size=nq - ne)])
        mask = torch.zeros(1, nq, dtype=torch.bool); mask[0, :ne] = True
        zt = torch.as_tensor(z, dtype=torch.float64)[None]
        for tau in lo + (min(hi, lo * 50.0) - lo) * rng.random(3):
            p = sparsemax_masked(tau * zt, mask)[0].numpy(); n += 1
            if set(np.nonzero(p)[0].tolist()) != T: v += 1
    assert v == 0, f"{v}/{n}"


# ----------------------------------------------------------------------------- T13
def test_t13_gradcheck_sparsemax(rng):
    ok = True
    for _ in range(20):
        L = int(rng.integers(2, 9))
        z = (torch.randn(3, L, dtype=torch.float64) * 2).requires_grad_(True)
        mask = torch.rand(3, L) < 0.8; mask[:, 0] = True
        ok &= torch.autograd.gradcheck(lambda zz: SparsemaxMasked.apply(zz, mask), (z,), eps=1e-6, atol=1e-5)
    assert ok


def test_t13_gradcheck_proj_le(rng):
    """the projections are differentiable a.e. in v AND in the quota s (the tau_i path)."""
    ok = True
    for _ in range(20):
        L = int(rng.integers(2, 8))
        v = (torch.randn(2, L, dtype=torch.float64).abs() * 0.4).requires_grad_(True)
        s = (torch.rand(2, dtype=torch.float64) * 0.8 + 0.05).requires_grad_(True)
        mask = torch.rand(2, L) < 0.8; mask[:, 0] = True
        ok &= torch.autograd.gradcheck(lambda vv, ss: proj_le_masked(vv, ss, mask)[0], (v, s), eps=1e-6, atol=1e-5)
    assert ok


def test_proj_le_zero_quota_backward_is_finite():
    """H11: the zero-quota row (the MAJORITY row under random scores) must have a finite (zero) gradient."""
    v = torch.zeros(3, 4, requires_grad=True)
    s = torch.zeros(3, requires_grad=True)
    mask = torch.ones(3, 4, dtype=torch.bool)
    a, th = proj_le_masked(v, s, mask)
    (a.sum() + th.sum()).backward()
    assert torch.isfinite(v.grad).all() and torch.isfinite(s.grad).all()
    assert (a == 0).all() and (th == 0).all()


def test_entmax_alpha15_properties(rng):
    """B4's alpha = 1.5 bisection: distribution on the mask, exact zeros off it, support monotone in tau."""
    for _ in range(100):
        L = int(rng.integers(3, 25)); z = torch.randn(1, L) * 2
        mask = torch.rand(1, L) < 0.8; mask[:, 0] = True
        taus = np.sort(rng.uniform(0.05, 20.0, size=8))
        sizes = []
        for t in taus:
            p = entmax_masked(t * z, mask, 1.5)
            assert (p[~mask] == 0).all()
            assert abs(float(p.sum()) - 1) < 1e-4
            sizes.append(int((p > 1e-12).sum()))
        assert all(sizes[i + 1] <= sizes[i] for i in range(len(sizes) - 1)), sizes


def test_summation_tolerance_is_indexed_on_the_axis_being_summed():
    """tol_sum scaled with n_q for every invariant, while INV-3 / INV-7 / SAN-1 sum a ROW (n_k terms).  Square
    self-attention hides it; at n_k > n_q -- a decode row, a logits_to_keep pass -- it is too tight, and at
    n_q > n_k it is ~100x looser than needed and would swallow a real error (review 30372ae C)."""
    import torch
    from marsea.invariants import invariant_report
    from marsea.normalizer import Diagnostics

    def report(n_q, n_k, san_error=0.0):
        vis = torch.ones(1, 1, n_q, n_k, dtype=torch.bool)
        A_sm = torch.full((1, 1, n_q, n_k), 1.0 / n_k)
        A_sm[..., 0] += san_error                                   # push the row sum off one by exactly this much
        E = torch.zeros(1, 1, n_q, n_k, dtype=torch.bool)
        d = Diagnostics(E=E, A=A_sm, A_sm=A_sm, Atil=A_sm, a1=A_sm, cbar_j=A_sm.sum(-2), cbar_i=(A_sm * E).sum(-1),
                        tau_j=torch.ones(1, 1, n_k), tau_i=torch.ones(1, 1, n_q), nu=torch.ones(1, 1, n_k))
        return invariant_report(A_sm, d, vis)

    # n_k > n_q: the row sum carries sqrt(n_k) rounding and must not be judged by an n_q-sized bound
    assert report(1, 16384, san_error=1.2e-5)["SAN-1"][0], "SAN-1 is too tight at n_k >> n_q"
    assert report(64, 8192, san_error=1.0e-5)["SAN-1"][0], "SAN-1 is too tight at n_k > n_q"
    # n_q > n_k: the bound must NOT have grown with n_q -- a real 1e-4 error has to fail
    assert not report(8192, 64, san_error=1e-4)["SAN-1"][0], "SAN-1 is loose enough to swallow a real error"
    # INV-1 genuinely sums a column, so it keeps the n_q scaling
    import inspect
    from marsea import invariants
    src = inspect.getsource(invariants.invariant_report)
    assert "tol_col = unit * max(1.0, (n_q / 64.0) ** 0.5)" in src and "tol_sum = unit * max(1.0, (n_k / 64.0) ** 0.5)" in src
    assert 'out["INV-1"] = (bool((err1 <= tol_col)' in src
