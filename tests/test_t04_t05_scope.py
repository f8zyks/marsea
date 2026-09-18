"""T4 (prop:scopemono) and T5 (cor:capacity) on the REAL tensor path.  Spec Sec. 11 says port every block of
verify_all.py to the tensor implementation; these two existed only as numpy (read-through 2026-09-11, Sec. 3), and T5
is the one acceptance test that states the paper's headline -- E1's degree = m_j and P_j = 1 as |E_.j| grows -- in code.
T6's remaining halves (prop:stagecomp (ii) precision preserved, (iv) false negatives stick) are covered here too."""
import numpy as np
import torch
from conftest import interval
from marsea.normalizer import MarSeaNormalizer, State

D, R = 16, 8


def _column_case(s: np.ndarray, E_mask: torch.Tensor, tau_j: float, tau_i: float = 1.0):
    """one key column through the real normaliser: S [n_q, 1], relation E_mask [n_q], oracle tau_j."""
    n_q = len(s)
    S = torch.as_tensor(s, dtype=torch.float32).view(1, 1, n_q, 1)
    vis = torch.ones(1, 1, n_q, 1, dtype=torch.bool)
    logits = torch.where(E_mask.view(1, 1, n_q, 1), torch.ones(1, 1, n_q, 1), -torch.ones(1, 1, n_q, 1))
    norm = _column_case.norm
    with torch.no_grad():
        A, d = norm.normalize(S, vis, torch.zeros(1, 1, 1, D), torch.zeros(1, 1, n_q, D), State(),
                              logits_override=logits, tau_j_override=torch.tensor([[[tau_j]]]),
                              tau_i_override=torch.full((1, 1, n_q), tau_i))
    return A[0, 0, :, 0], d


_column_case.norm = MarSeaNormalizer(D, R)


def test_t5_capacity_degree_and_precision_as_the_relation_grows():
    """cor:capacity, cardinality half: m_j = 4 fixed, |E_.j| from 4 to 256, tau_j inside Thm. 2's interval.
    MarSea's fan-out DEGREE stays 4 and precision stays 1; the full-support comparator's degree is |E_.j| and its
    precision m_j/|E_.j| -- E1(a) on the deployed code path rather than on a numpy re-implementation."""
    MJ = 4
    rows = []
    for n in (4, 8, 16, 32, 64, 128, 256):
        rng = np.random.default_rng(400 + n)
        # the sink (position 0) is excluded from every relation by D-31, so the planted set starts at index 1
        s = np.concatenate([[0.0], rng.uniform(0, 1, size=n) * 0.9])
        s[1:1 + MJ] = 1.0 + rng.random(MJ)
        T = set(range(1, 1 + MJ))
        E = torch.zeros(len(s), dtype=torch.bool); E[1:] = True                      # the relation is the candidate set
        lo, hi, delta, W = interval(s[1:], {i - 1 for i in T})
        tau = lo + 0.5 * (min(hi, lo * 20.0) - lo)
        col, d = _column_case(s, E, tau)
        # the degree is measured ON THE RELATION (spec Sec. 12.4): off it the entries are the unmodified layer's, and the
        # sink row -- which D-31 keeps out of every relation -- carries its standard-attention weight by construction
        supp = set(torch.nonzero((d.Atil[0, 0, :, 0] > 0) & d.E[0, 0, :, 0]).flatten().tolist())
        P = len(supp & T) / len(supp)
        assert len(supp) == MJ, f"|E_.j| = {n}: degree {len(supp)} != m_j = {MJ}"
        assert P == 1.0 and int(d.kstar[0, 0, 0]) == MJ
        rows.append((n, len(supp), P, MJ / n))
    degrees = [r[1] for r in rows]
    slope = float(np.polyfit([r[0] for r in rows], degrees, 1)[0])
    assert abs(slope) < 1e-9, f"fan-out degree must not grow with |E_.j| (slope {slope})"
    assert [r[3] for r in rows][-1] < 0.02                                            # the comparator's identity falls as m/n


def test_t4_scope_monotonicity_delta_and_interval_containment():
    """prop:scopemono: shrinking the relation cannot lower delta_j, cannot narrow the relative interval width, and the
    smaller scope's interval CONTAINS the larger one's (given Assumption 2 on the larger).  Checked on the real path:
    both scopes are run through the normaliser and the endpoints are computed from its own diagnostics."""
    v_d = v_w = v_cont = 0; n = 0; rescued = 0
    for trial in range(400):
        rng = np.random.default_rng(1000 + trial)
        n_q = int(rng.integers(12, 60)); m = int(rng.integers(2, 6))
        s = np.concatenate([[0.0], rng.normal(size=n_q)])
        T = set(range(1, 1 + m)); s[1:1 + m] = 2.0 + rng.random(m)
        pool = list(range(1 + m, len(s)))
        E_big = sorted(T | set(rng.choice(pool, size=int(rng.integers(3, len(pool))), replace=False).tolist()))
        outer = [i for i in E_big if i not in T]
        E_small = sorted(T | set(rng.choice(outer, size=int(rng.integers(1, max(2, len(outer)))), replace=False).tolist()))
        Eb = torch.zeros(len(s), dtype=torch.bool); Eb[E_big] = True
        Es = torch.zeros(len(s), dtype=torch.bool); Es[E_small] = True
        # delta and W FROM THE SCOPE the program ran on
        W = float(sum(s[i] - min(s[k] for k in T) for i in T))
        d_big = min(s[i] for i in T) - max([s[i] for i in E_big if i not in T] or [-np.inf])
        d_small = min(s[i] for i in T) - max([s[i] for i in E_small if i not in T] or [-np.inf])
        n += 1
        if d_small < d_big - 1e-9: v_d += 1                                    # (i) delta non-decreasing
        if m * d_small < m * d_big - 1e-9: v_w += 1                            # (ii) relative width non-decreasing
        if d_big > 0:
            lo_b = 1.0 / (W + m * d_big); lo_s = 1.0 / (W + m * d_small)
            if not lo_s <= lo_b + 1e-9: v_cont += 1                            # (iii) interval containment
            # and the recovery the containment promises actually happens on the real path at the smaller scope's tau
            tau = lo_s + 0.5 * (min(1.0 / W if W > 0 else np.inf, lo_s * 20.0) - lo_s)
            _, ds = _column_case(s, Es, tau)
            supp = set(torch.nonzero((ds.Atil[0, 0, :, 0] > 0) & ds.E[0, 0, :, 0]).flatten().tolist())
            assert supp == T, f"scope recovery failed at trial {trial}: {supp} != {T}"
        if d_big <= 0 < d_small: rescued += 1                                  # (iv) the rescue direction
    assert v_d == 0 and v_w == 0 and v_cont == 0, (v_d, v_w, v_cont, n)
    assert rescued > 0, "no column was rescued by shrinking the scope: the test never exercised (iv)"


def test_t6_stage_composition_precision_preserved_and_false_negatives_stick():
    """prop:stagecomp (ii) and (iv) on the real eq:rowprog path: a Stage-1 column that was precise stays precise after
    Stage 2, and a target Stage 1 excluded is never restored by the row program."""
    v_prec = v_fn = 0; n = 0
    norm = MarSeaNormalizer(D, R)
    for trial in range(200):
        rng = np.random.default_rng(2000 + trial)
        n_q = int(rng.integers(8, 40)); n_k = int(rng.integers(2, 8))
        S = torch.as_tensor(rng.normal(size=(1, 1, n_q, n_k)), dtype=torch.float32)
        vis = torch.ones(1, 1, n_q, n_k, dtype=torch.bool)
        E = torch.as_tensor(rng.random((1, 1, n_q, n_k)) < rng.uniform(0.3, 0.9))
        logits = torch.where(E, torch.ones_like(S), -torch.ones_like(S))
        T = torch.as_tensor(rng.random((1, 1, n_q, n_k)) < 0.35) & E
        with torch.no_grad():
            A, d = norm.normalize(S, vis, torch.randn(1, 1, n_k, D), torch.randn(1, 1, n_q, D), State(),
                                  logits_override=logits,
                                  tau_j_override=torch.as_tensor(rng.uniform(0.3, 4.0, size=(1, 1, n_k)), dtype=torch.float32),
                                  tau_i_override=torch.as_tensor(rng.uniform(0.3, 4.0, size=(1, 1, n_q)), dtype=torch.float32))
        for j in range(n_k):
            Tj = T[0, 0, :, j]
            if not Tj.any(): continue
            S1 = (d.Atil[0, 0, :, j] > 0) & d.E[0, 0, :, j]
            S2 = (A[0, 0, :, j] > 0) & d.E[0, 0, :, j]
            n += 1
            P1 = float((S1 & Tj).sum()) / max(1, int(S1.sum()))
            P2 = float((S2 & Tj).sum()) / max(1, int(S2.sum()))
            if S1.any() and abs(P1 - 1) < 1e-9 and S2.any() and abs(P2 - 1) > 1e-9: v_prec += 1   # (ii)
            if ((Tj & ~S1) & S2).any(): v_fn += 1                                                  # (iv)
    assert v_prec == 0 and v_fn == 0, (v_prec, v_fn, n)
    assert n > 100
