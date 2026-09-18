"""T7 (prop:prefix at SEALED tau_j; frozen-prefix over-admits, never under-admits; a live head is expected to
fail and that failure is logged), T8 (prop:tournament: hierarchical == flat whenever K_ret >= k*), and the
decode path: token-for-token agreement of decode_step with the dense re-solve on the emitted row."""
import numpy as np
import torch
import pytest
from conftest import rand_case
from marsea.primitives import sparsemax_masked, sorted_prefix_stat
from marsea.causal import hierarchical_topk, FrozenPrefixCache, decode_step, prefix_psi_trace
from marsea.normalizer import MarSeaNormalizer, State, row_softmax
from marsea.relation import repeat_kv

D, R = 16, 8


def test_t7_psi_monotone_and_exclusion_permanent_at_sealed_tau(rng):
    v = 0; n = 0
    for _ in range(1500):
        T = int(rng.integers(4, 80)); tau = float(rng.uniform(0.3, 6.0))
        z = torch.randn(T, dtype=torch.float64) * float(rng.uniform(0.3, 3.0))
        keep = torch.rand(T) < float(rng.uniform(0.2, 1.0)); keep[0] = True       # arrival-sealed relation
        psis, supports = prefix_psi_trace(z, keep, tau)
        prev = -np.inf; excluded = set()
        for t in range(T):
            if psis[t] is None: continue
            n += 1
            if psis[t] < prev - 1e-9: v += 1                                   # (i) psi non-decreasing
            prev = psis[t]
            visible_kept = {i for i in range(t + 1) if keep[i]}
            now0 = visible_kept - supports[t]
            if excluded & supports[t]: v += 1                                    # (ii) exclusion permanent
            excluded |= now0
        # (iii) the frozen-prefix support (entry fixed when first visible) contains the final support
        frozen = {t for t in range(T) if keep[t] and t in supports[t]}
        if not supports[-1] <= frozen: v += 1
    assert v == 0, f"{v} violations / {n}"


def test_t7_live_head_fails_and_is_logged(rng, capsys):
    """with a LIVE TauK re-reading the growing column, psi_j can decrease -- prop:prefix's hypothesis (sealing) is
    necessary; this is reported, not asserted as a pass (spec T7 note [v4])."""
    torch.manual_seed(0)
    norm = MarSeaNormalizer(D, R)
    dec = 0; n = 0
    for _ in range(40):
        n_q = int(rng.integers(6, 30)); n_k = int(rng.integers(1, 6))
        S = torch.randn(1, 1, n_q, n_k) * 2.0
        K_kv = torch.randn(1, 1, n_k, D); Q = torch.randn(1, 1, n_q, D)
        prev = torch.full((n_k,), float("-inf"))
        for t in range(n_q):
            vis = torch.zeros(n_q, n_k, dtype=torch.bool); vis[:t + 1] = True
            with torch.no_grad():
                _, d = norm.normalize(S, vis, K_kv, Q, State(), logits_override=torch.ones(1, 1, n_q, n_k),
                                      tau_i_override=torch.ones(1, 1, n_q))
            psi = d.psi_j[0, 0]
            ne = d.E[0, 0].sum(0) > 0
            n += int(ne.sum()); dec += int(((psi < prev - 1e-6) & ne).sum())
            prev = torch.where(ne, psi, prev)
    print(f"\n[T7 note] live TauK: psi_j decreased on {dec} of {n} (column, t) pairs -- expected; sealing is the hypothesis")
    assert n > 0


def test_t8_hierarchical_equals_flat_when_K_ge_kstar(rng):
    v = 0; kmax = 0
    for _ in range(400):
        L = int(rng.choice([64, 128, 256, 512, 1024]))
        z = torch.as_tensor(rng.normal(scale=2.0, size=L) * rng.uniform(0.05, 20.0), dtype=torch.float64)[None]
        mask = torch.ones(1, L, dtype=torch.bool)
        p_flat = sparsemax_masked(z, mask)
        kstar, _, _, _, _ = sorted_prefix_stat(z, mask); kk = int(kstar); kmax = max(kmax, kk)
        K = kk + int(rng.integers(0, 20))
        Bk = int(rng.choice([8, 16, 32, 64]))
        p_h, ks_h, trunc = hierarchical_topk(z, mask, K_ret=K, block_size=Bk)
        if not torch.allclose(p_h, p_flat, atol=1e-12): v += 1
        if int(ks_h) != kk: v += 1
    assert v == 0, f"{v} violations (largest k* seen {kmax})"


def test_t8_under_retention_gives_prefix_of_targets(rng):
    """Prop. 22: K < m retains the top-K targets (P = 1, R = K/m, target mass 1)."""
    from conftest import planted_column, interval
    v = 0; n = 0
    for _ in range(600):
        nq = int(rng.integers(32, 256)); m = int(rng.integers(3, 12))
        s, T = planted_column(rng, nq, m)
        lo, hi, delta, W = interval(s, T)
        if not W > 0: continue
        tau = lo + (hi - lo) * rng.random()
        K = int(rng.integers(1, m))
        z = torch.as_tensor(tau * s, dtype=torch.float64)[None]
        p, _, _ = hierarchical_topk(z, torch.ones(1, nq, dtype=torch.bool), K_ret=K, block_size=int(rng.choice([8, 16, 32])))
        Ssup = set(torch.nonzero(p[0] > 0).flatten().tolist())
        top = set(np.array(sorted(T))[np.argsort(s[sorted(T)])[::-1][:K]].tolist())
        n += 1
        if Ssup != top or abs(float(p[0][sorted(T)].sum()) - 1) > 1e-9: v += 1
    assert v == 0 and n > 0


def test_t8_batched_masked_columns(rng):
    """batched [B,H,n_k,n_q] with ragged masks (the shape the normaliser uses)."""
    for _ in range(50):
        B, H, nk, nq = 2, 3, 4, int(rng.integers(20, 300))
        z = torch.randn(B, H, nk, nq, dtype=torch.float64) * float(rng.uniform(0.5, 10))
        mask = torch.rand(B, H, nk, nq) < 0.7
        p_flat = sparsemax_masked(z, mask)
        kstar, _, _, _, _ = sorted_prefix_stat(z, mask)
        K = int(kstar.max()) + 2
        p_h, ks_h, _ = hierarchical_topk(z, mask, K_ret=K, block_size=32)
        assert torch.allclose(p_h, p_flat, atol=1e-12)
        assert (ks_h == kstar).all()


def test_decode_step_matches_dense_resolve_on_emitted_row(rng):
    """frozen-prefix decode (Sec. 6.4): at step t with tau_j frozen and cbar_j incremental, row t's Stage-1 entries
    equal the dense normaliser's on the prefix [0..t] with the SAME frozen tau_j; and the fan-in on row t is the
    dense one.  (K_ret large enough that no truncation occurs.)"""
    torch.manual_seed(1)
    norm = MarSeaNormalizer(D, R)
    worst = 0.0
    for _ in range(30):
        T = int(rng.integers(6, 24)); n_pre = int(rng.integers(3, T - 1)); Hkv, g = 1, 2; H = Hkv * g
        Q = torch.randn(1, H, T, D); K_kv = torch.randn(1, Hkv, T, D)
        k_rep = repeat_kv(K_kv, g)
        S_full = torch.einsum("bhid,bhjd->bhij", Q, k_rep) * D ** -0.5
        # prefill on [0..n_pre)
        i = torch.arange(n_pre)[:, None]; j = torch.arange(n_pre)[None, :]
        vis_pre = (j <= i)[None, None]
        with torch.no_grad():
            _, dpre = norm.normalize(S_full[..., :n_pre, :n_pre], vis_pre, K_kv[..., :n_pre, :], Q[..., :n_pre, :], State())
        cache = FrozenPrefixCache(K_ret=64)
        cache.init_from_prefill(S_full[..., :n_pre, :n_pre].masked_fill(~vis_pre, float("-inf")), dpre)
        tau_frozen = dpre.tau_j.clone()
        for t in range(n_pre, T):
            n_k = t + 1
            S_row = S_full[..., t:t + 1, :n_k]
            vis_row = torch.ones(1, 1, 1, n_k, dtype=torch.bool)
            with torch.no_grad():
                A_row, info = decode_step(norm, cache, S_row, vis_row, K_kv[..., :n_k, :], Q[..., t:t + 1, :])
                # dense reference on the prefix with the frozen tau for old keys and the cache's tau for new keys
                i = torch.arange(n_k)[:, None]; jj = torch.arange(n_k)[None, :]
                vis_d = (jj <= i)[None, None]
                _, dd = norm.normalize(S_full[..., :n_k, :n_k], vis_d, K_kv[..., :n_k, :], Q[..., :n_k, :], State(),
                                       tau_j_override=cache.tau_j)
            worst = max(worst, float((A_row[0, :, 0] - dd.A[0, :, t]).abs().max()))
            worst = max(worst, float((info["Atil"][0, :, 0] - dd.Atil[0, :, t]).abs().max()))
    assert worst < 1e-5, worst
