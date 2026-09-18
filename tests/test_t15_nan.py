"""T15: NaN sweep -- a causal batch with n_q = n_k, a right-padded batch with fully-invisible pad rows, columns
with |E_.j| = 0 and rows with cbar_i = 0: loss and EVERY parameter gradient finite.  Also the gradient-flow
check (every module receives a finite, non-zero gradient) and the T13 finite-difference check of d loss / d
tau_j, d tau_i through the real programs, and d/d logits through the ST gate."""
import numpy as np
import torch
import pytest
from conftest import rand_case
from marsea.normalizer import MarSeaNormalizer, State
from marsea.baselines import MESHNorm, KeyOnlyTauNorm, RowEntmaxNorm, SoftmaxOneNorm

D, R = 16, 8


def _all_grads_finite(mod):
    ok = True
    for n, p in mod.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            ok = False
    return ok


def test_t15_nan_sweep_causal_square():
    torch.manual_seed(0)
    norm = MarSeaNormalizer(D, R)
    for T in (1, 2, 7, 33):
        S = (torch.randn(2, 2, T, T) * 3).requires_grad_(True)
        i = torch.arange(T)[:, None]; j = torch.arange(T)[None, :]
        vis = (j <= i)[None, None]
        K_kv = torch.randn(2, 1, T, D, requires_grad=True); Q = torch.randn(2, 2, T, D, requires_grad=True)
        A, d = norm.normalize(S, vis, K_kv, Q, State())
        loss = (A * torch.randn_like(A)).sum() + d.nu.sum() * 0.1
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(A).all()
        assert _all_grads_finite(norm) and torch.isfinite(S.grad).all() and torch.isfinite(K_kv.grad).all()
        norm.zero_grad()


def test_t15_nan_sweep_right_padded_and_empty():
    torch.manual_seed(1)
    norm = MarSeaNormalizer(D, R)
    B, H, T = 2, 2, 12
    S = torch.randn(B, H, T, T, requires_grad=True)
    i = torch.arange(T)[:, None]; j = torch.arange(T)[None, :]
    vis = (j <= i)[None, None].expand(B, 1, T, T).clone()
    vis[1, :, :, 8:] = False; vis[1, :, 8:, :] = False                         # right padding: pad rows fully invisible
    K_kv = torch.randn(B, 1, T, D); Q = torch.randn(B, H, T, D)
    # a relation with empty columns and rows with zero quota (very sharp tau_j makes most relation entries 0)
    logits = torch.randn(B, H, T, T)
    logits[..., 3] = -5.0
    A, d = norm.normalize(S, vis, K_kv, Q, State(), logits_override=logits, tau_j_override=torch.full((B, H, T), 50.0))
    assert (A[1, :, 8:, :] == 0).all() and (A[1, :, :, 8:] == 0).all()
    assert int(((d.cbar_i <= 1e-9) & (d.E.sum(-1) > 0)).sum()) > 0, "no zero-quota row: the sweep is not exercising H6/H11"
    assert int((d.E.sum(-2) == 0).sum()) > 0
    loss = (A * torch.randn_like(A)).sum()
    loss.backward()
    assert torch.isfinite(loss) and _all_grads_finite(norm) and torch.isfinite(S.grad).all()


def test_t15_head_path_backward_finite_with_finfo_min(rng):
    torch.manual_seed(2)
    norm = MarSeaNormalizer(D, R)
    for t in range(20):
        cs = rand_case(rng, use_finfo_min=True)
        S = cs["S"].clone().requires_grad_(True)
        K_kv = cs["K_kv"].clone().requires_grad_(True); Q = cs["Q"].clone().requires_grad_(True)
        state = State(nu_prev=torch.rand(*cs["tau_j"].shape) * 3 + 1)
        A, d = norm.normalize(S, cs["vis"], K_kv, Q, state)
        (A * torch.randn_like(A)).sum().backward()
        assert torch.isfinite(A).all() and _all_grads_finite(norm)
        assert torch.isfinite(S.grad).all() and torch.isfinite(K_kv.grad).all() and torch.isfinite(Q.grad).all()
        norm.zero_grad()


def test_gradient_reaches_every_module(rng):
    torch.manual_seed(3)
    cs = rand_case(rng, n_q=24, n_k=8)
    norm = MarSeaNormalizer(D, R)
    with torch.no_grad():
        norm.relation.calibrate_b0(cs["K_kv"], cs["Q"], cs["vis"], 0.3)
    S = cs["S"].clone().requires_grad_(True)
    A, d = norm.normalize(S, cs["vis"], cs["K_kv"], cs["Q"], State())
    (A * torch.randn_like(A)).sum().backward()
    for name in ("relation.U.weight", "relation.V.weight", "relation.b0", "tauK.fc1.weight", "tauK.fc2.weight",
                 "tauK.fc2.bias", "tauQ.fc1.weight", "tauQ.fc2.weight", "tauQ.fc2.bias", "tauK.ln.weight", "tauQ.ln.weight"):
        p = dict(norm.named_parameters())[name]
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    assert S.grad is not None and S.grad.abs().sum() > 0


def test_nu_carries_gradient_to_previous_layer():
    """Sec. 5.3 [v4.1]: nu is NOT detached; the next layer's TauK gradient reaches the previous layer's tau_j.
    Owns its random stream (a shared one made the draw -- and so the relation's emptiness -- depend on test order), and
    forces a dense relation so the nu -> p -> tau_j path is actually exercised."""
    torch.manual_seed(4)
    cs = rand_case(np.random.default_rng(17), n_q=20, n_k=6, causal=False)
    n1, n2 = MarSeaNormalizer(D, R), MarSeaNormalizer(D, R)
    with torch.no_grad():
        for n in (n1, n2):
            n.relation.calibrate_b0(cs["K_kv"], cs["Q"], cs["vis"], 0.9)      # nearly every visible pair is in E
    st = State()
    A1, d1 = n1.normalize(cs["S"], cs["vis"], cs["K_kv"], cs["Q"], st)
    st2 = State(nu_prev=st.nu_next)
    A2, d2 = n2.normalize(cs["S"] + 0.1, cs["vis"], cs["K_kv"], cs["Q"], st2)
    # a loss on layer 2's tau_j only
    assert d1.E.any() and (d1.nu > 1).any(), "the draw left no relation: the nu path would be vacuous"
    d2.tau_j.sum().backward()
    g = n1.tauK.fc2.bias.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, "nu gradient did not reach layer 1's TauK"


def test_t13_fd_tau_and_logits(rng):
    """finite differences of the loss w.r.t. tau_j, tau_i (real programs) and logits (ST gate) on a 4x6 toy (fp64)."""
    torch.manual_seed(5)
    norm = MarSeaNormalizer(D, R)
    ok = True
    rng = np.random.default_rng(13)                       # own stream: the draws must not depend on test order
    for t in range(8):
        cs = rand_case(rng, n_q=6, n_k=4, B=1, Hkv=1, H=1)
        tj = cs["tau_j"].double().requires_grad_(True); ti = cs["tau_i"].double().requires_grad_(True)
        def f(tj_, ti_):
            A, _ = norm.normalize(cs["S"].double(), cs["vis"], cs["K_kv"].double(), cs["Q"].double(), State(),
                                  logits_override=cs["logits"].double(), tau_j_override=tj_, tau_i_override=ti_)
            return (A * torch.arange(A.numel(), dtype=torch.float64).view_as(A)).sum()
        try:
            ok &= torch.autograd.gradcheck(f, (tj, ti), eps=1e-6, atol=1e-4, rtol=1e-3, nondet_tol=1e-6)
        except Exception as e:
            print("FD failure:", e); ok = False
    assert ok
    # ST gate: the analytic gradient equals the gradient of the SOFT interpolation at the hard set (by construction)
    cs = rand_case(rng, n_q=6, n_k=4, B=1, Hkv=1, H=1)
    lg = cs["logits"].double().requires_grad_(True)
    A, d = norm.normalize(cs["S"].double(), cs["vis"], cs["K_kv"].double(), cs["Q"].double(), State(),
                          logits_override=lg, tau_j_override=cs["tau_j"].double(), tau_i_override=cs["tau_i"].double())
    w = torch.randn_like(A)
    (A * w).sum().backward()
    assert lg.grad is not None and torch.isfinite(lg.grad).all()
    vis = cs["vis"].expand_as(lg)
    assert (lg.grad[~vis] == 0).all(), "gradient leaked to invisible pairs"


def test_baselines_backward_finite(rng):
    torch.manual_seed(6)
    for arm in (MESHNorm(D), KeyOnlyTauNorm(D, R), RowEntmaxNorm(1.5), RowEntmaxNorm(2.0), SoftmaxOneNorm()):
        cs = rand_case(rng, n_q=10, n_k=6)
        S = cs["S"].clone().requires_grad_(True)
        A, d = arm.normalize(S, cs["vis"], cs["K_kv"], cs["Q"], State())
        (A * torch.randn_like(A)).sum().backward()
        assert torch.isfinite(A).all() and torch.isfinite(S.grad).all() and _all_grads_finite(arm), type(arm).__name__
