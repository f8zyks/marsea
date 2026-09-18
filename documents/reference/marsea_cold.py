"""
marsea_cold.py -- cold implementation of spec.md (v3) Secs. 3, 4, 5, 8.
PyTorch, CPU, fp32.  All program tensors are [B, H, n_q, n_k].
Written from the spec text; verify_all.py was NOT consulted for the program logic.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_PAD = -1e30          # spec Sec. 3: pad sorts last, is excluded from the prefix count


def _fp(x: torch.Tensor) -> torch.Tensor:
    """H1: programs run in fp32 (half types upcast); fp64 is kept for gradcheck."""
    return x if x.dtype == torch.float64 else x.float()

TAU_MIN = 0.05           # spec Sec. 5.2
T_ST = 1.0               # spec Sec. 4.3
NU_EPS = 1e-12           # spec Sec. 5 'eps' -- value not given (ambiguity #3)
STD_EPS = 1e-6           # column_stats std clamp (ambiguity #1)


# ============================================================================ Sec. 3 primitives
def sorted_prefix_stat(z: torch.Tensor, mask: torch.Tensor):
    """Lem. 1's G and k*, batched over leading dims, ragged via `mask` on the last dim.
    z    [..., L] fp32 (values at ~mask are ignored)
    mask [..., L] bool
    returns kstar [...] (0 if no valid entry), psi [...], G [..., L] sorted-order (+inf at pads),
            u [..., L] sorted values, mvalid [..., L] validity in sorted order.
    """
    z = _fp(z)
    zp = z.masked_fill(~mask, NEG_PAD)                           # pad, not -inf (spec Sec. 3)
    u, order = torch.sort(zp, dim=-1, descending=True, stable=True)   # H8: stable
    mvalid = torch.gather(mask, -1, order)
    uv = torch.where(mvalid, u, torch.zeros_like(u))            # pads contribute 0 to cs
    cs = torch.cumsum(uv, dim=-1)
    k = torch.cumsum(mvalid.to(z.dtype), dim=-1)                # prefix COUNT of valid entries
    G = cs - k * u                                              # G(k) = sum_{r<=k} (z_(r) - z_(k))
    G = torch.where(mvalid, G, torch.full_like(G, float("inf")))
    cond = (G < 1.0) & mvalid
    kstar = (k * cond).amax(dim=-1)                             # max{k : G[k] < 1}  (1-based count)
    idx = (kstar.long() - 1).clamp_min(0).unsqueeze(-1)
    cs_k = torch.gather(cs, -1, idx).squeeze(-1)
    psi = (cs_k - 1.0) / kstar.clamp_min(1.0)
    return kstar, psi, G, u, mvalid


class _SparsemaxMasked(torch.autograd.Function):
    """sparsemax on the last dim restricted to `mask`; exact zeros; Jacobian diag(1_S) - 1_S 1_S^T/|S|."""

    @staticmethod
    def forward(ctx, z, mask):
        kstar, psi, _, _, _ = sorted_prefix_stat(z, mask)
        p = torch.clamp(_fp(z) - psi.unsqueeze(-1), min=0.0)
        p = torch.where(mask, p, torch.zeros_like(p))
        supp = (p > 0)
        ctx.save_for_backward(supp)
        ctx.psi = psi
        ctx.kstar = kstar
        return p

    @staticmethod
    def backward(ctx, grad_p):
        (supp,) = ctx.saved_tensors
        s = supp.to(grad_p.dtype)
        n = s.sum(-1, keepdim=True).clamp_min(1.0)
        mean = (grad_p * s).sum(-1, keepdim=True) / n
        return s * (grad_p - mean), None


def sparsemax_masked(z, mask):
    return _SparsemaxMasked.apply(z, mask)


def sparsemax_stats(z, mask):
    """p, kstar, psi -- forward-only helper for diagnostics."""
    with torch.no_grad():
        kstar, psi, _, _, _ = sorted_prefix_stat(z, mask)
    p = sparsemax_masked(z, mask)
    return p, kstar, psi


def entmax_bisect_masked(z, mask, alpha: float, iters: int = 50):
    """alpha-entmax by bisection on the threshold (spec Sec. 3, alpha != 2). Ambiguity #14."""
    z = _fp(z)
    am1 = alpha - 1.0
    zm = (am1 * z).masked_fill(~mask, float("-inf"))
    any_valid = mask.any(-1, keepdim=True)
    zmax = torch.where(any_valid, zm.amax(-1, keepdim=True), torch.zeros_like(zm[..., :1]))
    lo, hi = zmax - 1.0, zmax
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        p = torch.clamp(zm - mid, min=0.0) ** (1.0 / am1)
        s = p.sum(-1, keepdim=True)
        big = s > 1.0
        lo = torch.where(big, mid, lo)
        hi = torch.where(big, hi, mid)
    p = torch.clamp(zm - 0.5 * (lo + hi), min=0.0) ** (1.0 / am1)
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-30)
    return torch.where(mask, p, torch.zeros_like(p))


def ragged_entmax(z, mask, alpha: float = 2.0):
    if alpha == 2.0:
        return sparsemax_masked(z, mask)
    return entmax_bisect_masked(z, mask, alpha)


def proj_eq_masked(v, s, mask):
    """argmin .5||a-v||^2  s.t. sum a = s, a >= 0  on `mask`;  s [...] > 0.  = s * sparsemax(v/s)."""
    s_ = s.unsqueeze(-1)
    return s_ * sparsemax_masked(v / s_, mask)


def proj_le_masked(v, s, mask):
    """argmin .5||a-v||^2  s.t. sum a <= s, a >= 0  on `mask`.  s [...] >= 0 (s == 0 -> a = 0).
    returns a, theta (implied threshold: 0 if the cap was slack else v_j - a_j on supp(a))."""
    v = _fp(v)
    w = torch.clamp(v, min=0.0) * mask.to(v.dtype)
    sw = w.sum(-1)
    pos = s > 0
    s_safe = torch.where(pos, s, torch.ones_like(s))            # no division by 0 in either branch
    pe = proj_eq_masked(v, s_safe, mask)
    binds = (sw > s)
    a = torch.where(binds.unsqueeze(-1), pe, w)
    a = torch.where(pos.unsqueeze(-1), a, torch.zeros_like(a))  # H6 / ambiguity #2: proj_le(v, 0) = 0
    # implied threshold (spec Sec. 3)
    with torch.no_grad():
        on = (a > 0) & mask
        th = torch.where(on, v - a, torch.full_like(v, float("-inf"))).amax(-1)
        th = torch.where(on.any(-1) & binds, th, torch.zeros_like(th))
    return a, th


def implied_threshold(v, a, mask, binds):
    with torch.no_grad():
        on = (a > 0) & mask
        th = torch.where(on, v - a, torch.full_like(v, float("-inf"))).amax(-1)
        return torch.where(on.any(-1) & binds, th, torch.zeros_like(th))


# ============================================================================ helpers
def repeat_kv(x: torch.Tensor, g: int) -> torch.Tensor:
    """[B, H_kv, n, d] -> [B, H_kv*g, n, d], each KV head repeated g times consecutively (HF convention)."""
    if g == 1:
        return x
    B, Hkv, n, d = x.shape
    return x[:, :, None].expand(B, Hkv, g, n, d).reshape(B, Hkv * g, n, d)


def row_softmax(S: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
    S32 = _fp(S).masked_fill(~vis, float("-inf"))
    A = torch.softmax(S32, dim=-1)
    # a fully masked row is NaN; the spec is silent (ambiguity #6) -> return 0s
    anyv = vis.any(-1, keepdim=True)
    return torch.where(anyv, A, torch.zeros_like(A))


def column_stats(S: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
    """(max, second max, mean, std, max - second max, (max - mean)/std) over visible i. -> [B,H,n_k,6]"""
    S0 = _fp(S).masked_fill(~vis, 0.0)                          # never multiply -inf (ambiguity #17)
    Sm = _fp(S).masked_fill(~vis, float("-inf"))
    cnt = vis.sum(-2).float()                                   # [B,H,n_k]
    has = cnt > 0
    has2 = cnt > 1
    mx = Sm.amax(-2)
    mx = torch.where(has, mx, torch.zeros_like(mx))
    if S.shape[-2] >= 2:
        top2 = Sm.topk(2, dim=-2).values                        # [B,H,2,n_k]
        second = top2[..., 1, :]
    else:
        second = mx
    second = torch.where(has2, second, mx)                      # ambiguity #1: second := max if < 2 visible
    mean = S0.sum(-2) / cnt.clamp_min(1.0)
    var = (((S0 - mean.unsqueeze(-2)) ** 2) * vis).sum(-2) / cnt.clamp_min(1.0)
    std = torch.sqrt(var.clamp_min(0.0))
    gap = mx - second
    zsc = (mx - mean) / std.clamp_min(STD_EPS)
    out = torch.stack([mx, second, mean, std, gap, zsc], dim=-1)
    return torch.where(has.unsqueeze(-1), out, torch.zeros_like(out))


def row_summary(Atil, E, vis, cbar_i, Rtil):
    """(Rtil_i, cbar_i, max_j Atil_ij, entropy of the row's relation part) -> [B,H,n_q,4]"""
    mx = Atil.masked_fill(~vis, 0.0).amax(-1)
    q = (Atil * E) / Rtil.clamp_min(1e-30).unsqueeze(-1)
    ent = -(q * torch.log(q.clamp_min(1e-30))).sum(-1)
    ent = torch.where(Rtil > 0, ent, torch.zeros_like(ent))
    return torch.stack([Rtil, cbar_i, mx, ent], dim=-1)


def inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


# ============================================================================ Sec. 4 relation
class RelationHead(nn.Module):
    """e_ij = 1[<U k_j, V q_i>/sqrt(r) + b0 > 0] & vis.  Shared across heads.  U reads KV-head keys."""

    def __init__(self, d_head: int, r: int = 16, b0: float = 0.0):
        super().__init__()
        self.r = r
        self.U = nn.Linear(d_head, r, bias=False)
        self.V = nn.Linear(d_head, r, bias=False)
        nn.init.normal_(self.U.weight, std=1.0 / math.sqrt(d_head))   # N(0, 1/d_head) read as variance (#12)
        nn.init.normal_(self.V.weight, std=1.0 / math.sqrt(d_head))
        self.b0 = nn.Parameter(torch.tensor(float(b0)))

    def raw(self, K_kv, Q):
        g = Q.shape[1] // K_kv.shape[1]
        u = repeat_kv(self.U(K_kv), g)                    # [B,H,n_k,r]
        v = self.V(Q)                                     # [B,H,n_q,r]
        return torch.einsum("bhir,bhjr->bhij", v, u) / math.sqrt(self.r)

    def forward(self, K_kv, Q):
        return self.raw(K_kv, Q) + self.b0

    @torch.no_grad()
    def calibrate_b0(self, K_kv, Q, vis, rho0: float = 0.05):
        """spec Sec. 4.2: set b0 so that mean(E & vis)/mean(vis) == rho0 on this batch."""
        raw = self.raw(K_kv, Q)
        visb = vis.expand_as(raw)
        vals = raw[visb]
        q = torch.quantile(vals, 1.0 - rho0)
        self.b0.fill_(-q.item())
        return self.b0.item()


def straight_through(E: torch.Tensor, logits: torch.Tensor, vis: torch.Tensor, T_st: float = T_ST):
    """hard in forward, sigmoid in backward (spec Sec. 4.3); soft term restricted to vis (#7)."""
    soft = torch.sigmoid(logits / T_st) * vis.to(logits.dtype)
    return E.to(logits.dtype) + (soft - soft.detach())


# ============================================================================ Sec. 5.2 heads
class TauK(nn.Module):
    """MLP(concat(k_j[d], column_stats[6], log nu_prev[1]) -> 64 -> 1), GELU; tau = tau_min + softplus."""

    def __init__(self, d_head: int, hidden: int = 64, tau_min: float = TAU_MIN, init_tau: float = 1.0,
                 zero_field_inputs: bool = False):
        super().__init__()
        self.tau_min = tau_min
        self.zero_field_inputs = zero_field_inputs             # B3
        self.fc1 = nn.Linear(d_head + 6 + 1, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        nn.init.normal_(self.fc2.weight, std=0.01)              # ambiguity #8: bias should dominate at init
        self.fc2.bias.data.fill_(inv_softplus(max(init_tau - tau_min, 1e-3)))

    def forward(self, k_rep, stats, nu_prev):
        lognu = torch.log(nu_prev.clamp_min(1e-30)).unsqueeze(-1)
        if self.zero_field_inputs:
            stats = torch.zeros_like(stats)
            lognu = torch.zeros_like(lognu)
        x = torch.cat([k_rep, stats, lognu], dim=-1)
        out = self.fc2(F.gelu(self.fc1(x))).squeeze(-1)
        return self.tau_min + F.softplus(out)

    @torch.no_grad()
    def calibrate(self, S, vis):
        """set the last bias so tau_j ~= median_j 1/std_j at step 0 (ambiguity #8)."""
        st = column_stats(S, vis)[..., 3]
        st = st[vis.sum(-2) > 1]
        target = (1.0 / st.clamp_min(STD_EPS)).median().item()
        self.fc2.bias.fill_(inv_softplus(max(target - self.tau_min, 1e-3)))
        return target


class TauQ(nn.Module):
    """MLP(concat(q_i[d], row_summary[4]) -> 64 -> 1), GELU; tau = tau_min + softplus; init tau_i ~= 1."""

    def __init__(self, d_head: int, hidden: int = 64, tau_min: float = TAU_MIN, init_tau: float = 1.0):
        super().__init__()
        self.tau_min = tau_min
        self.fc1 = nn.Linear(d_head + 4, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        nn.init.normal_(self.fc2.weight, std=0.01)
        self.fc2.bias.data.fill_(inv_softplus(init_tau - tau_min))

    def forward(self, q, summary):
        x = torch.cat([q, summary], dim=-1)
        out = self.fc2(F.gelu(self.fc1(x))).squeeze(-1)
        return self.tau_min + F.softplus(out)


# ============================================================================ Sec. 5 normaliser
@dataclass
class Diagnostics:
    E: torch.Tensor = None
    cbar_j: torch.Tensor = None
    cbar_i: torch.Tensor = None
    tau_j: torch.Tensor = None
    tau_i: torch.Tensor = None
    kstar: torch.Tensor = None
    nu: torch.Tensor = None
    theta: torch.Tensor = None
    Rtil: torch.Tensor = None
    cap_binds: torch.Tensor = None
    # extras (ambiguity #13)
    psi_j: torch.Tensor = None
    A_sm: torch.Tensor = None
    c: torch.Tensor = None
    Atil: torch.Tensor = None
    a1: torch.Tensor = None
    p: torch.Tensor = None
    u: torch.Tensor = None
    supp_rel: torch.Tensor = None       # |supp(A[i, E_i.])| per row  (B5 trace)
    logits: torch.Tensor = None
    extra: dict = field(default_factory=dict)


@dataclass
class State:
    nu_prev: Optional[torch.Tensor] = None      # [B,H,n_k] from the previous patched layer; None -> 1.0
    nu_next: Optional[torch.Tensor] = None


def _broadcast_vis(vis, S):
    vis = vis.bool()
    while vis.dim() < 4:
        vis = vis.unsqueeze(0)
    return vis.expand(S.shape)


class MarSeaNormalizer(nn.Module):
    """spec Sec. 5 (dense).  normalize(S, vis, K_kv, Q, state) -> (A, Diagnostics)."""

    def __init__(self, d_head: int, r: int = 16, alpha: float = 2.0, tau_min: float = TAU_MIN,
                 key_only: bool = False):
        super().__init__()
        self.alpha = alpha
        self.relation = RelationHead(d_head, r)
        self.tauK = TauK(d_head, tau_min=tau_min, zero_field_inputs=key_only)
        self.tauQ = TauQ(d_head, tau_min=tau_min)

    def normalize(self, S, vis, K_kv, Q, state: Optional[State] = None, *,
                  logits_override=None, tau_j_override=None, tau_i_override=None):
        out_dtype = S.dtype
        vis = _broadcast_vis(vis, S)
        B, H, n_q, n_k = S.shape
        g_kv = H // K_kv.shape[1]
        S32 = _fp(S).masked_fill(~vis, float("-inf"))
        A_sm = row_softmax(S32, vis)                                   # fp32, exact 0 off vis
        c = A_sm.sum(-2)                                               # [B,H,n_k]; INV-2 derived

        logits = self.relation(K_kv, Q) if logits_override is None else logits_override
        E = (logits > 0) & vis
        g = straight_through(E, logits, vis)

        # ---- STEP 1 fan-out: inherited quota from A_sm
        cbar_j = (g * A_sm).sum(-2)                                    # [B,H,n_k]

        # ---- STEP 2 fan-out: shape + exact zeros over the relation only
        stats = column_stats(S32, vis)                                 # [B,H,n_k,6]
        nu_prev = state.nu_prev if (state is not None and state.nu_prev is not None) \
            else torch.ones(B, H, n_k, dtype=S32.dtype, device=S.device)
        if tau_j_override is None:
            k_rep = repeat_kv(_fp(K_kv), g_kv)                         # [B,H,n_k,d]
            tau_j = self.tauK(k_rep, stats, nu_prev)                   # [B,H,n_k]
        else:
            tau_j = torch.as_tensor(tau_j_override).to(S32.dtype).expand(B, H, n_k)
        zcol = (S32.masked_fill(~E, 0.0) * tau_j.unsqueeze(-2)).transpose(-1, -2)   # [B,H,n_k,n_q]
        ET = E.transpose(-1, -2)
        pT = ragged_entmax(zcol, ET, self.alpha)                       # p_ij = 0 off E_.j
        with torch.no_grad():
            kstar, psi_j, _, _, _ = sorted_prefix_stat(zcol, ET) if self.alpha == 2.0 else (None, None, 0, 0, 0)
        p = pT.transpose(-1, -2)                                       # [B,H,n_q,n_k]
        Atil = A_sm + g * (cbar_j.unsqueeze(-2) * p - A_sm)
        kstar_c = (p > 0).sum(-2)
        p2 = (p * p).sum(-2)
        nu = torch.where(kstar_c > 0, 1.0 / p2.clamp_min(NU_EPS), torch.ones_like(p2))   # ambiguity #3
        if state is not None:
            state.nu_next = nu.detach()

        # ---- STEP 1 fan-in: cap at one unit (identity on rows summing <= 1).  NOT a softmax.
        one = torch.ones(B, H, n_q, dtype=S32.dtype, device=S.device)
        a1, _ = proj_le_masked(Atil, one, vis)
        cbar_i = (g * a1).sum(-1)                                      # [B,H,n_q]
        Rtil = (E * Atil).sum(-1)
        cap_binds = Atil.sum(-1) > 1.0

        # ---- STEP 2 fan-in: ceiling at the quota, on the relation, target = Atil (pre-cap)  (ambiguity #4)
        if tau_i_override is None:
            summ = row_summary(Atil, E, vis, cbar_i, Rtil)
            tau_i = self.tauQ(_fp(Q), summ)                            # [B,H,n_q]
        else:
            tau_i = torch.as_tensor(tau_i_override).to(S32.dtype).expand(B, H, n_q)
        s_i = cbar_i / tau_i
        u, theta = proj_le_masked(Atil.masked_fill(~E, 0.0), s_i, E)   # <=, never proj_eq; u = 0 off E
        A = a1 + g * (tau_i.unsqueeze(-1) * u - a1)

        diag = Diagnostics(E=E, cbar_j=cbar_j, cbar_i=cbar_i, tau_j=tau_j, tau_i=tau_i, kstar=kstar_c,
                           nu=nu, theta=theta, Rtil=Rtil, cap_binds=cap_binds, psi_j=psi_j,
                           A_sm=A_sm, c=c, Atil=Atil, a1=a1, p=p, u=u,
                           supp_rel=((A > 0) & E).sum(-1), logits=logits)
        return A.to(out_dtype), diag


# ============================================================================ Sec. 8 baselines
class SoftmaxNorm(nn.Module):                                          # B0
    def normalize(self, S, vis, K_kv=None, Q=None, state=None):
        vis = _broadcast_vis(vis, S)
        return row_softmax(S, vis).to(S.dtype), Diagnostics()


class SoftmaxOneNorm(nn.Module):                                       # B1 (Miller 2023)
    def normalize(self, S, vis, K_kv=None, Q=None, state=None):
        vis = _broadcast_vis(vis, S)
        S32 = _fp(S).masked_fill(~vis, float("-inf"))
        zero = torch.zeros_like(S32[..., :1])
        lse = torch.logsumexp(torch.cat([S32, zero], dim=-1), dim=-1, keepdim=True)   # log(1 + sum exp)
        A = torch.exp(S32 - lse)
        return A.to(S.dtype), Diagnostics(extra={"row_sum": A.sum(-1)})


def sinkhorn_log(C, log_a, log_b, eps, iters: int = 30):
    """entropic OT in the log domain.  C [B,H,n_q,n_k] (+inf where invisible), log_a [B,H,n_k], log_b [B,H,n_q].
    rows ~ b (queries), cols ~ a (keys).  returns the plan P.  Unseen keys / empty rows carry no mass."""
    ninf = float("-inf")
    key_ok = torch.isfinite(log_a)
    row_ok = torch.isfinite(log_b)
    negC = (-C) / eps.unsqueeze(-1).unsqueeze(-1)                     # -inf where invisible
    # an all -inf reduction has a NaN backward: give dead rows/cols a finite stand-in and mask the duals
    dead = (~row_ok).unsqueeze(-1) | (~key_ok).unsqueeze(-2)
    negC_safe = negC.masked_fill(dead, 0.0)
    la = log_a.masked_fill(~key_ok, 0.0)
    lb = log_b.masked_fill(~row_ok, 0.0)
    f = torch.zeros_like(lb)
    gd = torch.zeros_like(la)
    for _ in range(iters):
        f = lb - torch.logsumexp(negC_safe + gd.unsqueeze(-2), dim=-1)
        f = f.masked_fill(~row_ok, ninf)
        gd = la - torch.logsumexp(negC_safe + f.masked_fill(~row_ok, 0.0).unsqueeze(-1), dim=-2)
        gd = gd.masked_fill(~key_ok, ninf)
    logP = negC + f.unsqueeze(-1) + gd.unsqueeze(-2)
    return torch.exp(logP.masked_fill(dead, ninf))


class MESHNorm(nn.Module):                                             # B2 (ambiguity #20)
    def __init__(self, d_head: int, hidden: int = 64, descent_steps: int = 4, descent_lr: float = 0.1,
                 sinkhorn_iters: int = 30):
        super().__init__()
        self.h_a = nn.Sequential(nn.Linear(d_head, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.h_b = nn.Sequential(nn.Linear(d_head, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.descent_steps = descent_steps
        self.descent_lr = descent_lr
        self.sinkhorn_iters = sinkhorn_iters

    def normalize(self, S, vis, K_kv, Q, state=None):
        vis = _broadcast_vis(vis, S)
        B, H, n_q, n_k = S.shape
        g_kv = H // K_kv.shape[1]
        S32 = _fp(S).masked_fill(~vis, float("-inf"))
        m = float(n_q)
        key_vis = vis.any(-2)                                          # [B,H,n_k]  key seen by some query
        row_vis = vis.any(-1)                                          # [B,H,n_q]
        la = repeat_kv(self.h_a(_fp(K_kv)), g_kv).squeeze(-1)          # [B,H,n_k]
        lb = self.h_b(_fp(Q)).squeeze(-1)                              # [B,H,n_q]
        log_a = math.log(m) + torch.log_softmax(la.masked_fill(~key_vis, float("-inf")), dim=-1)
        log_b = math.log(m) + torch.log_softmax(lb.masked_fill(~row_vis, float("-inf")), dim=-1)
        S0 = S32.masked_fill(~vis, 0.0)
        cnt = vis.sum((-1, -2)).float().clamp_min(1.0)
        mean = S0.sum((-1, -2)) / cnt
        std = torch.sqrt((((S0 - mean[..., None, None]) ** 2) * vis).sum((-1, -2)) / cnt)
        eps = (0.1 * std).clamp_min(1e-4)                              # per (b,h) slice
        C = (-S32).masked_fill(~vis, float("inf"))
        # cost descent on the plan entropy, gradient through the unrolled Sinkhorn
        create = torch.is_grad_enabled()
        with torch.enable_grad():
            Cd = C.detach().requires_grad_(True) if not create else C
            for _ in range(self.descent_steps):
                Cv = Cd if Cd.requires_grad else Cd.requires_grad_(True)
                P = sinkhorn_log(Cv, log_a, log_b, eps, self.sinkhorn_iters)
                Hent = -(P * torch.log(P.clamp_min(1e-30))).sum()
                (grad,) = torch.autograd.grad(Hent, Cv, create_graph=create)
                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0).masked_fill(~vis, 0.0)
                Cd = Cv - self.descent_lr * grad
        A = sinkhorn_log(Cd, log_a, log_b, eps, self.sinkhorn_iters)
        return A.to(S.dtype), Diagnostics(extra={"row_sum": A.sum(-1), "col_sum": A.sum(-2),
                                                 "b": log_b.exp(), "a": log_a.exp(), "eps": eps})


class KeyOnlyTauNorm(MarSeaNormalizer):                                # B3
    def __init__(self, d_head: int, r: int = 16, alpha: float = 2.0, tau_min: float = TAU_MIN):
        super().__init__(d_head, r, alpha, tau_min, key_only=True)


class RowEntmaxNorm(nn.Module):                                        # B4
    def __init__(self, alpha: float = 2.0):
        super().__init__()
        self.alpha = alpha

    def normalize(self, S, vis, K_kv=None, Q=None, state=None):
        vis = _broadcast_vis(vis, S)
        S32 = _fp(S).masked_fill(~vis, 0.0)
        A = ragged_entmax(S32, vis, self.alpha)
        return A.to(S.dtype), Diagnostics(extra={"supp": (A > 0).sum(-1)})


class MatchedSparsityNorm(nn.Module):                                  # B5 (two-pass; ambiguity #21)
    """trace: (E [B,H,n_q,n_k] bool, k [B,H,n_q] long) recorded from the paired MarSea run."""

    def normalize(self, S, vis, K_kv=None, Q=None, state=None, *, trace=None):
        assert trace is not None, "B5 needs the MarSea trace (E, k) for this batch"
        E, k = trace
        vis = _broadcast_vis(vis, S)
        A_sm = row_softmax(S, vis)
        n_k = S.shape[-1]
        rel = A_sm.masked_fill(~E, -1.0)
        order = torch.argsort(rel, dim=-1, descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(-1, order, torch.arange(n_k, device=S.device).expand_as(order))
        keep = E & (rank < k.unsqueeze(-1))
        rel_mass = (A_sm * E).sum(-1, keepdim=True)
        kept_mass = (A_sm * keep).sum(-1, keepdim=True)
        scale = torch.where(kept_mass > 0, rel_mass / kept_mass.clamp_min(1e-30), torch.zeros_like(kept_mass))
        A = torch.where(E, A_sm * keep * scale, A_sm)
        return A.to(S.dtype), Diagnostics(extra={"supp_rel": (keep).sum(-1)})
