"""
Sec. 5.2: field statistics and the two exclusivity heads.

column_stats(S, vis)[j] = (max, second max, mean, std, max - second max, (max - mean)/std) over visible i
    [v4] biased std over visible entries, clamped at 1e-6 in the ratio; second max := max when
    n_vis_j < 2 (gap 0); all six := 0 when n_vis_j = 0.  n_vis_j = 1 occurs on the LAST key of every
    causal block -- an unguarded std NaNs tau_j there and the whole loss with it (H12).
row_summary(Atil, E)[i] = (Rtil_i, cbar_i, max over VISIBLE j of Atil_ij, H(Atil[i,E_i.]/Rtil_i))
TauK : MLP(concat(LN(k_j), f(column_stats), f(log nu_prev)) -> 64 -> 1), GELU; tau = tau_min + softplus
TauQ : MLP(concat(LN(q_i), f(row_summary)) -> 64 -> 1), GELU;              tau = tau_min + softplus
f(x) = sign(x) log1p(|x|) on every scalar feature.  tau_min = 0.05.
Init: last-layer weight ~ N(0, 0.01^2) (output = bias +- O(0.01)); TauK bias CALIBRATED on a
warm-up batch to softplus^-1(median_j 1/std_j - tau_min); TauQ bias = softplus^-1(1 - tau_min).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

TAU_MIN = 0.05
STD_EPS = 1e-6


def _fp(x):
    return x if x.dtype == torch.float64 else x.float()


def inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def feat(x: torch.Tensor) -> torch.Tensor:
    """f(x) = sign(x) * log1p(|x|) -- the [v4] scalar-feature normalisation."""
    return torch.sign(x) * torch.log1p(x.abs())


def column_stats(S: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
    """-> [B,H,n_k,6] over visible i.  S may hold -inf/finfo.min off vis; it is masked_fill'ed first (H13)."""
    S = _fp(S)
    vis = vis.bool()
    S0 = S.masked_fill(~vis, 0.0)                                       # never multiply -inf
    Sm = S.masked_fill(~vis, float("-inf"))
    cnt = vis.sum(-2).to(S.dtype)                                        # [B,H,n_k]  n_vis_j
    has = cnt > 0
    has2 = cnt > 1
    mx = Sm.amax(-2)
    mx = torch.where(has, mx, torch.zeros_like(mx))
    if S.shape[-2] >= 2:
        second = Sm.topk(2, dim=-2).values[..., 1, :]
        second = torch.where(has2, second, mx)                           # second := max if < 2 visible
    else:
        second = mx
    cnt_safe = torch.where(has, cnt, torch.ones_like(cnt))               # H11
    mean = S0.sum(-2) / cnt_safe
    var = (((S0 - mean.unsqueeze(-2)) ** 2) * vis.to(S.dtype)).sum(-2) / cnt_safe
    # H11/H12: sqrt has an infinite derivative at 0 and a column seen by ONE query has var = 0 exactly;
    # 0 * inf = NaN in the backward, so the sqrt runs on a safe argument and std is 0 there (subgradient 0).
    var_pos = var > 0
    std = torch.where(var_pos, torch.sqrt(torch.where(var_pos, var, torch.ones_like(var))), torch.zeros_like(var))
    gap = mx - second
    zsc = (mx - mean) / std.clamp_min(STD_EPS)
    out = torch.stack([mx, second, mean, std, gap, zsc], dim=-1)
    return torch.where(has.unsqueeze(-1), out, torch.zeros_like(out))


def row_summary(Atil: torch.Tensor, E: torch.Tensor, vis: torch.Tensor, cbar_i: torch.Tensor,
                Rtil: torch.Tensor) -> torch.Tensor:
    """-> [B,H,n_q,4] = (Rtil_i, cbar_i, max over visible j of Atil_ij, entropy of Atil[i,E_i.]/Rtil_i).
    Uses the HARD E (no logit gradient through the summary; round-2 R-12)."""
    mx = Atil.masked_fill(~vis.bool(), 0.0).amax(-1)
    R_safe = torch.where(Rtil > 0, Rtil, torch.ones_like(Rtil))          # H11
    q = (Atil * E.to(Atil.dtype)) / R_safe.unsqueeze(-1)
    ent = -(q * torch.log(q.clamp_min(1e-30))).sum(-1)
    ent = torch.where(Rtil > 0, ent, torch.zeros_like(ent))
    return torch.stack([Rtil, cbar_i, mx, ent], dim=-1)


def row_relation_stats(Atil: torch.Tensor, E: torch.Tensor, col_size: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """-> [B,H,n_q,8]: what a row's relation set looks like when its row programme runs (owner's proposal, 2026-09-21):
      (|E_i.|, max / max - second / mean / std of Atil on E_i.,  mean and max SIZE of the columns the row belongs to,
       the row's mean share p in them)
    col_size [B,H,n_k] (one size per column) or [B,H,n_q,n_k] (the size when THAT row triggered it).  One function for the
    dense form, the triggered form and decode_step.  Hard E, like row_summary."""
    A = _fp(Atil); Eb = E.bool(); Ef = Eb.to(A.dtype)
    n = Ef.sum(-1)
    top = A.masked_fill(~Eb, float("-inf")).topk(min(2, A.shape[-1]), dim=-1).values
    mx = torch.where(n > 0, top[..., 0], torch.zeros_like(n))
    second = torch.where(n > 1, top[..., -1], mx)
    mean = (A * Ef).sum(-1) / n.clamp_min(1.0)
    std = ((((A - mean.unsqueeze(-1)) ** 2) * Ef).sum(-1) / n.clamp_min(1.0) + 1e-12).sqrt()
    cs = _fp(col_size); cs = cs.unsqueeze(-2) if cs.dim() == A.dim() - 1 else cs
    cmean = (cs * Ef).sum(-1) / n.clamp_min(1.0); cmax = (cs * Ef).amax(-1)
    share = (_fp(p) * Ef).sum(-1) / n.clamp_min(1.0)
    return torch.stack([n, mx, mx - second, mean, std, cmean, cmax, share], dim=-1)


class _Head(nn.Module):
    def __init__(self, d_head: int, n_scalar: int, hidden: int, tau_min: float, init_tau: float,
                 per_head: int = 0):
        super().__init__()
        self.tau_min = tau_min
        self.per_head = per_head
        self.ln = nn.LayerNorm(d_head)                                    # [v4] on the vector input
        if per_head:
            # one MLP per Q-head, batched: weights [H, in, hidden] / [H, hidden, 1]
            din = d_head + n_scalar
            self.w1 = nn.Parameter(torch.empty(per_head, din, hidden)); self.b1 = nn.Parameter(torch.zeros(per_head, hidden))
            self.w2 = nn.Parameter(torch.empty(per_head, hidden, 1));   self.b2 = nn.Parameter(torch.zeros(per_head, 1))
            bound = 1.0 / math.sqrt(din)
            nn.init.uniform_(self.w1, -bound, bound); nn.init.uniform_(self.b1, -bound, bound)
            nn.init.normal_(self.w2, std=0.01)
            self.b2.data.fill_(inv_softplus(max(init_tau - tau_min, 1e-3)))
        else:
            self.fc1 = nn.Linear(d_head + n_scalar, hidden)
            self.fc2 = nn.Linear(hidden, 1)
            nn.init.normal_(self.fc2.weight, std=0.01)                    # output = bias +- O(0.01)
            self.fc2.bias.data.fill_(inv_softplus(max(init_tau - tau_min, 1e-3)))

    @property
    def last_bias(self) -> torch.Tensor:
        return self.b2 if self.per_head else self.fc2.bias

    def _mlp(self, x: torch.Tensor, head_offset: int = 0) -> torch.Tensor:
        if self.per_head:
            # `head_offset` is the global index of x's first Q-head: with head blocking x carries only a slice of the
            # heads, and these parameters are indexed in GLOBAL head space.  Without the slice the einsum either fails
            # on the head dimension or -- worse, where the widths happen to agree -- silently uses the wrong head's MLP.
            hs = slice(head_offset, head_offset + x.shape[1])
            h = F.gelu(torch.einsum("bhnd,hdk->bhnk", x, self.w1[hs]) + self.b1[hs][None, :, None, :])
            return (torch.einsum("bhnk,hko->bhno", h, self.w2[hs]) + self.b2[hs][None, :, None, :]).squeeze(-1)
        return self.fc2(F.gelu(self.fc1(x))).squeeze(-1)

    def set_init_tau(self, tau: float):
        self.last_bias.data.fill_(inv_softplus(max(tau - self.tau_min, 1e-3)))


class TauK(_Head):
    """tau_j = tau_min + softplus(MLP(LN(k_j), f(column_stats_j), f(log nu_prev_j))).  One per layer.
    zero_field_inputs=True is B3 (KeyOnlyTauNorm): stats and log nu zeroed AT THE INPUT, byte-identical
    otherwise.  no_nu=True is the E9 "no nu" arm."""

    def __init__(self, d_head: int, hidden: int = 64, tau_min: float = TAU_MIN, init_tau: float = 1.0,
                 zero_field_inputs: bool = False, no_nu: bool = False, per_head: int = 0):
        super().__init__(d_head, 6 + 1, hidden, tau_min, init_tau, per_head)
        self.zero_field_inputs = zero_field_inputs
        self.no_nu = no_nu

    def forward(self, k_rep: torch.Tensor, stats: torch.Tensor, nu_prev: torch.Tensor, head_offset: int = 0) -> torch.Tensor:
        """k_rep [B,H,n_k,d] (KV-head key expanded to Q-heads); stats [B,H,n_k,6]; nu_prev [B,H,n_k].
        `head_offset`: the global index of the first Q-head present, for the per-head (E9) parameters."""
        x_stats = feat(_fp(stats))
        x_nu = feat(torch.log(_fp(nu_prev).clamp_min(1e-30))).unsqueeze(-1)
        if self.zero_field_inputs:
            x_stats = torch.zeros_like(x_stats)
            x_nu = torch.zeros_like(x_nu)
        elif self.no_nu:
            x_nu = torch.zeros_like(x_nu)
        x = torch.cat([self.ln(_fp(k_rep)), x_stats, x_nu], dim=-1)
        return self.tau_min + F.softplus(self._mlp(x, head_offset))

    @torch.no_grad()
    def calibrate(self, S: torch.Tensor, vis: torch.Tensor, stds=None):
        """Sec. 5.2 [v4]: bias := softplus^-1(median_j (1/std_j) - tau_min), a MEDIAN over columns with
        >= 2 visible entries.  `stds` may be passed pre-collected over several sequences."""
        if self.per_head and (stds is None or isinstance(stds, (list, tuple))):
            # one target per Q-head (2026-09-20 recipe): each head's columns have their own score spread, and one pooled
            # median starts every head but the median one at the wrong sharpness.  `stds`: a list of 1-D tensors, per head.
            if stds is None:
                st = column_stats(S, vis)[..., 3]; ok = vis.bool().expand_as(S).sum(-2) > 1
                stds = [st[:, h][ok[:, h]] for h in range(st.shape[1])]
            assert len(stds) == self.per_head, (len(stds), self.per_head)
            targets = []
            for h, sd in enumerate(stds):
                t = (1.0 / sd.clamp_min(STD_EPS)).median().item() if sd.numel() else 1.0
                self.b2.data[h].fill_(inv_softplus(max(t - self.tau_min, 1e-3))); targets.append(t)
            return targets
        if stds is None:
            st = column_stats(S, vis)[..., 3]
            stds = st[vis.bool().expand_as(S).sum(-2) > 1]
        if stds.numel() == 0:
            return float(1.0)
        target = (1.0 / stds.clamp_min(STD_EPS)).median().item()
        self.set_init_tau(target)
        return target


def trigger_stats(vals: torch.Tensor, valid: torch.Tensor, count: torch.Tensor, s_trig: torch.Tensor) -> torch.Tensor:
    """-> [..., 6]: what a column looks like at the moment a row triggers it.  vals / valid [..., K]: the column's member
    list (its top-K_ret relation scores, the triggering row's own among them if it made the list); count [...]: the TRUE
    size of the relation set so far (the list is truncated, the size is not); s_trig [...]: the triggering row's score.
      (size, max, max - second, mean, std, s_trig - max)
    One function for the teacher-forced form and for decode_step, so the two cannot disagree on a feature."""
    v = _fp(vals); m = valid.bool()
    n = m.sum(-1).to(v.dtype)
    neg = v.masked_fill(~m, float("-inf"))
    vmax = torch.where(n > 0, neg.amax(-1), torch.zeros_like(n))
    if v.shape[-1] > 1:
        second = torch.where(n > 1, neg.topk(2, dim=-1).values[..., 1], vmax)
    else:
        second = vmax
    v0 = v.masked_fill(~m, 0.0)
    mean = v0.sum(-1) / n.clamp_min(1.0)
    var = (((v0 - mean.unsqueeze(-1)) ** 2) * m.to(v.dtype)).sum(-1) / n.clamp_min(1.0)
    std = (var + 1e-8).sqrt()
    return torch.stack([_fp(count), vmax, vmax - second, mean, std, _fp(s_trig) - vmax], dim=-1)


class TauKTrigger(_Head):
    """tau for ONE solve of a column, predicted when a row triggers it (owner's decision, 2026-09-21):
        tau_tj = tau_min + softplus(MLP(LN(k_j), f(trigger_stats)))
    TauK's tau_j is sealed at prefill from the PROMPT rows' column statistics: it cannot know how many members the
    column has when a later row arrives, nor where that row stands among them -- and that gap decides whether the row is
    starved (score more than ~1/tau below the members: an exact zero) or over-paid (top score: most of a quota the
    earlier rows also paid into; relation mass 1-4 at the answer rows in the 2026-09-20 pilots).  Everything it reads
    exists at trigger time, so nothing leaks.  The price is prop:prefix's hypothesis (psi_j monotone at SEALED tau_j):
    exclusion is no longer permanent inside the programme -- each row's EMITTED output still is (no row is revised)."""

    def __init__(self, d_head: int, hidden: int = 64, tau_min: float = TAU_MIN, init_tau: float = 1.0, per_head: int = 0):
        super().__init__(d_head, 6, hidden, tau_min, init_tau, per_head)

    def forward(self, k_vec: torch.Tensor, stats: torch.Tensor, head_offset: int = 0) -> torch.Tensor:
        """k_vec [B,H,N,d] the triggered columns' keys (expanded to Q-heads); stats [B,H,N,6] -> tau [B,H,N]."""
        x = torch.cat([self.ln(_fp(k_vec)), feat(_fp(stats))], dim=-1)
        return self.tau_min + F.softplus(self._mlp(x, head_offset))


class TauQ(_Head):
    """tau_i = tau_min + softplus(MLP(LN(q_i), f(row_summary_i))).  Init tau_i = 1: the identity on step 1's row."""

    def __init__(self, d_head: int, hidden: int = 64, tau_min: float = TAU_MIN, init_tau: float = 1.0,
                 per_head: int = 0, sized: bool = False):
        # sized: the summary carries row_relation_stats too (4 + 8 scalars) -- a tau_i that can see how large the row's
        # relation set is, how peaked it is and how crowded its columns are.  It shapes the RELATION part of the row
        # (s_i = cbar_i / tau_i); the off-relation tail is the unit cap's business, not tau_i's.
        super().__init__(d_head, 12 if sized else 4, hidden, tau_min, init_tau, per_head)
        self.sized = bool(sized)

    def summary(self, Atil, E, vis, cbar_i, Rtil, col_size=None, p=None) -> torch.Tensor:
        s = row_summary(Atil, E, vis, cbar_i, Rtil)
        return torch.cat([s, row_relation_stats(Atil, E, col_size, p)], dim=-1) if self.sized else s

    def forward(self, q: torch.Tensor, summary: torch.Tensor, head_offset: int = 0) -> torch.Tensor:
        x = torch.cat([self.ln(_fp(q)), feat(_fp(summary))], dim=-1)
        return self.tau_min + F.softplus(self._mlp(x, head_offset))
