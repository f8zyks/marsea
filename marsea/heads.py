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
    def calibrate(self, S: torch.Tensor, vis: torch.Tensor, stds: torch.Tensor | None = None) -> float:
        """Sec. 5.2 [v4]: bias := softplus^-1(median_j (1/std_j) - tau_min), a MEDIAN over columns with
        >= 2 visible entries.  `stds` may be passed pre-collected over several sequences."""
        if stds is None:
            st = column_stats(S, vis)[..., 3]
            stds = st[vis.bool().expand_as(S).sum(-2) > 1]
        if stds.numel() == 0:
            return float(1.0)
        target = (1.0 / stds.clamp_min(STD_EPS)).median().item()
        self.set_init_tau(target)
        return target


class TauQ(_Head):
    """tau_i = tau_min + softplus(MLP(LN(q_i), f(row_summary_i))).  Init tau_i = 1: the identity on step 1's row."""

    def __init__(self, d_head: int, hidden: int = 64, tau_min: float = TAU_MIN, init_tau: float = 1.0,
                 per_head: int = 0):
        super().__init__(d_head, 4, hidden, tau_min, init_tau, per_head)

    def forward(self, q: torch.Tensor, summary: torch.Tensor, head_offset: int = 0) -> torch.Tensor:
        x = torch.cat([self.ln(_fp(q)), feat(_fp(summary))], dim=-1)
        return self.tau_min + F.softplus(self._mlp(x, head_offset))
