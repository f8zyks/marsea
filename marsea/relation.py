"""
Sec. 4: the relation E -- ONE boolean object whose slices are E_.j and E_i. (INV-5 rule of
construction).  Pairwise by requirement (prop:prefix needs arrival-sealing), never a function of
the query set as a whole.

  logits_ij = <U_phi(k_j), V_phi(q_i)> / sqrt(r) + b0        (spec Sec. 4.1; the paper's e_ij with a
                                                              constant coordinate appended)
  E = (logits > 0) & vis
  g = E.float() + (sigmoid(logits/T_st) * vis - detach(.))   straight-through (Sec. 4.3): hard forward,
                                                              sigmoid backward, no gradient off vis
"""
from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn

T_ST = 1.0                                  # spec Sec. 4.3


def repeat_kv(x: torch.Tensor, g: int) -> torch.Tensor:
    """[B, H_kv, n, ...] -> [B, H_kv*g, n, ...]; each KV head repeated g times consecutively (HF order).

    This COPIES for g > 1 (merging a stride-0 dimension into its neighbour is not viewable, so `reshape` falls back
    to `contiguous`) -- HF's own repeat_kv does the same.  At 8K one copy of K is 50 MB, so it is not a term that
    matters, but call it once and slice, rather than once per chunk or per row block."""
    if g == 1:
        return x
    B, Hkv = x.shape[:2]
    rest = x.shape[2:]
    return x.unsqueeze(2).expand(B, Hkv, g, *rest).reshape(B, Hkv * g, *rest)


SINK_LOGIT = -1e4                           # D-31: the sink pairs' logits (E False; sigmoid' == 0, so no ST gradient)


def sink_pair_mask(n_q: int, n_k: int, device, key_offset: int = 0, n_k_total: Optional[int] = None) -> torch.Tensor:
    """D-31 (spec Sec. 4.1, 2026-09-09): position 0 -- the attention sink of a BOS-less decoder -- is excluded from every
    relation: E[:, 0] = E[0, :] = False.  Returns [n_q, n_k] bool, True on the excluded pairs.  `key_offset` is the
    global index of the first key of a chunk; `n_k_total` the full key count (query row i sits at global position
    i + n_k_total - n_q)."""
    n_k_total = n_k if n_k_total is None else n_k_total
    m = torch.zeros(n_q, n_k, dtype=torch.bool, device=device)
    j0 = -key_offset
    if 0 <= j0 < n_k:
        m[:, j0] = True                                            # the sink COLUMN (key at position 0)
    i0 = n_q - n_k_total
    if 0 <= i0 < n_q:
        m[i0, :] = True                                            # the sink ROW (query at position 0)
    return m


def candidate_logits(raw: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
    """the visible pairs that can enter a relation: vis minus the D-31 sink pairs.  Returns the 1-D logit values.
    Both b0 calibrators take their candidates from here, so they cannot disagree about the quantity rho_0 fixes."""
    visb = vis.bool().expand_as(raw) & ~sink_pair_mask(raw.shape[-2], raw.shape[-1], raw.device)
    return raw[visb]


def bisect_b0(vals: torch.Tensor, rho0: float = 0.05, iters: int = 30) -> float:
    """the b0 that puts a rho0 fraction of the candidate logits above 0 (spec Sec. 4.2 / training procedure Sec. 6.3)."""
    if vals.numel() == 0:
        return 0.0
    lo = -(vals.abs().max().item() + 1.0); hi = -lo
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if (vals + mid > 0).float().mean().item() > rho0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


class RelationHead(nn.Module):
    """One per patched layer, shared across heads (per-head is an E9 ablation).

    U_phi reads the KV-head key (post-RoPE, as the patch point hands it over), V_phi reads the
    Q-head query; the product is expanded to Q-heads with repeat_kv.  Weights ~ N(0, var 1/d_head).
    b0 is a trainable scalar per layer, calibrated once at the start of Phase B (Sec. 4.2).

    key_only=True is the E9 "relation from u_phi(k_j) alone" arm: logits_ij = <U k_j, 1>/sqrt(r) + b0,
    the same for every query i (still pairwise-sealed, trivially).
    """

    def __init__(self, d_head: int, r: int = 16, b0: float = 0.0, key_only: bool = False,
                 per_head: int = 0, per_head_b0: bool = False):
        super().__init__()
        self.r = r
        self.key_only = key_only
        self.per_head = per_head                       # 0 = shared; H = one (U,V) per Q-head (E9)
        # per_head_b0 (2026-09-20 recipe): one threshold per Q-head.  With ONE b0 per layer the rho_0 calibration is met
        # on average and by no head: at calibration layer 24's heads 0-5 (KV head 0) sat at 7-13 % and heads 6-11 at
        # 0.06-0.6 %, and training drove them to 50-89 % against 0.01 %.  Needs per_head (U_h reads the keys of head h's
        # KV head, V_h its queries).  The E9 `per_head` arm of 2026-09 keeps its scalar b0 (its checkpoints load).
        self.per_head_b0 = bool(per_head_b0)
        assert not (self.per_head_b0 and not per_head), "per_head_b0 needs per_head = H"
        if per_head:
            self.U = nn.Parameter(torch.empty(per_head, d_head, r))
            self.V = nn.Parameter(torch.empty(per_head, d_head, r))
            nn.init.normal_(self.U, std=1.0 / math.sqrt(d_head))
            nn.init.normal_(self.V, std=1.0 / math.sqrt(d_head))
        else:
            self.U = nn.Linear(d_head, r, bias=False)
            self.V = nn.Linear(d_head, r, bias=False)
            nn.init.normal_(self.U.weight, std=1.0 / math.sqrt(d_head))   # variance 1/d_head
            nn.init.normal_(self.V.weight, std=1.0 / math.sqrt(d_head))
        self.b0 = nn.Parameter(torch.full((per_head,), float(b0)) if self.per_head_b0 else torch.tensor(float(b0)))

    def b0_view(self, H: int, head_offset: int = 0) -> torch.Tensor:
        """b0 broadcastable against [B, H, n_q, n_k] logits: the scalar, or the H heads from `head_offset` on."""
        return self.b0[head_offset:head_offset + H].view(1, H, 1, 1) if self.per_head_b0 else self.b0

    def b0_of(self, h: int) -> torch.Tensor:
        """the threshold of GLOBAL Q-head h (a 0-d tensor either way)."""
        return self.b0[h] if self.per_head_b0 else self.b0

    def raw(self, K_kv: torch.Tensor, Q: torch.Tensor, head_offset: int = 0) -> torch.Tensor:
        """[B,H,n_q,n_k] logits WITHOUT b0.  K_kv [B,H_kv,n_k,d] post-RoPE; Q [B,H,n_q,d].
        `head_offset` is the global index of Q's first head: with head blocking Q carries a slice of the heads while
        the per-head (E9) U/V are indexed in global head space."""
        H = Q.shape[1]
        g = H // K_kv.shape[1]
        wdt = self.U.dtype if self.per_head else self.U.weight.dtype
        Kf = K_kv.to(wdt)
        Qf = Q.to(wdt)
        if self.per_head:
            hs = slice(head_offset, head_offset + H)
            k_rep = repeat_kv(Kf, g)                                      # [B,H,n_k,d]
            u = torch.einsum("bhjd,hdr->bhjr", k_rep, self.U[hs])
            v = torch.einsum("bhid,hdr->bhir", Qf, self.V[hs])
        else:
            u = repeat_kv(self.U(Kf), g)                                  # [B,H,n_k,r]
            v = self.V(Qf)                                                # [B,H,n_q,r]
        if self.key_only:
            # a per-key score broadcast over queries: <u_j, 1>/sqrt(r)
            uj = u.sum(-1) / math.sqrt(self.r)                            # [B,H,n_k]
            return uj.unsqueeze(-2).expand(-1, -1, Q.shape[-2], -1)
        return torch.einsum("bhir,bhjr->bhij", v, u) / math.sqrt(self.r)

    def forward(self, K_kv: torch.Tensor, Q: torch.Tensor, key_offset: int = 0, n_k_total: Optional[int] = None,
                head_offset: int = 0) -> torch.Tensor:
        """logits with b0, and the D-31 sink mask applied (position 0 is never in a relation)."""
        logits = self.raw(K_kv, Q, head_offset) + self.b0_view(Q.shape[1], head_offset)
        m = sink_pair_mask(Q.shape[-2], K_kv.shape[-2], logits.device, key_offset, n_k_total)
        return logits.masked_fill(m, SINK_LOGIT)

    @torch.no_grad()
    def calibrate_b0(self, K_kv: torch.Tensor, Q: torch.Tensor, vis: torch.Tensor,
                     rho0: float = 0.05, iters: int = 20, raw: Optional[torch.Tensor] = None) -> float:
        """Sec. 4.2 / training procedure Sec. 6.3: bisection on b0 so that
        mean(E & vis) / mean(vis) == rho0 on this batch (hard E, all Q-heads pooled, per layer).
        Accepts a precomputed `raw` (e.g. concatenated over several sequences) to calibrate on a
        multi-sequence batch without recomputing.  Returns the new b0."""
        if raw is None:
            raw = self.raw(K_kv, Q)
        if self.per_head_b0:
            assert raw.dim() == 4, "per-head calibration needs the [B, H, n_q, n_k] logits (or calibrate_b0_per_head)"
            return self.calibrate_b0_per_head([candidate_logits(raw[:, h:h + 1], vis) for h in range(raw.shape[1])], rho0, iters)
        vals = candidate_logits(raw, vis) if raw.dim() > 1 else raw       # pre-filtered 1-D values are accepted as-is
        if vals.numel() == 0:
            return float(self.b0)
        self.b0.fill_(bisect_b0(vals, rho0, iters))
        return float(self.b0)

    @torch.no_grad()
    def calibrate_b0_per_head(self, vals_per_head, rho0: float = 0.05, iters: int = 30) -> list:
        """one bisection per Q-head, each over that head's own candidate logits (1-D, WITHOUT b0): every head sits at
        rho0, not the layer on average.  A head with no candidates keeps its b0."""
        assert self.per_head_b0 and len(vals_per_head) == self.b0.numel()
        for h, vals in enumerate(vals_per_head):
            if vals.numel():
                self.b0[h] = bisect_b0(vals, rho0, iters)
        return [float(x) for x in self.b0]

    @torch.no_grad()
    def coverage(self, logits: torch.Tensor, vis: torch.Tensor) -> float:
        visb = vis.bool().expand_as(logits)
        E = (logits > 0) & visb
        return (E.sum() / visb.sum().clamp_min(1)).item()


def straight_through(E: torch.Tensor, logits: torch.Tensor, vis: torch.Tensor, T_st: float = T_ST) -> torch.Tensor:
    """g: forward == E exactly (so every INV holds); backward d/dlogit = sigmoid'(logit/T)/T on vis.
    T_st > 1 keeps the backward alive away from the threshold (at logit -6: 0.0025 at T = 1, 0.037 at T = 4); the trainer
    anneals MarSeaNormalizer.st_temperature to 1.  The forward never depends on it."""
    soft = torch.sigmoid(logits / T_st) * vis.to(logits.dtype)            # no gradient off vis
    return E.to(logits.dtype) + (soft - soft.detach())


def hard_concrete_gate(E: torch.Tensor, logits: torch.Tensor, vis: torch.Tensor, temperature: float,
                       training: bool) -> torch.Tensor:
    """E9 alternative to ST (spec Sec. 4.3): Gumbel-sigmoid, hard forward, soft backward, temperature
    annealed 1.0 -> 0.2 by the caller.  The forward set is still HARD (spec Sec. 15)."""
    if training:
        u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
        noise = torch.log(u) - torch.log1p(-u)
    else:
        noise = torch.zeros_like(logits)
    soft = torch.sigmoid((logits + noise) / temperature) * vis.to(logits.dtype)
    hard = ((logits + noise) > 0) & vis.bool()
    return hard.to(logits.dtype) + (soft - soft.detach()), hard
