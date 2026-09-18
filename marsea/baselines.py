"""
Sec. 8: one interface, six implementations.  Every arm is a Normalizer called at the softmax line;
matched only if every arm shares the same backbone, patched layers, LoRA and schedule.

  B0 SoftmaxNorm          A = row_softmax(S)                        (the backbone unchanged; T0)
  B1 SoftmaxOneNorm       A = exp(s) / (1 + sum_visible exp(s))     (Miller 2023; rows < 1, no zeros)
  B2 MESHNorm             learned marginals on BOTH sides + entropic Sinkhorn on a MESH-descended cost
                          (Zhang et al. ICML 2023, Sec. 3 / App. A); eps = 1; straight-through descent;
                          ENDS ON THE ROW SCALING; column residual logged (infeasible under causal masks)
  B3 KeyOnlyTauNorm       MarSea with TauK's column_stats and nu zeroed at the input (byte-identical otherwise)
  B4 RowEntmaxNorm        A[i,:] = entmax_alpha(S[i,:]) over visible j, alpha in {1.5, 2}, no temperature
  B5 MatchedSparsityNorm  two-pass, evaluation-time only: B0's A_sm sparsified to MarSea's recorded
                          relation support size n_i within MarSea's recorded E_i.
"""
from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn

from .primitives import entmax_masked, _fp
from .relation import repeat_kv
from .normalizer import MarSeaNormalizer, Diagnostics, State, row_softmax, broadcast_vis
from .heads import TAU_MIN


class SoftmaxNorm(nn.Module):                                          # B0
    def normalize(self, S, vis, K_kv=None, Q=None, state=None, **kw):
        vis = broadcast_vis(vis, S)
        with torch.autocast(device_type=S.device.type, enabled=False):
            A = row_softmax(S, vis)
        return A.to(S.dtype), Diagnostics(A_sm=A, A=A)

    forward = normalize


class SoftmaxOneNorm(nn.Module):                                       # B1 (Miller 2023, NOT Xiao et al.)
    def normalize(self, S, vis, K_kv=None, Q=None, state=None, **kw):
        vis = broadcast_vis(vis, S)
        with torch.autocast(device_type=S.device.type, enabled=False):
            S32 = _fp(S).masked_fill(~vis, float("-inf"))
            zero = torch.zeros_like(S32[..., :1])                          # the "+1" at score 0 AFTER scaling
            lse = torch.logsumexp(torch.cat([S32, zero], dim=-1), dim=-1, keepdim=True)   # log(1 + sum_visible exp)
            A = torch.exp(S32 - lse)                                       # exact 0 off vis; rows sum to < 1
        return A.to(S.dtype), Diagnostics(A_sm=A, A=A, extra={"row_sum": A.sum(-1)})

    forward = normalize


# ------------------------------------------------------------------------------- B2 MESH
def sinkhorn_log(C: torch.Tensor, log_a: torch.Tensor, log_b: torch.Tensor, eps: float, iters: int,
                 end_on: str = "row") -> torch.Tensor:
    """Entropic OT in the log domain.  C [B,H,n_q,n_k] (+inf where invisible), log_a [B,H,n_k] (keys;
    -inf on unseen keys), log_b [B,H,n_q] (queries; -inf on pad rows).  Returns the plan P with the
    LAST scaling on the side named by `end_on` ("row": rows carry b_i exactly, spec Sec. 8 [v4])."""
    ninf = float("-inf")
    key_ok = torch.isfinite(log_a)
    row_ok = torch.isfinite(log_b)
    negC = (-C) / eps                                                       # -inf off vis
    dead = (~row_ok).unsqueeze(-1) | (~key_ok).unsqueeze(-2)
    negC_safe = negC.masked_fill(dead, 0.0)                                # an all -inf reduction has a NaN backward
    la = log_a.masked_fill(~key_ok, 0.0)
    lb = log_b.masked_fill(~row_ok, 0.0)
    f = torch.zeros_like(lb)                                                # row dual
    gd = torch.zeros_like(la)                                               # column dual
    for _ in range(iters):
        if end_on == "row":
            gd = la - torch.logsumexp(negC_safe + f.unsqueeze(-1), dim=-2)  # column step
            gd = gd.masked_fill(~key_ok, 0.0)
            f = lb - torch.logsumexp(negC_safe + gd.unsqueeze(-2), dim=-1)  # row step LAST
            f = f.masked_fill(~row_ok, 0.0)
        else:
            f = lb - torch.logsumexp(negC_safe + gd.unsqueeze(-2), dim=-1)
            f = f.masked_fill(~row_ok, 0.0)
            gd = la - torch.logsumexp(negC_safe + f.unsqueeze(-1), dim=-2)
            gd = gd.masked_fill(~key_ok, 0.0)
    logP = negC_safe + f.unsqueeze(-1) + gd.unsqueeze(-2)
    return torch.exp(logP.masked_fill(dead | ~torch.isfinite(negC), ninf))


class MESHNorm(nn.Module):                                             # B2
    """Pinned to Zhang et al. 2023 ("Unlocking Slot Attention by Changing Optimal Transport Costs").
      a = m * softmax_{j visible}(h_a(LN(K_kv))[j])      (repeated across the GQA group)
      b = m * softmax_{i real}   (h_b(LN(Q))[i])          m = n_real_q  (the row-softmax total)
      C = -S on vis, +inf off vis; eps = 1.0
      MESH step (Eq. 11): C'(0) = C + N(0, 1e-6); 4 x { C' -= lam * grad_C' H(sinkhorn(C', a, b, 5 iters)) / ||grad|| }
          -- descent MINIMISES plan entropy (sharpens); NOT differentiated through: straight-through to C
      A = sinkhorn(C'(4), a, b, eps, 20 log-domain iters), ENDING ON THE ROW SCALING; column residual logged.
    """

    def __init__(self, d_head: int, hidden: int = 64, eps: float = 1.0, lam: float = 1.0, mesh_steps: int = 4,
                 inner_iters: int = 5, outer_iters: int = 20, noise: float = 1e-6):
        super().__init__()
        self.h_a = nn.Sequential(nn.LayerNorm(d_head), nn.Linear(d_head, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.h_b = nn.Sequential(nn.LayerNorm(d_head), nn.Linear(d_head, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.h_a[-1].bias); nn.init.zeros_(self.h_b[-1].bias)     # uniform marginals at init
        self.eps, self.lam, self.mesh_steps = eps, lam, mesh_steps
        self.inner_iters, self.outer_iters, self.noise = inner_iters, outer_iters, noise

    def normalize(self, S, vis, K_kv, Q, state=None, **kw):
        vis = broadcast_vis(vis, S)
        B, H, n_q, n_k = S.shape
        g = H // K_kv.shape[1]
        with torch.autocast(device_type=S.device.type, enabled=False):
            S32 = _fp(S).masked_fill(~vis, float("-inf"))
            key_vis = vis.any(-2)                                           # [B,H,n_k] seen by >= 1 real query
            row_vis = vis.any(-1)                                           # [B,H,n_q] real rows
            m = row_vis.sum(-1).to(S32.dtype).clamp_min(1.0)                # [B,H]  n_real_q
            la = repeat_kv(self.h_a(_fp(K_kv)), g).squeeze(-1)              # [B,H,n_k]
            lb = self.h_b(_fp(Q)).squeeze(-1)                               # [B,H,n_q]
            log_a = torch.log(m).unsqueeze(-1) + torch.log_softmax(la.masked_fill(~key_vis, float("-inf")), -1)
            log_b = torch.log(m).unsqueeze(-1) + torch.log_softmax(lb.masked_fill(~row_vis, float("-inf")), -1)
            log_a = log_a.masked_fill(~key_vis, float("-inf"))
            log_b = log_b.masked_fill(~row_vis, float("-inf"))
            C = (-S32).masked_fill(~vis, float("inf"))
            # ---- MESH step: not differentiated through (straight-through to C)
            with torch.no_grad():
                Cp = C + (torch.randn_like(C) * self.noise).masked_fill(~vis, 0.0)
                for _ in range(self.mesh_steps):
                    with torch.enable_grad():
                        Cv = Cp.detach().requires_grad_(True)
                        P = sinkhorn_log(Cv, log_a.detach(), log_b.detach(), self.eps, self.inner_iters, "row")
                        Hp = -(P * torch.log(P.clamp_min(1e-30))).sum((-1, -2))
                        (grad,) = torch.autograd.grad(Hp.sum(), Cv)
                    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0).masked_fill(~vis, 0.0)
                    gn = grad.flatten(-2).norm(dim=-1).clamp_min(1e-12)[..., None, None]
                    Cp = Cp - self.lam * grad / gn                           # descent: sharpens
            C_st = C + (Cp - C).detach().masked_fill(~vis, 0.0)             # forward = Cp; backward straight to C (S)
            A = sinkhorn_log(C_st, log_a, log_b, self.eps, self.outer_iters, "row")
            a = log_a.exp().masked_fill(~key_vis, 0.0)
            col_resid = (A.sum(-2) - a).abs().masked_fill(~key_vis, 0.0).amax(-1)       # [B,H] logged per layer
        diag = Diagnostics(A_sm=None, A=A, extra={"row_sum": A.sum(-1), "col_resid": col_resid,
                                                  "a": a, "b": log_b.exp().masked_fill(~row_vis, 0.0)})
        return A.to(S.dtype), diag

    forward = normalize

    def new_module_params(self):
        weights, others = [], []
        for name, p in self.named_parameters():
            (weights if p.dim() >= 2 else others).append(p)
        return weights, others


class KeyOnlyTauNorm(MarSeaNormalizer):                                # B3
    def __init__(self, d_head: int, r: int = 16, alpha: float = 2.0, tau_min: float = TAU_MIN, **kw):
        kw.pop("key_only_tau", None)
        super().__init__(d_head, r, alpha, tau_min, key_only_tau=True, **kw)


class RowEntmaxNorm(nn.Module):                                        # B4
    def __init__(self, alpha: float = 2.0):
        super().__init__()
        self.alpha = alpha

    def normalize(self, S, vis, K_kv=None, Q=None, state=None, **kw):
        vis = broadcast_vis(vis, S)
        with torch.autocast(device_type=S.device.type, enabled=False):
            S32 = _fp(S).masked_fill(~vis, 0.0)                              # H13 before any arithmetic
            A = entmax_masked(S32, vis, self.alpha)
        return A.to(S.dtype), Diagnostics(A=A, extra={"supp": (A > 0).sum(-1)})

    forward = normalize


class MatchedSparsityNorm(nn.Module):                                  # B5 (evaluation-time only; D-29)
    """trace = (E [B,H,n_q,n_k] bool, n [B,H,n_q] long) recorded from the paired MarSea pass on the SAME
    example (same order, same seed).  Runs on the trained B0 weights:
      keep = top-n_i entries of A_sm within E_i. (ties: stable, lower index first);
      A[E_i.] = A_sm * keep * (sum_{E_i.} A_sm / sum_keep A_sm);  A[~E_i.] = A_sm;  n_i = 0 => A[E_i.] = 0.
    Teacher-forced only: its trace is per gold row, so no task-level number exists for it."""

    def __init__(self):
        super().__init__()
        self.trace = None                      # set per example by the evaluator

    def normalize(self, S, vis, K_kv=None, Q=None, state=None, *, trace=None, **kw):
        trace = trace if trace is not None else self.trace
        assert trace is not None, "B5 needs the MarSea trace (E, n) for this example"
        E, n = trace
        vis = broadcast_vis(vis, S)
        assert E.shape == S.shape, f"B5 trace shape {tuple(E.shape)} != scores {tuple(S.shape)}"
        with torch.autocast(device_type=S.device.type, enabled=False):
            A_sm = row_softmax(S, vis)
            E = E.bool() & vis
            n_k = S.shape[-1]
            rel = A_sm.masked_fill(~E, -1.0)
            order = torch.argsort(rel, dim=-1, descending=True, stable=True)
            rank = torch.empty_like(order)
            rank.scatter_(-1, order, torch.arange(n_k, device=S.device).expand_as(order))
            keep = E & (rank < n.to(rank.dtype).unsqueeze(-1))
            rel_mass = (A_sm * E).sum(-1, keepdim=True)
            kept_mass = (A_sm * keep).sum(-1, keepdim=True)
            km_safe = torch.where(kept_mass > 0, kept_mass, torch.ones_like(kept_mass))
            scale = torch.where(kept_mass > 0, rel_mass / km_safe, torch.zeros_like(kept_mass))
            A = torch.where(E, A_sm * keep * scale, A_sm)
        return A.to(S.dtype), Diagnostics(A_sm=A_sm, A=A, E=E, extra={"supp_rel": keep.sum(-1)})

    forward = normalize


ARMS = ("marsea", "B0", "B1", "B2", "B3", "B4", "B5")


def make_normalizer(arm: str, d_head: int, **kw) -> nn.Module:
    """Factory for the seven arms plus the E9 variants (passed as MarSeaNormalizer kwargs)."""
    arm_l = arm.lower()
    if arm_l in ("marsea", "m"):
        return MarSeaNormalizer(d_head, **kw)
    if arm_l == "b0":
        return SoftmaxNorm()
    if arm_l == "b1":
        return SoftmaxOneNorm()
    if arm_l == "b2":
        return MESHNorm(d_head, **{k: v for k, v in kw.items() if k in ("hidden", "eps", "lam", "mesh_steps", "inner_iters", "outer_iters")})
    if arm_l == "b3":
        return KeyOnlyTauNorm(d_head, **kw)
    if arm_l == "b4":
        return RowEntmaxNorm(alpha=kw.get("alpha", 2.0))
    if arm_l == "b5":
        return MatchedSparsityNorm()
    raise ValueError(f"unknown arm {arm}")
