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
                 end_on: str = "row", return_duals: bool = False):
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
    P = torch.exp(logP.masked_fill(dead | ~torch.isfinite(negC), ninf))
    return (P, f, gd) if return_duals else P


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
                 inner_iters: int = 5, outer_iters: int = 20, noise: float = 1e-6, causal: bool = False, causal_iters: int = 3):
        super().__init__()
        # causal (2026-09-23): the prompt is ONE Sinkhorn block (= the generation prefill); every answer row t gets its own
        # Sinkhorn over rows <= t, warm-started from row t - 1's duals (causal_iters iterations, no MESH sharpening steps
        # on the per-row solves), and only row t is emitted -- so teacher forcing IS decoding, and no row sees a later
        # one.  The gradient of a per-row solve flows through row t's own scaling (the inherited duals are constants);
        # the prompt block's gradient is the stock one.  Stock MESH normalises the whole teacher-forced matrix at once:
        # a row's column scaling then depends on LATER rows (the gold answer), which decoding never has (2026-09-20).
        self.causal = bool(causal); self.causal_iters = int(causal_iters)
        self.h_a = nn.Sequential(nn.LayerNorm(d_head), nn.Linear(d_head, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.h_b = nn.Sequential(nn.LayerNorm(d_head), nn.Linear(d_head, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.h_a[-1].bias); nn.init.zeros_(self.h_b[-1].bias)     # uniform marginals at init
        self.eps, self.lam, self.mesh_steps = eps, lam, mesh_steps
        self.inner_iters, self.outer_iters, self.noise = inner_iters, outer_iters, noise

    def normalize(self, S, vis, K_kv, Q, state=None, n_prefill=None, **kw):
        vis = broadcast_vis(vis, S)
        B, H, n_q, n_k = S.shape
        if self.causal and n_prefill is not None and 0 < int(n_prefill) < n_q and n_q == n_k:
            return self._normalize_causal(S, vis, K_kv, Q, int(n_prefill))
        A, diag = self._block(S, vis, K_kv, Q)
        return A, diag

    def _heads(self, K_kv, Q):
        """the two marginal heads on every key / row, once per sequence: la [B,H,n_k], lb [B,H,n_q]."""
        g = Q.shape[1] // K_kv.shape[1]
        return repeat_kv(self.h_a(_fp(K_kv)), g).squeeze(-1), self.h_b(_fp(Q)).squeeze(-1)

    def _marginals(self, la, lb, key_vis, row_vis):
        """log a (keys) and log b (rows): m x softmax of the two heads over what is visible, m = the number of real rows."""
        m = row_vis.sum(-1).to(torch.float32).clamp_min(1.0)                 # [B,H]
        log_a = (torch.log(m).unsqueeze(-1) + torch.log_softmax(la.masked_fill(~key_vis, float("-inf")), -1)).masked_fill(~key_vis, float("-inf"))
        log_b = (torch.log(m).unsqueeze(-1) + torch.log_softmax(lb.masked_fill(~row_vis, float("-inf")), -1)).masked_fill(~row_vis, float("-inf"))
        return log_a, log_b

    def _normalize_causal(self, S, vis, K_kv, Q, n_p: int):
        B, H, n_q, n_k = S.shape
        A_p, d_p = self._block(S[:, :, :n_p, :n_p], vis[:, :, :n_p, :n_p], K_kv[:, :, :n_p], Q[:, :, :n_p])
        with torch.autocast(device_type=S.device.type, enabled=False):
            S32 = _fp(S).masked_fill(~vis, float("-inf"))
            f, gd = d_p.extra["dual_f"], d_p.extra["dual_g"]                  # [B,H,n_p], [B,H,n_p]: the prompt block's duals
            la, lb = self._heads(K_kv, Q)
            rows = []
            for t in range(n_p, n_q):
                A_t, f, gd = self._row_solve(S32[:, :, :t + 1, :t + 1], vis[:, :, :t + 1, :t + 1], la[..., :t + 1], lb[..., :t + 1], f, gd)
                rows.append(torch.nn.functional.pad(A_t, (0, n_k - t - 1)))
            A_a = torch.cat(rows, -2)                                          # [B,H,m,n_k]
            pad = torch.zeros(B, H, n_p, n_q - n_p, dtype=A_p.dtype, device=S.device)
            A = torch.cat([torch.cat([_fp(A_p), pad], -1), A_a], -2)
        diag = Diagnostics(A_sm=None, A=A, extra={"row_sum": A.sum(-1), "col_resid": d_p.extra["col_resid"], "causal": True,
                                                  "dual_f": f.detach(), "dual_g": gd.detach()})
        return A.to(S.dtype), diag

    def _row_solve(self, S32, vis, la, lb_all, f_prev, gd_prev):
        """row t = the last row of S32 [B,H,t+1,t+1]: Sinkhorn over rows <= t warm-started from (f_prev, gd_prev) [B,H,t],
        causal_iters iterations, no MESH steps; la / lb_all the marginal heads on keys / rows <= t.  Returns
        (A_t [B,H,1,t+1], f [B,H,t+1], gd [B,H,t+1]).  The inherited duals are constants for the gradient; row t's own
        scaling carries it."""
        key_vis = vis.any(-2); row_vis = vis.any(-1)
        log_a, log_b = self._marginals(la, lb_all, key_vis, row_vis)
        negC = (S32 / self.eps)                                                # -C / eps, -inf off vis
        key_ok = torch.isfinite(log_a); row_ok = torch.isfinite(log_b)
        dead = (~row_ok).unsqueeze(-1) | (~key_ok).unsqueeze(-2)
        negC_safe = negC.masked_fill(dead | ~torch.isfinite(negC), 0.0)
        la = log_a.masked_fill(~key_ok, 0.0); lb = log_b.masked_fill(~row_ok, 0.0)
        zero1 = torch.zeros_like(la[..., :1])
        f = torch.cat([f_prev.detach(), zero1], -1); gd = torch.cat([gd_prev.detach(), zero1], -1)
        with torch.no_grad():
            for _ in range(self.causal_iters):
                gd = (la - torch.logsumexp(negC_safe + f.unsqueeze(-1), dim=-2)).masked_fill(~key_ok, 0.0)
                f = (lb - torch.logsumexp(negC_safe + gd.unsqueeze(-2), dim=-1)).masked_fill(~row_ok, 0.0)
        # row t with the gradient: its own row scaling against the (constant) column duals
        gd_c = gd.detach()
        row_t = negC_safe[..., -1:, :] + gd_c.unsqueeze(-2)                                   # [B,H,1,t+1]
        f_t = lb[..., -1:] - torch.logsumexp(row_t, dim=-1)
        logP = row_t + f_t.unsqueeze(-1)
        A_t = torch.exp(logP.masked_fill((dead | ~torch.isfinite(negC))[..., -1:, :], float("-inf")))
        f = torch.cat([f[..., :-1], f_t.detach()], -1)
        return A_t, f, gd

    # ---- causal decoding: the prefill's duals and the query history, one step = one warm-started row solve
    def init_decode(self, diag, Q):
        return dict(f=diag.extra["dual_f"].detach().clone(), g=diag.extra["dual_g"].detach().clone(), Q=Q.detach())

    @torch.no_grad()
    def decode_step(self, cache: dict, K_kv, q_t, scaling: float):
        """q_t [B,H,1,d] the new query; K_kv [B,H_kv,t+1,d] every key so far.  Returns A_t [B,H,1,t+1] fp32."""
        cache["Q"] = torch.cat([cache["Q"], q_t.detach()], -2)
        Qh = cache["Q"]; g = Qh.shape[1] // K_kv.shape[1]
        n = Qh.shape[-2]
        S32 = (torch.matmul(_fp(Qh), repeat_kv(_fp(K_kv), g).transpose(-1, -2)) * scaling)
        vis = torch.tril(torch.ones(n, n, dtype=torch.bool, device=S32.device)).view(1, 1, n, n).expand(S32.shape[0], S32.shape[1], n, n)
        S32 = S32.masked_fill(~vis, float("-inf"))
        la, lb = self._heads(K_kv, Qh)
        A_t, f, gd = self._row_solve(S32, vis, la, lb, cache["f"], cache["g"])
        cache["f"], cache["g"] = f, gd
        return A_t

    def _block(self, S, vis, K_kv, Q):
        """stock MESH on one block (the whole teacher-forced matrix, or the prompt as the prefill)."""
        B, H, n_q, n_k = S.shape
        g = H // K_kv.shape[1]
        with torch.autocast(device_type=S.device.type, enabled=False):
            S32 = _fp(S).masked_fill(~vis, float("-inf"))
            key_vis = vis.any(-2)                                           # [B,H,n_k] seen by >= 1 real query
            row_vis = vis.any(-1)                                           # [B,H,n_q] real rows
            la, lb = self._heads(K_kv, Q)
            log_a, log_b = self._marginals(la, lb, key_vis, row_vis)
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
            A, f, gd = sinkhorn_log(C_st, log_a, log_b, self.eps, self.outer_iters, "row", return_duals=True)
            a = log_a.exp().masked_fill(~key_vis, 0.0)
            col_resid = (A.sum(-2) - a).abs().masked_fill(~key_vis, 0.0).amax(-1)       # [B,H] logged per layer
        diag = Diagnostics(A_sm=None, A=A, extra={"row_sum": A.sum(-1), "col_resid": col_resid,
                                                  "a": a, "b": log_b.exp().masked_fill(~row_vis, 0.0),
                                                  "dual_f": f.detach(), "dual_g": gd.detach()})
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
