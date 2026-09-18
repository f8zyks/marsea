"""
Sec. 3 primitives.  `sorted_prefix_stat` is the one place the mechanism's numerics live:
write it once, test it (T1/T2/T13), call it from everywhere.

Conventions (spec Sec. 3, batched form):
  * every program runs on the LAST dim of a padded tensor, restricted to a boolean `mask`;
  * pads are written as NEG_PAD = -1e30 (never -inf: a -inf corrupts G), sorted last, and
    EXCLUDED from the prefix count (H5);
  * all arithmetic in fp32 (or fp64 for gradcheck); half types are upcast on entry (H1);
  * exact zeros are produced by clamp and never perturbed by an epsilon (H4);
  * every division goes through a safe denominator BEFORE any torch.where (H11).
"""
from __future__ import annotations
from typing import Optional
import torch

NEG_PAD = -1e30          # spec Sec. 3: pad value, sorts last, excluded from cs / k
TOL_QUOTA = 1e-9         # spec Sec. 3 / H6: a quota <= TOL_QUOTA is treated as zero
INF = float("inf")


def _fp(x: torch.Tensor) -> torch.Tensor:
    """H1: programs run in fp32; fp64 is kept (gradcheck, T11's fp64 pass)."""
    return x if x.dtype == torch.float64 else x.float()


# --------------------------------------------------------------------------- Lem. 1 / k*
def sorted_prefix_stat(z: torch.Tensor, mask: torch.Tensor):
    """Lemma 1's prefix statistic G and the sparsemax support size k*, batched, ragged via mask.

    z    [..., L]  scores (values at ~mask are ignored)
    mask [..., L]  bool validity
    returns
      kstar  [...]      int-valued float: |supp(sparsemax(z on mask))|; 0 if no valid entry
      psi    [...]      the threshold: p = clamp(z - psi, 0) on mask   (psi = (cs[k*-1] - 1)/k*)
      G      [..., L]   G[r] = sum_{t<=r}(u[t] - u[r]) in SORTED order, +inf at pads (non-decreasing)
      u      [..., L]   sorted values (descending, stable), pads last
      mvalid [..., L]   validity in sorted order
    """
    z = _fp(z)
    mask = mask.bool()
    zp = z.masked_fill(~mask, NEG_PAD)
    u, order = torch.sort(zp, dim=-1, descending=True, stable=True)      # H8: stable sort
    mvalid = torch.gather(mask, -1, order)
    uv = torch.where(mvalid, u, torch.zeros_like(u))                     # pads contribute 0 to cs
    cs = torch.cumsum(uv, dim=-1)                                        # cs[r] = u[0]+...+u[r]
    k = torch.cumsum(mvalid.to(z.dtype), dim=-1)                         # 1-based prefix COUNT of valid entries
    G = cs - k * u                                                       # G[r] = sum_{t<=r} (u[t]-u[r])
    G = torch.where(mvalid, G, torch.full_like(G, INF))
    cond = (G < 1.0) & mvalid                                            # 1 + k*u > cs  <=>  G < 1
    kstar = (k * cond).amax(dim=-1)                                      # max{k : G(k) < 1}
    idx = (kstar.long() - 1).clamp_min(0).unsqueeze(-1)
    cs_k = torch.gather(cs, -1, idx).squeeze(-1)                         # cs[k*-1]  (0-based)  -- NOT cs[k*]
    psi = (cs_k - 1.0) / kstar.clamp_min(1.0)
    return kstar, psi, G, u, mvalid


# --------------------------------------------------------------------------- sparsemax (alpha = 2)
class SparsemaxMasked(torch.autograd.Function):
    """sparsemax on the last dim restricted to `mask`.  Exact zeros.
    Jacobian on the support S:  diag(1_S) - 1_S 1_S^T / |S|   (spec Sec. 3).

    Returns (p, kstar, psi): the forward already computes the sorted prefix statistic, and a caller that wants the
    diagnostics would otherwise pay a second full sort over the identical input (~25 % of the dense normaliser's
    transient peak, and once per chunk on the chunked path).  k* and psi are marked non-differentiable, so there is
    no second gradient path into z; they were previously handed out through a class-level `last_stats`, which left a
    stale device tensor reachable for the process lifetime and was process-global state on a path autograd
    recomputes lazily (review d5bd980)."""

    @staticmethod
    def forward(ctx, z, mask):
        kstar, psi, _, _, _ = sorted_prefix_stat(z, mask)
        p = torch.clamp(_fp(z) - psi.unsqueeze(-1), min=0.0)
        p = torch.where(mask, p, torch.zeros_like(p))
        ctx.save_for_backward(p > 0)
        kstar, psi = kstar.detach(), psi.detach()
        ctx.mark_non_differentiable(kstar, psi)
        return p, kstar, psi

    @staticmethod
    def backward(ctx, grad_p, _grad_kstar, _grad_psi):
        (supp,) = ctx.saved_tensors
        s = supp.to(grad_p.dtype)
        n = s.sum(-1, keepdim=True).clamp_min(1.0)
        mean = (grad_p * s).sum(-1, keepdim=True) / n
        return s * (grad_p - mean), None


def sparsemax_masked(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """p = argmax_{p in simplex(mask)} <p, z> - 1/2 ||p||^2 ; p = 0 off mask; p = 0 on an empty mask."""
    return SparsemaxMasked.apply(z, mask.bool())[0]


def sparsemax_masked_with_stats(z: torch.Tensor, mask: torch.Tensor):
    """sparsemax together with the (k*, psi) its own forward already computed.  Exactly the values
    sorted_prefix_stat would return."""
    return SparsemaxMasked.apply(z, mask.bool())


# --------------------------------------------------------------------------- entmax (alpha != 2)
def entmax_bisect_masked(z: torch.Tensor, mask: torch.Tensor, alpha: float, iters: int = 50) -> torch.Tensor:
    """alpha-entmax by bisection on the dual threshold t (spec Sec. 3 [v4]; harness L22-33).
    Bounds [(alpha-1) max z - 1, (alpha-1) max z]; zeros are exact (the clamp); the final
    renormalisation absorbs the bisection residual; the gradient is the unrolled iteration's.
    Used by B4 at alpha = 1.5 only."""
    z = _fp(z)
    mask = mask.bool()
    am1 = alpha - 1.0
    zm = (am1 * z).masked_fill(~mask, -INF)
    any_valid = mask.any(-1, keepdim=True)
    zmax = torch.where(any_valid, zm.amax(-1, keepdim=True), torch.zeros_like(zm[..., :1]))
    zm_safe = zm.masked_fill(~mask, 0.0)                     # never let -inf meet arithmetic (H13)
    lo, hi = zmax - 1.0, zmax
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        p = torch.clamp(zm_safe - mid, min=0.0) ** (1.0 / am1)
        p = torch.where(mask, p, torch.zeros_like(p))
        s = p.sum(-1, keepdim=True)
        big = s > 1.0
        lo = torch.where(big, mid, lo)
        hi = torch.where(big, hi, mid)
    p = torch.clamp(zm_safe - 0.5 * (lo + hi), min=0.0) ** (1.0 / am1)
    p = torch.where(mask, p, torch.zeros_like(p))
    den = p.sum(-1, keepdim=True)
    den_safe = torch.where(den > 0, den, torch.ones_like(den))            # H11
    return p / den_safe


def entmax_masked(z: torch.Tensor, mask: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    if abs(alpha - 2.0) < 1e-12:
        return sparsemax_masked(z, mask)
    return entmax_bisect_masked(z, mask, alpha)


# --------------------------------------------------------------------------- projections
def proj_eq_masked(v: torch.Tensor, s: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """argmin .5||a-v||^2  s.t. sum a = s, a >= 0  on `mask`;  s [...] > 0 guaranteed by the caller.
    Exact: the simplex of radius s is s * sparsemax(v / s)."""
    s_ = s.unsqueeze(-1)
    return s_ * sparsemax_masked(_fp(v) / s_, mask)


def proj_le_masked(v: torch.Tensor, s: torch.Tensor, mask: torch.Tensor, tol: float = TOL_QUOTA,
                   binds: Optional[torch.Tensor] = None):
    """argmin .5||a-v||^2  s.t. sum a <= s, a >= 0  on `mask`.

    s <= tol  =>  a = 0  (spec Sec. 3 [v4]: the zero-quota row is the MAJORITY row under random
    scores; a batched where() evaluates v/0 in the dead branch and its backward is NaN unless the
    divisor is made safe BEFORE the where -- H11).
    binds: the constraint's binding set, DECIDED BY THE CALLER, when the fp row sum of `v` is not a trustworthy test of
    "sum v > s".  The unit cap passes it (normalizer.relation_excess): a softmax row's fp32 total is off one by a
    kernel-dependent amount -- up to n_k eps / 2 on a razor-edge row under a lane-wise CPU kernel -- so `sum(v) > 1 +
    slack` bound the cap on rows the relation never touched, at any slack a constant could give (reviews 30372ae,
    e982f83, e16a843).  The quota cap (s = cbar_i / tau_i, target Atil on E) leaves it None and tests the sum.
    returns (a, theta) with theta the implied dual for logging: 0 if the cap was slack, else
    max over supp(a) of (v_j - a_j)  (no gradient; H7: theta is derived, never a leaf)."""
    v = _fp(v)
    mask = mask.bool()
    s = _fp(s)
    # right-derivative at v == 0: clamp() has derivative 0 there, and every relation pair the fan-out sparsemax excluded
    # has Atil == 0 EXACTLY, which would cut the ST signal of site (2) off from site (3) (review 2026-09-09, A10)
    w = torch.where(v >= 0, v, torch.zeros_like(v)) * mask.to(v.dtype)
    sw = w.sum(-1)
    pos = s > tol
    s_safe = torch.where(pos, s, torch.ones_like(s))                     # safe divisor before the where
    pe = proj_eq_masked(v, s_safe, mask)
    if binds is None:
        binds = sw > s
    else:
        binds = binds.bool().expand_as(sw) if binds.shape != sw.shape else binds.bool()
        # a cap never ADDS mass: if this projection's own fp32 arithmetic finds the row at or below the target (its
        # internal prefix sum is a third kernel-dependent total; on a razor-edge row it can sit below `s` while the
        # caller's exact excess says the row is over), proj_eq would scale the row UP with theta < 0 and resurrect
        # Stage-1 zeros.  Such a row is left as it is (review 7814665 C).
        binds = binds & (implied_threshold(v, pe, mask, torch.ones_like(binds)) > 0)
    a = torch.where(binds.unsqueeze(-1), pe, w)
    a = torch.where(pos.unsqueeze(-1), a, torch.zeros_like(a))           # proj_le(v, s <= tol) := 0
    theta = implied_threshold(v, a, mask, binds & pos)
    return a, theta


@torch.no_grad()
def implied_threshold(v: torch.Tensor, a: torch.Tensor, mask: torch.Tensor, binds: torch.Tensor) -> torch.Tensor:
    """theta for logging: 0 if the cap was slack, else v_j - a_j on supp(a) (constant over the support)."""
    on = (a > 0) & mask.bool()
    th = torch.where(on, _fp(v) - _fp(a), torch.full_like(_fp(v), -INF)).amax(-1)
    return torch.where(on.any(-1) & binds, th, torch.zeros_like(th))
