"""
Sec. 5: the MarSea normaliser, dense reference implementation.

Order of operations is fixed by spec Sec. 15 and must not be altered:
  row softmax (the ONLY normalisation, ever)  ->  relation E, gate g
  STEP 1 fan-out  cbar_j = sum_{E_.j} A_sm            (inherited, never predicted; reads A_sm, never S)
  STEP 2 fan-out  p = entmax(tau_j S on E_.j); Atil = A_sm + g (cbar_j p - A_sm)      exact zeros
  STEP 1 fan-in   a1 = proj_le(Atil_i., 1)             (a CAP; identity on sub-unit rows; NOT a softmax)
  STEP 2 fan-in   u = proj_le(Atil_iE, cbar_i/tau_i);  A = a1 + g (tau_i u - a1)       ceiling, never equality
The target of fan-in step 2 is Atil (pre-cap); its quota cbar_i comes from a1 (post-cap).  T14.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn

from .primitives import (sorted_prefix_stat, entmax_masked, proj_le_masked, sparsemax_masked_with_stats, TOL_QUOTA, _fp)
from .relation import RelationHead, straight_through, hard_concrete_gate, repeat_kv
from .heads import column_stats, row_summary, TauK, TauQ, TAU_MIN

NU_EPS = 1e-12
CAP_TOL = 1e-6          # the unit cap binds when the RELATION's mass excess exceeds this -- see relation_excess


EXCESS_CHUNK = 1024     # keys per fp64 partial in row_masses: bounds the fp64 temporaries to [B,H,n_q,1024]


@torch.no_grad()
def row_masses(Atil: torch.Tensor, A_sm: torch.Tensor, E: torch.Tensor):
    """(excess, sm_mass), both [B,H,n_q] in fp64, from ONE key-chunked pass:

        excess_i  = sum_{j in E_i.} (Atil_ij - A_sm_ij)          by how much the relation pushed the row over its
                                                                   softmax mass  ( = sum_j Atil_ij - sum_j A_sm_ij
                                                                   exactly, since Atil == A_sm bitwise off E )
        sm_mass_i = sum_j A_sm_ij                                  the row's own softmax mass as the fp32 tensor holds
                                                                   it: 1 + delta_kernel, NOT 1

    The unit cap binds iff excess_i > CAP_TOL, and a binding row is projected onto sm_mass_i (unit_cap).  Neither
    quantity is the fp32 row total the kernel produced -- both are fp64 sums of the fp32 entries, which is what makes
    the decision and the target independent of the softmax KERNEL: an fp32 softmax total is off one by ~1e-6 on CUDA
    (pairwise), by n_k eps / (2 L) on an L-lane x86 kernel (a razor-edge row -- one spike, a flat tail just under half
    an ulp of it -- loses 1/L of its tail: 6e-5 at 8K on 8 lanes, 3e-5 on 16), and by up to n_k eps / 2 ~ 1e-3 for a
    scalar reduction.  Three rounds of slack constants on `sum Atil > 1 + slack` were each right on one device
    (reviews 30372ae C, e982f83 D, e16a843 B; the general solution after the last).

    Error of `excess`: both totals are fp64 sums of the fp32 entries (cast on the way in: an fp32 subtraction of two
    entries not within a factor of two of each other rounds, up to 1.2e-7 in total; review 7814665 D-1), so the excess
    of the fp32 tensors carries only fp64 rounding -- the two sums cancel to ~n_k 2^-53 ~ 2e-12 at 16K, six orders
    below CAP_TOL, and an empty relation gives EXACTLY 0.0 (the two sums are the same operations on bitwise-equal
    inputs).  The decision therefore has no error of its own.  What remains is that the fp32 Atil is itself a rounded
    value of the exact-arithmetic one -- relative to the excess, and at the decision boundary ~1e-8 (measured; far from
    it discrete sparsemax-support flips between fp32 and fp64 show up as 1e-4, which is not an error of THIS quantity).
    Rows with no relation entry have excess exactly 0.0, so a forced-empty relation can never bind the cap: T0 and
    sanity 5(a) rest on that.

    Implementation: two fp64 row sums per key chunk of EXCESS_CHUNK, differenced.  `E` is never read: Atil == A_sm
    bitwise off the relation (both gates are hard in the forward, g == E exactly), so sum_J Atil - sum_J A_sm IS the
    masked sum.  `sum(dtype=float64)` over the WHOLE row copies the [B,H,T,T] tensor to fp64 on CUDA (+1.5 GB at 8K,
    head_block 2); `torch.nonzero` forces a host sync per call and holds ~40 bytes per relation entry (1.75 GB at
    rho = 0.5); the masked chunked form of 9bacef9 held four fp64 [B,H,n_q,1024] temporaries (0.375 GiB on CUDA at
    8K / head_block 2, 0.50 GiB on CPU -- review 9bacef9 B-1).  This form holds one (0.125 GiB, 16 ms against 32),
    is rho-independent and sync-free, and its output was bitwise equal to the masked form on every case measured."""
    B, H, n_q, n_k = E.shape
    excess = torch.zeros(B, H, n_q, dtype=torch.float64, device=E.device)
    sm_mass = torch.zeros(B, H, n_q, dtype=torch.float64, device=E.device)
    for j0 in range(0, n_k, EXCESS_CHUNK):
        J = slice(j0, min(n_k, j0 + EXCESS_CHUNK))
        s_J = A_sm[..., J].sum(-1, dtype=torch.float64)
        sm_mass += s_J
        excess += Atil[..., J].sum(-1, dtype=torch.float64) - s_J
    return excess, sm_mass


@torch.no_grad()
def relation_excess(Atil: torch.Tensor, A_sm: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """the relation's mass excess per row (see row_masses)."""
    return row_masses(Atil, A_sm, E)[0]


def unit_cap(Atil: torch.Tensor, vis: torch.Tensor, excess: torch.Tensor, sm_mass: torch.Tensor):
    """STEP 1 fan-in, proj_le(Atil_i., 1), as the fp32 tensors realise it.  Returns (a1, theta, binds).

    Binds iff the relation's excess exceeds CAP_TOL.  A binding row is projected onto sm_mass_i -- the row's own softmax
    mass, which IS one in exact arithmetic -- and not onto the literal constant 1: with the fp32 total below one
    (delta_kernel < -excess, a razor-edge row on a lane-wise CPU) the constant target made proj_eq scale the row UP,
    theta went negative and Stage-1 zeros were resurrected in a1 -- and a1's off-relation entries are A (INV-4), so the
    output moved, device-dependently (review 7814665 C).  Onto sm_mass the projection removes exactly the relation's
    excess.  proj_le_masked adds the last guard: if ITS fp32 arithmetic finds nothing above the target (theta <= 0 --
    the projection's internal sum is a third, kernel-dependent total), the row is left as it is.  A cap therefore never
    adds mass, never lifts a zero, and a1 <= Atil elementwise, on any kernel.  The target is a constant (one) of the
    exact program, so it is detached: no gradient flows into it.

    The guard has a WINDOW: a row whose excess exceeds CAP_TOL by less than the fp32 prefix scan's error over the row
    can be left uncapped (theta <= 0 in fp32 although the exact excess is positive).  The window is a device property
    like the kernel's row-sum error -- torch's CPU cumsum accumulates in higher precision (1.3e-7 here), CUDA's fp32
    scan gives 1.1e-6 at 8K and 2.0e-6 at 16K on this GPU, i.e. ABOVE CAP_TOL, and a strictly sequential fp32 scan could
    reach n_k eps / 4 -- so `check_unit_cap.py` bisects it on the target device and gates it against the tolerance INV-3
    allows (review 9bacef9 D).  A row inside the window keeps at most its excess -- less than the window -- above its
    softmax mass; what the mechanism promises on any device is sum_j A_ij <= sum_j A_sm_ij + max(CAP_TOL, window),
    never that the binding SET is device-independent inside the window.  Returned `binds` is the decision actually
    taken; `theta` is the cap's own dual (Diagnostics.cap_theta), positive on every row that bound.

    The dense and chunked paths differ in A_sm (torch.softmax against exp((S - m) - r)), so their sm_mass differs by
    the kernel row-sum difference (up to 6e-5 on a razor row at 16K on this CPU) while their excess differs by
    <= 4e-8 (measured; review 9bacef9 B-2): a binding row's target is path-dependent by the former, its decision by the
    latter plus the guard window."""
    binds = cap_binds_from_excess(excess)
    s = torch.where(binds, sm_mass, torch.ones_like(sm_mass)).to(Atil.dtype).detach()
    a1, theta = proj_le_masked(Atil, s, vis, binds=binds)
    return a1, theta, binds & (theta > 0)


def unit_cap_relation(Atil: torch.Tensor, A_sm: torch.Tensor, E: torch.Tensor, excess: torch.Tensor, tail_share=None):
    """STEP 1 fan-in with the excess taken back FROM THE RELATION (cap_mode = "relation"; owner's decision, 2026-09-21).

    The whole-row cap subtracts ONE threshold from every visible entry.  The off-relation part of an answer row is
    ~3,400 softmax entries each far below that threshold, so wherever the cap binds the whole softmax tail becomes exact
    zeros (P5, step 900: 0.2 % of it survives on binding rows, and the cap binds on 81 % / 91 % of answer rows at
    m = 4 / 16) -- although the excess was the relation's doing (the column programme pays it 1.0 -> 1.74 units).
    Here the relation is projected onto a budget of its own -- one threshold, exact zeros, restricted to E_i. -- and the
    off-relation entries are never thresholded:

        budget_i = R_i + lambda_i T_i,     R_i = sum_{j in E_i.} A_sm_ij  (>= 0 by construction),   T_i = the tail's softmax mass

    lambda = 0 (tail_share None): the relation redistributes the mass softmax already gave it, the tail is bitwise A_sm.
    With that budget a member pair can only LOSE mass by being a member (P8, 2026-09-21: the copy source goes
    0.72 -> 0.57 at m = 16, and the LM gradient takes it out of the relation: 1.00 -> 0.20 of rows in 150 steps).
    lambda > 0 (owner's decision, 2026-09-21; a float or a [B,H,n_q] tensor from heads.TailShare): the relation may keep
    up to lambda_i T_i of what the column programme over-paid it, and the tail pays for exactly what was kept by ONE
    factor per row -- scaled, never thresholded, so its support and its shape are untouched:

        kept_i = min(excess_i, lambda_i T_i)  (>= 0),      a1_ij = A_sm_ij (1 - kept_i / T_i)   off E_i.

    Three regimes, continuous across both boundaries: excess <= 0 untouched; 0 < excess <= lambda T the relation keeps
    Atil and the tail is scaled; above that the relation is projected onto the budget and the tail is scaled by
    1 - lambda.  Wherever the relation over-paid, the row sums to its softmax mass.  A cap still never adds mass, never
    lifts a zero, a1 <= Atil elementwise.  The budget is not a constant of the programme: its gradient flows into A_sm
    and into lambda."""
    from .primitives import TOL_QUOTA
    Ef = E.to(Atil.dtype)
    R = (A_sm * Ef).sum(-1).clamp_min(0.0)
    if tail_share is None:
        budget = R; binds = cap_binds_from_excess(excess)
    else:
        T_mass = (A_sm * (1.0 - Ef)).sum(-1)
        room = torch.as_tensor(tail_share, dtype=Atil.dtype, device=Atil.device) * T_mass       # lambda_i T_i
        budget = R + room
        binds = cap_binds_from_excess(excess - room.to(excess.dtype))
    a_rel, theta = proj_le_masked(Atil.masked_fill(~E, 0.0), budget, E, binds=binds)
    # a relation with (numerically) NO budget that the column programme paid: proj_le returns zeros with theta = 0,
    # which read as "did not bind" and left the row at 1 + its excess.  It binds, and the relation goes to zero.
    took = binds & ((theta > 0) | (budget <= TOL_QUOTA))
    rel = torch.where(took.unsqueeze(-1) & E, a_rel.to(Atil.dtype), Atil)
    if tail_share is None:
        return rel, theta, took
    kept = torch.minimum(((rel - A_sm) * Ef).sum(-1).clamp_min(0.0), room)                     # what the relation holds above R
    T_safe = torch.where(T_mass > 0, T_mass, torch.ones_like(T_mass))
    factor = torch.where(T_mass > 0, 1.0 - kept / T_safe, torch.ones_like(T_mass))
    a1 = torch.where(E, rel, Atil * factor.unsqueeze(-1))                                       # Atil == A_sm off E
    return a1, theta, took


def apply_unit_cap(norm, Atil: torch.Tensor, A_sm: torch.Tensor, E: torch.Tensor, vis: torch.Tensor, excess, sm_mass,
                   Q=None, col_size=None, p=None, head_offset: int = 0):
    """the cap the normaliser is configured with -- one entry point for the dense form, the triggered form and decode_step.
    Q / col_size / p: the rows' queries and what row_relation_stats reads, for the learned tail share only."""
    mode = getattr(norm, "cap_mode", "row")
    if mode == "row":
        return unit_cap(Atil, vis, excess, sm_mass)
    lam = None
    if getattr(norm, "tail_share_head", None) is not None:
        lam = norm.tail_share_head(_fp(Q), norm.tail_share_head.summary(Atil, A_sm, E, vis, col_size, p), head_offset).to(Atil.dtype)
    elif getattr(norm, "tail_share", 0.0) > 0.0:
        lam = float(norm.tail_share)
    return unit_cap_relation(Atil, A_sm, E, excess, lam)


def cap_binds_from_excess(excess: torch.Tensor) -> torch.Tensor:
    """the unit cap's binding set: rows the relation pushed more than CAP_TOL over their softmax mass."""
    return excess > CAP_TOL


def softmax_about_max(S_masked: torch.Tensor, lse_m: torch.Tensor, lse_r: torch.Tensor) -> torch.Tensor:
    """exp(S - lse) evaluated as exp((S - m) - r): m the row max (a value of the row itself, detached), r in [0, log n_k].
    Two failure modes it avoids, both coherent over the whole row so they land on the row SUM the unit cap tests:
    exp(S - (m + r)) rounds lse at the scores' magnitude (7.6e-6 at |lse| ~ 64 on CUDA, flat in n_k -- review e982f83
    D-8), and torch.softmax's lane-wise denominator drops the sub-ulp tail (n_k eps / 2L on CPU -- review e16a843 B-1).
    About the max the error is ~ulp(r): ~1e-6, flat in n_k."""
    return torch.exp((S_masked - lse_m.unsqueeze(-1)) - lse_r.unsqueeze(-1))


def row_softmax(S: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
    """Standard attention in fp32; -inf entries give exact 0; a fully masked row gives 0 (spec Sec. 2 [v4]).

    torch.softmax, deliberately: T0 requires the patched SoftmaxNorm arm to reproduce the unmodified backbone, and in
    bf16 a ~1e-7 change in the weights flips roundings that the network amplifies (computing this about the row max,
    softmax_about_max, moved T0's logits by 1.37e-2 relative).  Its row sums are therefore whatever the device's kernel
    makes them (off one by up to n_k eps / 2L on a razor-edge row under a lane-wise CPU kernel) -- which is why nothing
    in the mechanism reads them: the unit cap decides on relation_excess, and SAN-1 reports them (review e16a843)."""
    S32 = _fp(S).masked_fill(~vis.bool(), float("-inf"))
    A = torch.softmax(S32, dim=-1)
    anyv = vis.bool().any(-1, keepdim=True)
    return torch.where(anyv, A, torch.zeros_like(A))


def broadcast_vis(vis: torch.Tensor, S) -> torch.Tensor:
    """`S` may be a tensor or just its shape -- only the rank and the shape are read, and callers that have no such
    tensor should not have to allocate one (a [B,H,T,T] bool is 805 MB at 8K and 3.2 GB at 16K)."""
    shape = S.shape if torch.is_tensor(S) else torch.Size(S)
    vis = vis.bool()
    while vis.dim() < len(shape):
        vis = vis.unsqueeze(0)
    return vis.expand(shape)


@dataclass
class Diagnostics:
    """Everything Sec. 12.4 logs, plus the tensors B5 and the tests need.  All detached by the consumer."""
    E: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    cbar_j: Optional[torch.Tensor] = None
    cbar_i: Optional[torch.Tensor] = None
    tau_j: Optional[torch.Tensor] = None
    tau_i: Optional[torch.Tensor] = None
    psi_j: Optional[torch.Tensor] = None
    kstar: Optional[torch.Tensor] = None
    nu: Optional[torch.Tensor] = None
    theta: Optional[torch.Tensor] = None
    Rtil: Optional[torch.Tensor] = None
    supp_rel: Optional[torch.Tensor] = None       # |supp(A[i, E_i.])| per row   (B5 needs this)
    cap_binds: Optional[torch.Tensor] = None      # the unit cap's DECISION (excess > CAP_TOL and theta > 0)
    cap_theta: Optional[torch.Tensor] = None      # the unit cap's own dual; `theta` below is Stage 2's (review 9bacef9 E)
    p: Optional[torch.Tensor] = None
    a1: Optional[torch.Tensor] = None
    Atil: Optional[torch.Tensor] = None
    A_sm: Optional[torch.Tensor] = None
    c: Optional[torch.Tensor] = None
    u: Optional[torch.Tensor] = None
    A: Optional[torch.Tensor] = None
    extra: dict = field(default_factory=dict)

    def detach(self) -> "Diagnostics":
        out = Diagnostics()
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                setattr(out, k, v.detach())
            elif isinstance(v, dict):
                setattr(out, k, {kk: (vv.detach() if torch.is_tensor(vv) else vv) for kk, vv in v.items()})
            else:
                setattr(out, k, v)
        return out


@dataclass
class State:
    """Per-forward hand-off between patched layers (Sec. 5.3): nu from the previous patched layer.
    NOT detached (paper: nu carries a gradient to the previous layer's tau_j through p)."""
    nu_prev: Optional[torch.Tensor] = None
    nu_next: Optional[torch.Tensor] = None


class MarSeaNormalizer(nn.Module):
    """normalize(S, vis, K_kv, Q, state) -> (A, Diagnostics).

    Configuration knobs (all E9 ablations unless stated):
      r                 relation rank (16)
      alpha             2.0 (sparsemax; deployed).  1.5 only meaningful for B4, not here.
      key_only_tau      B3: TauK's field inputs zeroed
      no_nu             TauK without nu
      key_only_relation E from u_phi(k_j) alone
      tau_i_pinned      tau_i == 1 (Stage 2 reduced to the unit cap)
      tau_j_global      one learnable scalar tau_j for every key
      quota_mode        "inherited" (the mechanism) | "uniform" (the predicted-budget ablation: every
                        active column of a (b,h) slice gets the same quota, the slice's in-relation mass
                        divided by its number of active columns -- total in-relation mass preserved)
      gate              "st" (straight-through) | "hard_concrete"
      per_head          0 (shared heads) | H (one head per Q-head)
      head_block        None = every Q-head at once.  An int processes that many Q-heads at a time: the programs are
                        per Q-head, so this is arithmetically identical and divides the [B, H, n_q, n_k] transients --
                        which dominate memory at 8K and above -- by H / head_block.
      K_ret             None = the exact flat solve (deployed default).  An int DEPLOYS the hierarchical top-K of
                        prop:tournament (Sec. 2.4): top K_ret per block of `block_size` by ORIGINAL score, one exact
                        solve at the root -- exact whenever K_ret >= k*, and the E9 K_ret ablation.  The E8 truncation
                        flag (k* >= 0.9 K_ret) is only meaningful when this is set.
    """

    def __init__(self, d_head: int, r: int = 16, alpha: float = 2.0, tau_min: float = TAU_MIN,
                 key_only_tau: bool = False, no_nu: bool = False, key_only_relation: bool = False,
                 tau_i_pinned: bool = False, tau_j_global: bool = False, quota_mode: str = "inherited",
                 gate: str = "st", per_head: int = 0, hidden: int = 64, K_ret: Optional[int] = None,
                 block_size: int = 512, head_block: Optional[int] = None, head_block_recompute: bool = True,
                 relation_per_head: int = 0, trigger_tau: bool = False, size_aware_tau_i: bool = False,
                 cap_mode: str = "row", tail_share: float = 0.0, tail_share_learned: bool = False,
                 tail_share_max: float = 0.5, tail_share_init: float = 0.05, tau_i_floor: float = 0.0,
                 tail_share_min: float = 0.0, relation_heads: Optional[list] = None, relation_topk: int = 0,
                 relation_answer_rows_only: bool = False):
        super().__init__()
        assert quota_mode in ("inherited", "uniform")
        assert gate in ("st", "hard_concrete")
        self.d_head = d_head
        self.alpha = alpha
        self.tau_min = tau_min
        self.tau_i_pinned = tau_i_pinned
        self.tau_j_global = tau_j_global
        self.quota_mode = quota_mode
        self.gate = gate
        self.K_ret = K_ret                      # None: exact flat solve.  int: hierarchical top-K (prop:tournament)
        self.block_size = block_size
        self.head_block = head_block            # None: all Q-heads at once.  int: that many at a time (memory / H_block)
        self.head_block_recompute = head_block_recompute   # checkpoint each block (needed for the saving to appear)
        self.want_dense_diag = True             # the backbone clears this when it is not collecting diagnostics
        self.want_E = True                      # E alone: E8 needs it on every LOGGING step, the dense programs do not
        self.gate_temperature = 1.0                   # annealed 1.0 -> 0.2 by the trainer for hard-concrete
        # relation_per_head = H: the RELATION alone is per Q-head (U_h, V_h, b0_h), TauK / TauQ stay shared -- the
        # 2026-09-20 recipe.  `per_head` (E9) makes all three heads per-head and keeps a scalar b0.
        self.relation = RelationHead(d_head, r, key_only=key_only_relation, per_head=(relation_per_head or per_head),
                                     per_head_b0=bool(relation_per_head))
        # trigger_tau: the column temperature of a TRIGGERED solve is predicted then, from the column as it stands
        # (heads.TauKTrigger).  Read by marsea/triggered.py and causal.decode_step; the prompt's one solve keeps TauK.
        from .heads import TauKTrigger
        self.tauK_trig = TauKTrigger(d_head, hidden, tau_min, per_head=per_head) if trigger_tau else None
        assert cap_mode in ("row", "relation")
        self.cap_mode = cap_mode                # "relation": Step-1 fan-in takes the excess back from the relation only
        # the relation may take up to lambda x (the row's off-relation softmax mass); the tail is SCALED for it, never
        # thresholded.  tail_share: one fixed lambda.  tail_share_learned: lambda per row (per Q-head with per_head) from
        # heads.TailShare, in [0, tail_share_max], initialised at tail_share_init.  Both need cap_mode = "relation".
        assert 0.0 <= tail_share <= 1.0 and not (tail_share > 0.0 and tail_share_learned)
        assert cap_mode == "relation" or not (tail_share > 0.0 or tail_share_learned), "a tail share needs cap_mode = 'relation'"
        self.tail_share = float(tail_share)
        from .heads import TailShare
        self.tail_share_head = TailShare(d_head, hidden, tail_share_max, tail_share_init, per_head=per_head, lam_min=tail_share_min) if tail_share_learned else None
        self.st_temperature = 1.0               # straight-through backward temperature; the trainer anneals it to 1
        # ---- the PARTITION relation (owner's decision, 2026-09-22; P14 showed the 5 %-coverage relation is 98 % off-block):
        #   relation_heads         the Q-heads (global indices) that carry a relation at all; the others are plain softmax.
        #                          None: every head.  The retrieval heads come from the block metrics of the softmax model.
        #   relation_topk = k      E_i. = the k highest logits of row i among the candidates (a per-row budget) instead of
        #                          logits > 0; the straight-through gate is centred at the row's k-th logit.  b0 is then
        #                          irrelevant to E (a shift), and coverage is k / n_vis by construction.
        #   relation_answer_rows_only  prompt rows carry no relation: only GENERATED rows are members of any column, so a
        #                          column's competitors are the answer rows -- "an earlier answer row claimed this key".
        self.relation_heads = None if relation_heads is None else sorted(int(h) for h in relation_heads)
        self.relation_topk = int(relation_topk)
        self.relation_answer_rows_only = bool(relation_answer_rows_only)
        self.tauK = TauK(d_head, hidden, tau_min, zero_field_inputs=key_only_tau, no_nu=no_nu, per_head=per_head)
        # tau_i_floor = 1: fan-in step 2 can only SHARPEN.  With tau_i < 1 its quota cbar_i / tau_i exceeds what the relation
        # holds, the projection is the identity and the final relation is tau_i x Atil_E: the difference leaves the row
        # (P8 / P9 / P10, 2026-09-21: TauQ learned tau_i 0.63-0.89 at the answer rows; relation mass 0.79 -> 0.51).  With
        # tau_i >= 1 the final relation mass is its quota cbar_i exactly.  0 = the spec's tau_min (as before).
        self.tau_i_floor = float(tau_i_floor)
        self.tauQ = TauQ(d_head, hidden, (self.tau_i_floor if self.tau_i_floor > 0 else tau_min), per_head=per_head, sized=size_aware_tau_i)
        if tau_j_global:
            from .heads import inv_softplus
            self.tau_j_raw = nn.Parameter(torch.tensor(inv_softplus(1.0 - tau_min)))

    # ------------------------------------------------------------------ the normaliser
    def normalize(self, S: torch.Tensor, vis: torch.Tensor, K_kv: torch.Tensor, Q: torch.Tensor,
                  state: Optional[State] = None, *, logits_override: Optional[torch.Tensor] = None,
                  tau_j_override=None, tau_i_override=None, E_override: Optional[torch.Tensor] = None,
                  head_offset: int = 0, n_prefill: Optional[int] = None, decode_K_ret: int = 64, prompt_rows: Optional[int] = None):
        """n_prefill: the TRIGGERED teacher-forced form (marsea/triggered.py) -- rows < n_prefill are the prompt and
        are normalised as the generation prefill is; every later row re-solves the columns its own relation triggers
        over the members that exist by then, as decode_step does.  None: the full-sequence form.

        With head_block set, the Q-heads are processed in blocks and the results concatenated.  Every program is
        per Q-head (spec Sec. 6.2: "the programs, E, tau_j, tau_i, nu are all per Q-head ... log per Q-head, aggregate
        never"), so this is the head-axis analogue of the key chunking of Sec. 6.6 -- arithmetically identical, and it
        divides the [B, H, n_q, n_k] transients that dominate memory by H / head_block."""
        trig = {}
        # prompt_rows: how many leading rows are PROMPT rows, for relation_answer_rows_only on the full-sequence form (a
        # teacher-forced pass, or the generation prefill where every row is one); the triggered form knows from n_prefill
        self._n_prefill_dense = (int(prompt_rows) if prompt_rows is not None else
                                 int(n_prefill) if n_prefill is not None else None)
        if n_prefill is not None and 0 < int(n_prefill) < S.shape[-2] and S.shape[-2] == S.shape[-1]:
            assert tau_j_override is None and tau_i_override is None and E_override is None, "overrides: full-sequence form only"
            trig = dict(n_prefill=int(n_prefill), decode_K_ret=int(decode_K_ret))
        if self.head_block and S.shape[1] > self.head_block:
            return self._normalize_by_head_block(S, vis, K_kv, Q, state, logits_override=logits_override,
                                                 tau_j_override=tau_j_override, tau_i_override=tau_i_override,
                                                 E_override=E_override, head_offset=head_offset, **trig)
        return self._dispatch_one(S, vis, K_kv, Q, state, logits_override=logits_override,
                                  tau_j_override=tau_j_override, tau_i_override=tau_i_override, E_override=E_override,
                                  head_offset=head_offset, **trig)

    def select_E(self, logits: torch.Tensor, vis: torch.Tensor, head_offset: int = 0, first_row: int = 0, n_prefill: Optional[int] = None):
        """(E, g) from the relation logits: the hard set and its straight-through gate, under the partition options.
        first_row: the global index of logits' first row (the dense form: 0; the triggered answer block: n_p; decode: t).
        One function for the dense form, the triggered form and decode_step, so the three cannot disagree on E."""
        H = logits.shape[1]; n_q, n_k = logits.shape[-2:]
        lg = logits
        if self.relation_heads is not None:                                          # heads without a relation: E empty, no gradient
            keep = torch.zeros(H, dtype=torch.bool, device=logits.device)
            for h in self.relation_heads:
                if head_offset <= h < head_offset + H: keep[h - head_offset] = True
            lg = lg.masked_fill(~keep.view(1, H, 1, 1), -1e4)
        if self.relation_answer_rows_only and n_prefill is not None and first_row < n_prefill:   # prompt rows: no relation
            rows = torch.arange(first_row, first_row + n_q, device=logits.device) < n_prefill
            lg = lg.masked_fill(rows.view(1, 1, n_q, 1), -1e4)
        if self.relation_topk > 0:
            k = min(self.relation_topk, n_k)
            cand = lg.masked_fill(~vis, float("-inf"))
            kth = cand.topk(k, dim=-1).values[..., -1:]                                # the row's k-th logit
            # a row with fewer than k visible pairs has -inf there: every visible pair qualifies (threshold below them all)
            kth = torch.where(torch.isfinite(kth), kth, torch.full_like(kth, -1e4 + 0.5))
            E = (cand >= kth) & vis & (lg > -1e4 + 1.0)                                # ties: all admitted; masked heads / rows never
            g = straight_through(E, lg - kth.detach(), vis, self.st_temperature)
            return E, g
        E = (lg > 0) & vis
        return E, straight_through(E, lg, vis, self.st_temperature)

    def _dispatch_one(self, S, vis, K_kv, Q, state=None, *, n_prefill=None, decode_K_ret: int = 64, head_offset: int = 0,
                      logits_override=None, tau_j_override=None, tau_i_override=None, E_override=None):
        if n_prefill is None:
            return self._normalize_one(S, vis, K_kv, Q, state, logits_override=logits_override, tau_j_override=tau_j_override,
                                       tau_i_override=tau_i_override, E_override=E_override, head_offset=head_offset)
        from .triggered import normalize_triggered
        return normalize_triggered(self, S, vis, K_kv, Q, state, n_prefill, K_ret=decode_K_ret, head_offset=head_offset,
                                   logits_override=logits_override)

    def _normalize_by_head_block(self, S, vis, K_kv, Q, state, head_offset: int = 0, **kw):
        from .relation import repeat_kv as _rk
        B, H, n_q, n_k = S.shape
        g_kv = H // K_kv.shape[1]
        k_rep = _rk(K_kv, g_kv)                                   # per Q-head keys: the block path runs with g = 1
        vis_b = broadcast_vis(vis, S)
        outs, diags = [], []
        for h0 in range(0, H, self.head_block):
            sl = slice(h0, min(H, h0 + self.head_block))
            sub = {}
            for name, t in kw.items():
                if torch.is_tensor(t) and t.dim() >= 2 and t.shape[1] == H:
                    sub[name] = t[:, sl]
                else:
                    sub[name] = t
            st = State(nu_prev=(state.nu_prev[:, sl] if (state is not None and state.nu_prev is not None) else None))
            if self.head_block_recompute and torch.is_grad_enabled():
                # each block is its own checkpointed segment, so only ONE block's [B, H_block, n_q, n_k] transients are
                # live at a time during the backward.  Without this the concatenation holds every block's and the
                # blocking saves nothing.
                import torch.utils.checkpoint as _cp
                A_h, d_h = _cp.checkpoint(self._dispatch_one, S[:, sl], vis_b[:, sl], k_rep[:, sl], Q[:, sl], st,
                                          use_reentrant=False, head_offset=head_offset + h0, **sub)
            else:
                A_h, d_h = self._dispatch_one(S[:, sl], vis_b[:, sl], k_rep[:, sl], Q[:, sl], st,
                                              head_offset=head_offset + h0, **sub)
            outs.append(A_h); diags.append(d_h)
            if state is not None and st.nu_next is not None:
                state.nu_next = st.nu_next if state.nu_next is None or h0 == 0 else torch.cat([state.nu_next, st.nu_next], 1)
        A = torch.cat(outs, dim=1)
        merged = Diagnostics()
        # the small per-key / per-row diagnostics are always merged; the dense [B, H, n_q, n_k] ones only when a consumer
        # asked for them (the backbone clears want_dense_diag when it is not collecting).  Concatenating them
        # unconditionally allocates a second full-size copy of every program tensor and eats the blocking's saving.
        small = ("cbar_j", "cbar_i", "tau_j", "tau_i", "psi_j", "kstar", "nu", "theta", "Rtil", "supp_rel", "cap_binds",
                 "cap_theta", "c")
        dense = ("logits", "p", "a1", "Atil", "A_sm", "u", "A")
        # E is merged on its own gate.  It used to sit in `dense`, so an ordinary LOGGING step with head_block set
        # produced merged.E = None and the backbone skipped E8 entirely -- no coverage, no |E_.j| histogram, no k*
        # histogram, and sanity["coverage_live"] null in every run (review d5bd980 B-1).  A bool [1,H,T,T] is 0.81 GB
        # at 8K, an eighth of the fp32 tensors beside it, and only on the ~2 % of steps that log.
        for name in small + (("E",) if (self.want_E or self.want_dense_diag) else ()) + (dense if self.want_dense_diag else ()):
            vals = [getattr(d, name) for d in diags]
            if all(torch.is_tensor(v) for v in vals):
                setattr(merged, name, torch.cat(vals, dim=1))
        merged.extra["head_block"] = self.head_block
        if self.want_dense_diag and all("tau_trig_rows" in d.extra for d in diags):      # the triggered form's per-solve tau (monitor)
            merged.extra["tau_trig_rows"] = torch.cat([d.extra["tau_trig_rows"] for d in diags], dim=1)
        return A, merged

    def _normalize_one(self, S: torch.Tensor, vis: torch.Tensor, K_kv: torch.Tensor, Q: torch.Tensor,
                       state: Optional[State] = None, *, logits_override: Optional[torch.Tensor] = None,
                       tau_j_override=None, tau_i_override=None, E_override: Optional[torch.Tensor] = None,
                       head_offset: int = 0):
        out_dtype = S.dtype
        vis = broadcast_vis(vis, S)
        B, H, n_q, n_k = S.shape
        g_kv = H // K_kv.shape[1]
        with torch.autocast(device_type=S.device.type, enabled=False):
            S32 = _fp(S).masked_fill(~vis, float("-inf"))
            # ---- the ONLY normalisation: standard attention (record Sec. 38.1, 41)
            A_sm = row_softmax(S32, vis)                                     # fp32, exact 0 off vis
            c = A_sm.sum(-2)                                                 # [B,H,n_k]  (SAN-1 makes sum_j c_j = n_real)
            # ---- the relation (Sec. 4)
            if logits_override is not None:
                logits = _fp(logits_override).expand(B, H, n_q, n_k)
            else:
                logits = self.relation(K_kv, Q, key_offset=0, n_k_total=n_k, head_offset=head_offset)   # D-31 sink mask inside
            if E_override is not None:
                E = E_override.bool().expand(B, H, n_q, n_k) & vis
                g = E.to(S32.dtype)                                          # no gradient path to logits
            elif self.gate == "hard_concrete" and logits_override is None:
                g, E = hard_concrete_gate((logits > 0) & vis, logits, vis, self.gate_temperature, self.training)
            else:
                E, g = self.select_E(logits, vis, head_offset, first_row=0, n_prefill=getattr(self, "_n_prefill_dense", None))
            # ---- STEP 1 fan-out: the inherited quota reads A_sm, NEVER S     (eq:step1)
            cbar_j = (g * A_sm).sum(-2)                                      # [B,H,n_k]   ST site (1)
            if self.quota_mode == "uniform":                                 # E9: put a predicted budget back
                active = (E.sum(-2) > 0)
                mu = cbar_j.sum(-1, keepdim=True)
                n_act = active.sum(-1, keepdim=True).to(S32.dtype)
                n_safe = torch.where(n_act > 0, n_act, torch.ones_like(n_act))
                cbar_j = torch.where(active, (mu / n_safe).expand_as(cbar_j), torch.zeros_like(cbar_j))
            # ---- STEP 2 fan-out: shape + exact zeros, on the relation only    (eq:program)
            stats = column_stats(S32, vis)                                   # [B,H,n_k,6]
            if state is not None and state.nu_prev is not None:
                nu_prev = _fp(state.nu_prev)
            else:
                nu_prev = torch.ones(B, H, n_k, dtype=S32.dtype, device=S.device)   # first patched layer: 1.0
            if tau_j_override is not None:
                tau_j = torch.as_tensor(tau_j_override, dtype=S32.dtype, device=S.device).expand(B, H, n_k)
            elif self.tau_j_global:
                tau_j = (self.tau_min + torch.nn.functional.softplus(self.tau_j_raw)).expand(B, H, n_k)
            else:
                k_rep = repeat_kv(_fp(K_kv), g_kv)                           # [B,H,n_k,d]
                tau_j = self.tauK(k_rep, stats, nu_prev, head_offset)        # [B,H,n_k]
            zcol = (S32.masked_fill(~E, 0.0) * tau_j.unsqueeze(-2)).transpose(-1, -2)   # [B,H,n_k,n_q]; H13
            ET = E.transpose(-1, -2)
            if self.K_ret is not None and abs(self.alpha - 2.0) < 1e-12:
                from .causal import hierarchical_topk                        # prop:tournament; exact iff K_ret >= k*
                pT, _, _ = hierarchical_topk(zcol, ET, K_ret=self.K_ret, block_size=self.block_size)
                with torch.no_grad():
                    _, psi_j, _, _, _ = sorted_prefix_stat(zcol, ET)
            elif abs(self.alpha - 2.0) < 1e-12:
                # one sort, not two: the sparsemax forward already produced (k*, psi) on this very input
                pT, _, psi_j = sparsemax_masked_with_stats(zcol, ET)
            else:
                pT = entmax_masked(zcol, ET, self.alpha)                     # p_ij = 0 off E_.j
                psi_j = None
            p = pT.transpose(-1, -2)                                         # [B,H,n_q,n_k]
            Atil = A_sm + g * (cbar_j.unsqueeze(-2) * p - A_sm)             # ST site (2); forward == eq:program
            kstar = (p > 0).sum(-2)
            p2 = (p * p).sum(-2)
            has_rel = E.sum(-2) > 0
            p2_safe = torch.where(has_rel, p2.clamp_min(NU_EPS), torch.ones_like(p2))
            nu = torch.where(has_rel, 1.0 / p2_safe, torch.ones_like(p2))    # nu := 1 on an empty column; in [1,|E_.j|]
            if state is not None:
                state.nu_next = nu                                           # NOT detached (Sec. 5.3 [v4.1])
            # ---- STEP 1 fan-in: cap the WHOLE row at one unit.  NOT a softmax.   (eq:rowstep1)
            one = torch.ones(B, H, n_q, dtype=S32.dtype, device=S.device)
            excess, sm_mass = row_masses(Atil, A_sm, E)                      # decided from the relation, not the row total
            a1, cap_theta, cap_binds = apply_unit_cap(self, Atil, A_sm, E, vis, excess, sm_mass, Q=Q, col_size=E.sum(-2), p=p,
                                                      head_offset=head_offset)   # identity wherever the relation added no mass
            cbar_i = (g * a1).sum(-1)                                        # [B,H,n_q]   ST site (3); POST-cap quota
            Rtil = (Atil * E.to(S32.dtype)).sum(-1)
            # ---- STEP 2 fan-in: ceiling at the quota; TARGET is Atil (PRE-cap)   (eq:rowprog)
            if tau_i_override is not None:
                tau_i = torch.as_tensor(tau_i_override, dtype=S32.dtype, device=S.device).expand(B, H, n_q)
            elif self.tau_i_pinned:
                tau_i = one
            else:
                summ = self.tauQ.summary(Atil, E, vis, cbar_i, Rtil, col_size=E.sum(-2), p=p)
                tau_i = self.tauQ(_fp(Q), summ, head_offset)                 # [B,H,n_q]
            s_i = cbar_i / tau_i                                             # tau_i > tau_min > 0
            u, theta = proj_le_masked(Atil.masked_fill(~E, 0.0), s_i, E)     # <=, never proj_eq; u = 0 off E
            A = a1 + g * (tau_i.unsqueeze(-1) * u - a1)                      # ST site (4)
        diag = Diagnostics(E=E, logits=logits, cbar_j=cbar_j, cbar_i=cbar_i, tau_j=tau_j, tau_i=tau_i,
                           psi_j=psi_j, kstar=kstar, nu=nu, theta=theta, Rtil=Rtil,
                           supp_rel=((A > 0) & E).sum(-1), cap_binds=cap_binds, cap_theta=cap_theta, p=p, a1=a1,
                           Atil=Atil, A_sm=A_sm, c=c, u=u, A=A)
        if debug_enabled():
            from .invariants import check_invariants
            check_invariants(A, diag, vis)
        return A.to(out_dtype), diag

    forward = normalize

    # ------------------------------------------------------------------ helpers for Phase-B init
    def new_module_params(self):
        """(weights, biases_and_norms) for the optimiser groups G2 / G3 (training procedure Sec. 4)."""
        weights, others = [], []
        for name, p in self.named_parameters():
            leaf = name.split(".")[-1]
            is_bias_or_norm = leaf in ("bias", "b0", "b1", "b2", "tau_j_raw") or ".ln." in name or name.startswith("relation.b0")
            (others if is_bias_or_norm else weights).append(p)          # G3: biases, b0, LayerNorm gains (no weight decay)
        return weights, others


def debug_enabled() -> bool:
    return os.environ.get("MARSEA_DEBUG", "0") not in ("", "0", "false", "False")
