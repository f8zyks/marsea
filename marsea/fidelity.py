r"""
Sec. 12: instrumentation, ground truth, aggregation.  Every fidelity quantity is computed ON THE
RELATION (def:fidelity as revised); off the relation the entries are the unmodified layer's.

Column (needs T_j, Sec. 12.3 / 12.4):
    Erel = E_.j;  Trel = T_j & Erel;  relation_recall = |Trel| / |T_j|   (a miss BEFORE any program; never charged to tau_j)
    m = |Trel|  (m == 0 => the column is excluded from every column average; count reported)
    delta = min_{Trel} s - max_{Erel \ Trel} s     (+inf if the relation holds no non-target -> lower endpoint 0)
    W = sum_{Trel} (s - min_{Trel} s);  lo = 1/(W + m delta) (SHARP; never 1/(m delta));  hi = 1/W (inf if W = 0, H2)
    hit = lo <= tau_j < hi;  signed distance to the nearer endpoint when not hit
    P/R BEFORE Stage 2 (supp p_.j) and AFTER (supp A_.j on Erel);  rho = target mass;  k*, nu, |Erel|, rel. width m delta / W
Row (needs K_i):
    P_i, R_i on E_i. (before/after Stage 2), relation recall |K_i & E_i.|/|K_i|, theta_i, tau_i, Rtil_i, cbar_i,
    tau_i Rtil_i / cbar_i, cap_binds, |E_i.|, |supp(A[i,E_i.])|, and the three-way distractor split (Sec. 12.3a).
Per (layer, head, step): coverage, histograms, variances, k* histogram, truncation flag (Sec. 12.4).
Dense arms: scored on TWO domains -- the full visible column and MarSea's paired relation (C-10).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Iterable, Optional, Sequence
import numpy as np
import torch

INF = float("inf")
HIST_BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 10 ** 9)]
HIST_LABELS = ["0", "1", "2", "3-4", "5-8", "9-16", "17+"]


def _hist(x: torch.Tensor) -> dict:
    x = x.flatten().long()
    n = max(1, x.numel())
    return {lab: float(((x >= lo) & (x <= hi)).sum().item() / n) for lab, (lo, hi) in zip(HIST_LABELS, HIST_BINS)}


def interval_from_scores(s: torch.Tensor, T_mask: torch.Tensor, E_mask: torch.Tensor):
    """s [n_q] fp32 column scores; T_mask, E_mask [n_q] bool.  Returns dict(m, delta, W, lo, hi, hi_inf, rel_recall,
    m_total).  Follows Sec. 12.3 [v4.3] and round-2 R-5."""
    Trel = T_mask & E_mask
    m_total = int(T_mask.sum())
    m = int(Trel.sum())
    out = dict(m=m, m_total=m_total, rel_recall=(m / m_total if m_total else float("nan")))
    if m == 0:
        out.update(delta=float("nan"), W=float("nan"), lo=float("nan"), hi=float("nan"), hi_inf=False)
        return out
    st = s[Trel]
    smin = float(st.min())
    W = float((st - smin).sum())
    comp = E_mask & ~T_mask
    if comp.any():
        delta = smin - float(s[comp].max())
    else:
        delta = INF
    lo = 0.0 if not math.isfinite(delta) else 1.0 / (W + m * delta) if (W + m * delta) > 0 else INF
    hi = (1.0 / W) if W > 0 else INF
    out.update(delta=delta, W=W, lo=lo, hi=hi, hi_inf=not math.isfinite(hi))
    return out


def _pr(supp: torch.Tensor, T: torch.Tensor):
    ns = int(supp.sum()); nt = int(T.sum()); inter = int((supp & T).sum())
    P = inter / ns if ns else 1.0
    R = inter / nt if nt else float("nan")
    return P, R


@torch.no_grad()
def column_fidelity(S_col: torch.Tensor, E_col: torch.Tensor, p_col: torch.Tensor, A_col: torch.Tensor,
                    tau_j: float, T_mask: torch.Tensor, kstar: Optional[int] = None, nu: Optional[float] = None,
                    psi_j: Optional[float] = None, K_ret: Optional[int] = None) -> dict:
    """All column quantities of Sec. 12.4 for one (l, h, j).  S_col [n_q] (-inf off vis), E_col/T_mask [n_q] bool,
    p_col = Stage-1 distribution on E_.j, A_col = final column."""
    S_col = S_col.float()
    iv = interval_from_scores(S_col, T_mask, E_col)
    Erel = E_col
    Trel = T_mask & Erel
    supp1 = (p_col > 0) & Erel
    supp2 = (A_col > 0) & Erel
    P1, R1 = _pr(supp1, Trel); P2, R2 = _pr(supp2, Trel)
    rho1 = float(p_col[Trel].sum()) if iv["m"] else float("nan")
    tot2 = float(A_col[Erel].sum())
    rho2 = float(A_col[Trel].sum()) / tot2 if (iv["m"] and tot2 > 0) else float("nan")
    hit = None; dist = None; excluded = None
    E_size = int(Erel.sum())
    # Thm. 2 is scored only where it says something (review 2026-09-09, A5/A6): a relation with >= 2 members (App. E
    # cost (a): a singleton is product-form and hits at EVERY tau), a positive margin (delta <= 0: no temperature
    # recovers -- reported through frac_columns_delta_pos, never as a miss), and a non-degenerate interval (delta = inf
    # with W = 0 is [0, inf): a hit at every tau).
    if iv["m"] == 0:
        excluded = "m_j = 0"
    elif E_size < 2:
        excluded = "|E_.j| < 2"
    elif not (iv["delta"] > 0):
        excluded = "delta <= 0"
    elif (not math.isfinite(iv["delta"])) and iv["W"] == 0:
        excluded = "degenerate interval [0, inf)"
    else:
        lo, hi = iv["lo"], iv["hi"]
        hit = bool(lo <= tau_j < hi)
        if not hit:
            dist = (tau_j - lo) if tau_j < lo else (tau_j - hi)          # signed: negative below, positive above
    width = (iv["m"] * iv["delta"] / iv["W"]) if (iv["m"] and iv["W"] and iv["W"] > 0 and math.isfinite(iv["delta"])) else float("nan")
    return dict(**iv, tau_j=float(tau_j), hit=hit, dist=dist, hit_excluded=excluded, delta_pos=(iv["m"] > 0 and (iv["delta"] > 0)),
                P_before=P1, R_before=R1, P_after=P2, R_after=R2, rho_before=rho1, rho_after=rho2,
                kstar=int(kstar) if kstar is not None else int(supp1.sum()), nu=nu, psi_j=psi_j,
                E_size=int(Erel.sum()), rel_width=width,
                # only meaningful when the hierarchical solve is DEPLOYED (K_ret is None for the exact flat solve)
                truncated=(kstar is not None and K_ret is not None and kstar >= 0.9 * K_ret))


@torch.no_grad()
def dense_column_precision(A_col: torch.Tensor, vis_col: torch.Tensor, T_mask: torch.Tensor,
                           E_paired: Optional[torch.Tensor]) -> dict:
    """For dense arms (B0/B1/B2): precision on the full visible column (m_j / n_vis, an identity) AND on MarSea's
    paired relation E_.j from the same example (m_j / |E_.j|, the paper's identity; the headline domain)."""
    supp = (A_col > 0) & vis_col
    P_full, R_full = _pr(supp, T_mask & vis_col)
    out = dict(P_full=P_full, R_full=R_full, n_vis=int(vis_col.sum()), m_full=int((T_mask & vis_col).sum()))
    if E_paired is not None:
        sr = supp & E_paired
        P_rel, R_rel = _pr(sr, T_mask & E_paired)
        out.update(P_rel=P_rel, R_rel=R_rel, E_size=int(E_paired.sum()), m_rel=int((T_mask & E_paired).sum()))
    return out


@torch.no_grad()
def row_fidelity(A_row: torch.Tensor, Atil_row: torch.Tensor, E_row: torch.Tensor, K_mask: torch.Tensor,
                 theta: float, tau_i: float, Rtil: float, cbar_i: float, cap_binds: bool,
                 S_col_argmax_is_row: Optional[torch.Tensor] = None) -> dict:
    """All row quantities of Sec. 12.4 / 12.3a for one (l, h, i).  K_mask = the row's gold source keys."""
    Erow = E_row
    Krel = K_mask & Erow
    supp1 = (Atil_row > 0) & Erow
    supp2 = (A_row > 0) & Erow
    P1, R1 = _pr(supp1, Krel); P2, R2 = _pr(supp2, Krel)
    nK = int(K_mask.sum())
    rel_recall = int(Krel.sum()) / nK if nK else float("nan")
    # three-way split of the RELATION's distractors (Sec. 12.3a); off-relation distractors are a 4th share
    dis = Erow & ~K_mask
    excl = dis & (Atil_row == 0)
    rej = dis & (Atil_row > 0) & (A_row == 0)
    surv = dis & (A_row > 0)
    nd = int(dis.sum())
    split = dict(excluded_by_stage1=int(excl.sum()) / nd if nd else float("nan"),
                 rejected_by_row=int(rej.sum()) / nd if nd else float("nan"),
                 surviving=int(surv.sum()) / nd if nd else float("nan"), n_distractors_rel=nd,
                 n_distractors_off=int((~Erow & ~K_mask & (A_row > 0)).sum()))
    if S_col_argmax_is_row is not None:                                   # the m_j = 1 explanation: argmax not this row
        split["excluded_argmax_elsewhere"] = int((excl & ~S_col_argmax_is_row).sum()) / max(1, int(excl.sum())) if int(excl.sum()) else float("nan")
    regime = (tau_i * Rtil / cbar_i) if cbar_i > 0 else float("nan")
    return dict(P_before=P1, R_before=R1, P_after=P2, R_after=R2, rel_recall=rel_recall, theta=theta, tau_i=tau_i,
                Rtil=Rtil, cbar_i=cbar_i, regime=regime, cap_binds=bool(cap_binds), E_size=int(Erow.sum()),
                supp_rel=int(supp2.sum()), **split)


def passage_support(A_row: torch.Tensor, E_row: torch.Tensor, spans: Sequence[tuple[int, int]], on_relation: bool = True):
    """A passage is selected iff exists a key j in its token span with A[i,j] > 0 (and E[i,j] if on_relation).
    Returns a bool list over spans (Sec. 12.2: the union over the passage's token keys)."""
    out = []
    for (a, b) in spans:
        seg = A_row[a:b] > 0
        if on_relation:
            seg = seg & E_row[a:b]
        out.append(bool(seg.any()))
    return out


def set_prf(pred: Iterable, gold: Iterable):
    pred, gold = set(pred), set(gold)
    inter = len(pred & gold)
    P = inter / len(pred) if pred else (1.0 if not gold else 0.0)
    R = inter / len(gold) if gold else 1.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F, (pred == gold)


def coverage_residual(n_q: int, m_js: Iterable[int]) -> int:
    """Sec. 12 / App. H: n_q - sum_j m_j; positive when queries go ungoverned, negative when claimed by several keys."""
    return int(n_q - sum(m_js))


@torch.no_grad()
def e8_summary_from_diag(d, vis: torch.Tensor, K_ret: Optional[int] = None) -> dict:
    """Sec. 12.4 / training Sec. 8.1 per-layer block, computed from a Diagnostics on [B,H,n_q,n_k] tensors (all Q-heads
    kept separate: lists indexed by head, batch pooled).  Cheap enough to run every 50 steps at 8K."""
    E = d.E.bool(); vis = vis.bool().expand_as(E)
    B, H, n_q, n_k = E.shape
    col_has = vis.any(-2); row_has = vis.any(-1)                                   # [B,H,n_k], [B,H,n_q]
    Ecol = E.sum(-2); Erow = E.sum(-1)
    active = (Ecol > 0) & col_has
    rel_rows = (Erow > 0) & row_has
    def per_head(x, m, fn):
        out = []
        for h in range(H):
            xs = x[:, h][m[:, h]]
            out.append(float(fn(xs)) if xs.numel() else float("nan"))
        return out
    def hist(x, m):
        out = []
        for h in range(H):
            xs = x[:, h][m[:, h]].long()
            n = max(1, xs.numel())
            out.append({lab: float(((xs >= lo) & (xs <= hi)).sum().item() / n) for lab, (lo, hi) in zip(HIST_LABELS, HIST_BINS)})
        return out
    rho = float((E & vis).sum() / vis.sum().clamp_min(1))                    # pooled (kept for the trainer's one-liner)
    rho_head = [float((E[:, h] & vis[:, h]).sum() / vis[:, h].sum().clamp_min(1)) for h in range(H)]   # never pool heads
    # rows ending with ZERO total mass (D-31 diagnostic): a real row whose only visible key re-shaped its quota away.
    # The SAME definition as the sparse summary -- `<= _zero_mass_tol` with the ambiguous band reported beside it.  This
    # was an exact `== 0` while the chunked path used the tolerance, so a row with mass in (0, 2e-5] was zero-mass on one
    # path and not on the other under one E8 key (review e982f83 E).
    A = d.A if d.A is not None else None
    zero_rows = zero_tol = ambiguous = None
    if A is not None:
        ar = A.sum(-1)
        zero_tol = _zero_mass_tol(ar, n_k)
        zero_rows = per_head((ar <= zero_tol).float(), row_has, torch.mean)
        ambiguous = per_head(((ar > zero_tol) & (ar <= 1e-3)).float(), row_has, torch.mean)
    # the fan-in regime statistic tau_i Rtil_i / cbar_i about 1 (Sec. 2.2): distribution over rows with a quota
    quota_rows = rel_rows & (d.cbar_i > 1e-9)
    regime = d.tau_i * d.Rtil / torch.where(d.cbar_i > 1e-9, d.cbar_i, torch.ones_like(d.cbar_i))
    regime_q = []
    for h in range(H):
        xs = regime[:, h][quota_rows[:, h]]
        regime_q.append([float(q) for q in torch.quantile(xs.float(), torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32, device=xs.device))] if xs.numel() > 1 else [float("nan")] * 3)
    res = dict(rho=rho, rho_head=rho_head, frac_rows_zero_mass=zero_rows, regime_q10_50_90=regime_q,
               zero_mass_tol=(float(zero_tol) if zero_tol is not None else None),
               frac_rows_zero_mass_ambiguous=ambiguous,
               rho_col=per_head(active.float(), col_has, torch.mean),
               rho_row=per_head(rel_rows.float(), row_has, torch.mean),
               hist_Ecol=hist(Ecol, active), hist_Erow=hist(Erow, rel_rows),
               frac_Ecol_singleton=per_head((Ecol == 1).float(), active, torch.mean),
               var_tau_j=per_head(d.tau_j, active, lambda x: x.var() if x.numel() > 1 else torch.zeros(())),
               var_tau_i=per_head(d.tau_i, rel_rows, lambda x: x.var() if x.numel() > 1 else torch.zeros(())),
               tau_j_median=per_head(d.tau_j, active, torch.median), tau_i_median=per_head(d.tau_i, rel_rows, torch.median),
               hist_kstar=hist(d.kstar, active),
               frac_rows_over_unit=per_head(d.cap_binds.float(), row_has, torch.mean),
               frac_cbar_i_zero=per_head((d.cbar_i <= 1e-9).float(), rel_rows, torch.mean),
               truncation_flag=(bool(((d.kstar >= 0.9 * K_ret) & active).any()) if K_ret is not None else None),
               row_trigger_rate=per_head(rel_rows.float(), row_has, torch.mean),
               col_induced_rate=per_head(E.any(-1).float(), row_has, torch.mean),
               source="dense")                                              # the sparse summary tags itself too (collect.py)
    res["row_trigger_violation"] = any((a < b - 1e-9) for a, b in zip(res["row_trigger_rate"], res["col_induced_rate"]) if a == a and b == b)
    res["kstar_by_Ecol"] = []
    for h in range(H):
        row = {}
        for lab, (lo, hi) in zip(HIST_LABELS, HIST_BINS):
            sel = (Ecol[:, h] >= lo) & (Ecol[:, h] <= hi) & active[:, h]
            row[lab] = float(d.kstar[:, h][sel].float().mean()) if sel.any() else float("nan")
        res["kstar_by_Ecol"].append(row)
    return res


def _zero_mass_tol(ar, n_k: int) -> float:
    """The threshold below which a reconstructed row total counts as zero mass.

    `A_rowsum` is rebuilt as clamp(rowsum,1) - sum_E a1 + sum_E A.  A genuinely zero-mass row is the maximal-
    cancellation case, so in fp32 it lands at +-3.6e-7 rather than 0 and an exact `<= 0` decides at random
    (review 30372ae B-3).  sqrt(n_k) * eps is the accumulated rounding of a row sum; four of those separates a
    zero row from any row carrying real mass (which is O(1), and never below 1e-3 in practice).  The sqrt form is
    generous -- ~170x the measured residual at 16K (review e982f83 E) -- and harmless, because the gap it has to fall
    into is four orders of magnitude wide.  BOTH E8 summaries use it, so the key means one thing on both paths."""
    import math
    return 4.0 * torch.finfo(ar.dtype).eps * max(1.0, math.sqrt(max(1, n_k)))


@torch.no_grad()
def e8_summary_from_sparse(diag, vis: torch.Tensor, K_ret: Optional[int] = None) -> dict:
    """The Sec. 12.4 / training-procedure Sec. 8.1 block computed from the CHUNKED path's sparse relation, which never
    materialises E.  Without this the chunked path -- the path S2 trains on -- produces no E8 at all, which is the same
    loss the read-through's F-2 found on the dense path: no coverage, no |E_.j| histogram, no k* histogram, and so none
    of the three pre-registered degeneracy readings."""
    sp = diag.extra.get("sparse")
    if sp is None:
        return {}
    bhi, jj = sp["bhi"], sp["j"]
    B, H, n_q = diag.cbar_i.shape
    n_k = diag.cbar_j.shape[-1]
    vis = vis.bool()
    visf = vis.expand(B, H, n_q, n_k) if vis.shape[1] == 1 else vis
    n_vis = int(visf.sum())
    row_sizes = diag.extra.get("E_row_sizes")                       # |E_i.| per row, already counted by the chunked path
    # |E_.j| per (b, h, j) from the sparse column index
    flat_col = (bhi[:, 0] * H + bhi[:, 1]) * n_k + jj
    col_sizes = torch.bincount(flat_col, minlength=B * H * n_k).reshape(B, H, n_k)
    col_has = visf.any(-2); row_has = visf.any(-1)
    active = (col_sizes > 0) & col_has
    rel_rows = ((row_sizes > 0) & row_has) if row_sizes is not None else row_has
    # the same set counted from the COLUMN side: row_trigger_rate comes from the chunked path's own per-row counts,
    # col_induced_rate from the sparse column index, so on this path the two are genuinely independent computations
    # and their equality is a real cross-check of INV-5's one-object rule (on the dense path both read E, so the
    # comparison there is a tautology).
    rows_from_cols = torch.zeros(B * H * n_q, dtype=torch.bool, device=col_sizes.device)
    if bhi.numel():
        rows_from_cols[(bhi[:, 0] * H + bhi[:, 1]) * n_q + bhi[:, 2]] = True
    rows_from_cols = rows_from_cols.reshape(B, H, n_q) & row_has
    def per_head(x, m, fn):
        out = []
        for h in range(H):
            xs = x[:, h][m[:, h]]
            out.append(float(fn(xs)) if xs.numel() else float("nan"))
        return out
    def hist(x, m):
        out = []
        for h in range(H):
            xs = x[:, h][m[:, h]].long()
            n = max(1, xs.numel())
            out.append({lab: float(((xs >= lo) & (xs <= hi)).sum().item() / n) for lab, (lo, hi) in zip(HIST_LABELS, HIST_BINS)})
        return out
    rho_head = []
    for h in range(H):
        nv = int(visf[:, h].sum())
        rho_head.append(float(int((bhi[:, 1] == h).sum()) / max(1, nv)))
    res = dict(rho=float(bhi.shape[0] / max(1, n_vis)), rho_head=rho_head,
               rho_col=per_head(active.float(), col_has, torch.mean),
               rho_row=per_head(rel_rows.float(), row_has, torch.mean),
               hist_Ecol=hist(col_sizes, active),
               hist_Erow=hist(row_sizes, rel_rows) if row_sizes is not None else None,
               frac_Ecol_singleton=per_head((col_sizes == 1).float(), active, torch.mean),
               var_tau_j=per_head(diag.tau_j, active, lambda x: x.var() if x.numel() > 1 else torch.zeros(())),
               var_tau_i=per_head(diag.tau_i, rel_rows, lambda x: x.var() if x.numel() > 1 else torch.zeros(())),
               tau_j_median=per_head(diag.tau_j, active, torch.median),
               tau_i_median=per_head(diag.tau_i, rel_rows, torch.median),
               hist_kstar=hist(diag.kstar, active),
               frac_rows_over_unit=per_head(diag.cap_binds.float(), row_has, torch.mean),   # same definition as dense
               pass2_rebuilt_row_frac=diag.extra.get("pass2_rebuilt_row_frac"),              # memory diagnostic
               frac_cbar_i_zero=per_head((diag.cbar_i <= 1e-9).float(), rel_rows, torch.mean),
               truncation_flag=(bool(((diag.kstar >= 0.9 * K_ret) & active).any()) if K_ret is not None else None),
               # the five keys the first sparse summary had no counterpart for.  Missing them meant the D-31 zero-mass
               # diagnostic and the k*-by-column-size table were absent for the whole of S2, which trains chunked.
               frac_rows_zero_mass=(per_head((diag.extra["A_rowsum"] <= _zero_mass_tol(diag.extra["A_rowsum"], n_k)).float(),
                                             row_has, torch.mean)
                                    if torch.is_tensor(diag.extra.get("A_rowsum")) else None),
               row_trigger_rate=per_head(rel_rows.float(), row_has, torch.mean),
               col_induced_rate=per_head(rows_from_cols.float(), row_has, torch.mean),
               source="sparse (chunked path)")
    quota_rows = rel_rows & (diag.cbar_i > 1e-9)
    regime = diag.tau_i * diag.Rtil / torch.where(diag.cbar_i > 1e-9, diag.cbar_i, torch.ones_like(diag.cbar_i))
    if torch.is_tensor(diag.extra.get("A_rowsum")):
        ar = diag.extra["A_rowsum"]
        t0 = _zero_mass_tol(ar, n_k)
        res["zero_mass_tol"] = float(t0)
        # how many rows sit between the tolerance and a mass a reader would call real.  The dense path decides this
        # with an exact `== 0`; the sparse path decides it on a RECONSTRUCTION whose fp32 residual is ~1e-7 at
        # maximal cancellation -- which is exactly the zero-mass case -- so the band is reported rather than hidden
        # (review 30372ae B-3).
        res["frac_rows_zero_mass_ambiguous"] = per_head(((ar > t0) & (ar <= 1e-3)).float(), row_has, torch.mean)
    res["_nnz"] = int(bhi.shape[0]); res["_n_vis"] = int(n_vis)      # for merge_e8_summaries across head blocks
    res["_pass2_rows"] = int(diag.extra.get("pass2_rows", 0) or 0)   # the weight of pass2_rebuilt_row_frac
    res["regime_q10_50_90"] = [
        [float(q) for q in torch.quantile(regime[:, h][quota_rows[:, h]].float(),
                                          torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32, device=regime.device))]
        if quota_rows[:, h].sum() > 1 else [float("nan")] * 3 for h in range(H)]
    res["row_trigger_violation"] = any((a < b - 1e-9) for a, b in zip(res["row_trigger_rate"], res["col_induced_rate"])
                                       if a == a and b == b)
    res["kstar_by_Ecol"] = []
    for h in range(H):
        row = {}
        for lab, (lo, hi) in zip(HIST_LABELS, HIST_BINS):
            sel = (col_sizes[:, h] >= lo) & (col_sizes[:, h] <= hi) & active[:, h]
            row[lab] = float(diag.kstar[:, h][sel].float().mean()) if sel.any() else float("nan")
        res["kstar_by_Ecol"].append(row)
    return res


def merge_e8_summaries(parts: list) -> dict:
    """Merge per-head-block E8 summaries into one.  Every per-head quantity is a list indexed by head, so the blocks
    concatenate; the two pooled scalars (rho) are recomputed from the carried counts, and the flags are ORed.

    The chunked path computes E8 per head block, BEFORE the dense rebuild narrows the per-head tensors to the kept
    head -- otherwise the summary reads a one-head cbar_i against an all-head sparse store and either crashes or
    reports one head's numbers as the layer's (review 30372ae E)."""
    parts = [p for p in parts if p]
    if not parts:
        return {}
    out = {}
    keys = list(dict.fromkeys(k for p in parts for k in p))
    for k in keys:
        vals = [p.get(k) for p in parts]
        # the TYPE comes from a PRESENT value: with block 0 at None and a later block a list, this merged to True
        first = next((v for v in vals if v is not None), None)
        if k in ("_nnz", "_n_vis", "_pass2_rows"):
            out[k] = sum(int(v or 0) for v in vals)
        elif k == "rho":
            continue                                                    # recomputed below from the counts
        elif k == "pass2_rebuilt_row_frac":
            # COUNT-weighted, as _chunked_by_head_block computes it: an unweighted mean is wrong whenever H % hb != 0
            # (H = 6, hb = 4: 0.2865 against 0.2951 -- review e982f83 I)
            w = [int(p.get("_pass2_rows", 0) or 0) for p in parts]
            if sum(w):
                out[k] = float(sum(float(v or 0.0) * wi for v, wi in zip(vals, w)) / sum(w))
            else:
                out[k] = None if first is None else float(sum(float(v) for v in vals if v is not None)
                                                          / max(1, sum(v is not None for v in vals)))
        elif k == "zero_mass_tol":
            out[k] = max((float(v) for v in vals if v is not None), default=None)
        elif isinstance(first, list):
            # a per-head list with a block missing cannot be indexed by head any more: say so rather than shift it
            out[k] = None if any(v is None for v in vals) else [x for v in vals for x in v]
        elif first is None or isinstance(first, bool):
            out[k] = any(bool(v) for v in vals) if first is not None else None
        elif isinstance(first, (int, float)):
            out[k] = float(sum(float(v) for v in vals if v is not None) / max(1, sum(v is not None for v in vals)))
        else:
            out[k] = first
    out["rho"] = out["_nnz"] / max(1, out["_n_vis"])
    return out
