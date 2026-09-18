"""
Sec. 2 runtime invariants (v4.5 list, re-audited item by item) and SAN-1.  Asserted on every forward
of every patched layer when MARSEA_DEBUG=1; also used directly by the acceptance tests (T3, T6).

INV-1  sum_{i in E_.j} Atil[i,j] == cbar_j[j]       for every j with E_.j non-empty   (fp32 summation tol over n_q)
INV-2  Atil[i,j] == A_sm[i,j]                       for every (i,j) NOT in E                 exact
INV-3  A >= 0 and sum_j A[i,:] <= sum_j A_sm[i,:] + CAP_TOL + tol_cap + tol_sum   for every real row i   (fp64 sums)
INV-3a sum_j a1[i,:] <= sum_j A_sm[i,:] + CAP_TOL + tol_cap   (Stage 1's output alone: no Stage-2 fp32 allowance)
INV-4  A[i,j] == a1[i,j]                            for every j NOT in E_i.                  exact
INV-5  sum_j Atil[i,:] <= 1 (+1e-6)  =>  a1[i,:] == Atil[i,:]                               exact
INV-5b a binding row is projected DOWN onto its fp64 softmax mass: a1 <= Atil, |sum a1 - sum A_sm| <= tol_cap
INV-5c the cap's own dual is positive on every row that bound, and <= 0 on every over-CAP_TOL row it left alone
INV-6  Atil[i,j] == 0  =>  A[i,j] == 0                                                      exact
INV-7  sum_{j in E_i.} A[i,j] <= cbar_i[i] + tol    for every i with E_i. non-empty  (summation tol over n_k)
INV-8  ~vis[i,j] => E False and A_sm = Atil = a1 = A = 0 at (i,j)                           exact
INV-9  E all-False => A == A_sm to 1e-6             (elementwise: no summation scaling)
SAN-1  every real query row of A_sm sums to 1       (n_k eps / 2 + summation tol: a report about the KERNEL);
       fully-masked pad rows are exactly 0

Tolerances.  Two kinds of quantity are checked here and they carry different errors (review 9bacef9 E):
  * fp32 SUMS the mechanism itself produced (INV-1's relation column, INV-7's relation row, the chunked path's fp32
    `rowsum`): the summation tolerance tol_sum = 1e-5 sqrt(n_k / 64) (1e-12 in fp64), and for a whole-row fp32 total the
    kernel bound n_k eps / 2 on top (SAN-1's).
  * fp64 sums of fp32 ENTRIES taken by this module (INV-3a, INV-5b): the sums add no error, what remains is the fp32
    rounding of the entries -- the unit cap's projection lands on its fp64 target to the fp32 prefix scan's accuracy
    (measured <= 1.2e-7 on CPU, <= 9.4e-7 on CUDA at 8K), and the guard can leave a row inside its window (<= 2.0e-6 on
    CUDA at 16K) uncapped.  tol_cap = 8 log2(n_k) eps = 1.24e-5 at 8K, 1.34e-5 at 16K: ~10x above what is measured and
    ~10x below the fp32-era tol_sum these checks used to carry.
  INV-3 on the OUTPUT A adds Stage 2's fp32 allowance, tol_sum, on top: sum_E A <= cbar_i + err7 is INV-7's statement
  and its fp32 error reaches ~7e-6 at tau_i = 50 on a 10-key row (T6) -- the reviewer's fp64-level tightening of INV-3
  ignored that term (review 9bacef9 E).  The sensitivity it asked for lives in INV-3a, on Stage 1's output, where a
  1e-5 excess over the softmax mass now trips (INV-3 needed ~6e-4 at 16K; a +3e-4 corruption of A trips it now too).
"""
from __future__ import annotations
import math
import os
from typing import Optional
import torch

from .normalizer import CAP_TOL, EXCESS_CHUNK   # INV-3 and INV-5 are both statements about the unit cap's binding set

TOL = 1e-6


def debug_enabled() -> bool:
    return os.environ.get("MARSEA_DEBUG", "0") not in ("", "0", "false", "False")


class InvariantError(AssertionError):
    pass


def cap_tolerance(n_k: int, is64: bool = False) -> float:
    """what a row's fp64 total may exceed its exact-arithmetic value by, after the fp32 unit cap (module docstring).

    The default `8 eps log2(n_k)` is NOT a derived law -- the fp32 prefix scan's error is a device property, and a
    strictly sequential scan could reach n_k eps / 4 (36x this at 16K).  It is a number the guard window has to clear
    on the target device: preflight step 4b MEASURES the window and exits 3 if it does not.  The defined failure branch
    (review e6ca44f E) is to set MARSEA_TOL_CAP to 4 x the measured window and re-run the gate; the value in force is
    recorded by unit_cap_record in every README and eval table.  Either way the property is checked, never assumed."""
    eps = torch.finfo(torch.float64 if is64 else torch.float32).eps
    tol = 8.0 * eps * max(1.0, math.log2(max(2, n_k)))
    if TOL_CAP_OVERRIDE is not None and not is64:
        tol = max(tol, TOL_CAP_OVERRIDE)
    return tol


def _parse_tol_cap_override():
    """MARSEA_TOL_CAP, parsed ONCE at import: a malformed value used to raise inside an invariant check mid-run, with no
    mention of the variable (review 4f754ed 2.3)."""
    raw = os.environ.get("MARSEA_TOL_CAP", "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        raise ValueError(f"MARSEA_TOL_CAP={raw!r} is not a number (the INV-3 tolerance override; see RUNBOOK step 4b)") from None
    if not (v > 0):                                # NaN fails this comparison too
        raise ValueError(f"MARSEA_TOL_CAP={raw!r} must be a positive number")
    return v


TOL_CAP_OVERRIDE = _parse_tol_cap_override()


def tol_cap_ceiling(n_k: int) -> float:
    """n_k eps / 4: the largest fp32 prefix-scan error a strictly sequential reduction over n_k terms of total ~1 can
    make.  No real guard window is above it, so no override may be either."""
    return n_k * torch.finfo(torch.float32).eps / 4


def validate_tol_cap_override(L: int, window: Optional[float] = None) -> Optional[float]:
    """The override is bounded by the arithmetic and by the measurement, or it is refused (review 4f754ed 2): with
    max(default, override) and no ceiling, MARSEA_TOL_CAP=1 made INV-3a and INV-5b admit any row for a 30-hour queue --
    the cap_slack failure shape again, a number nothing physical bounds.  Called at startup with the run's length
    (unit_cap_record), and by check_unit_cap.py with the window it has just measured.
      * override > L eps / 4:      not a physically possible scan error at this length -> refused
      * override > 8 x window:     justified on a different device (a stale value after a VM reschedule) -> refused
    Returns the override (None if unset)."""
    v = TOL_CAP_OVERRIDE
    if v is None:
        return None
    ceil = tol_cap_ceiling(L)
    if v > ceil:
        raise ValueError(f"MARSEA_TOL_CAP={v:.3e} exceeds n_k eps / 4 = {ceil:.3e} at L = {L}: not a physically possible "
                         f"fp32 prefix-scan error.  The failure branch is 4 x the window check_unit_cap.py measured, not more.")
    if window is not None and v > 8 * window:
        raise ValueError(f"MARSEA_TOL_CAP={v:.3e} is more than 8 x the guard window measured on THIS device ({window:.3e}): "
                         f"the override was justified elsewhere -- re-measure (scripts/check_unit_cap.py) and set 4 x window.")
    return v


def _rowsum64(x: torch.Tensor) -> torch.Tensor:
    """sum over the last dim in fp64 WITHOUT an fp64 copy of the whole tensor: `sum(-1, dtype=float64)` on a [B,H,T,T]
    tensor materialises one on CUDA (1.0 GiB at 8K / head_block 2; `(x.double() - y.double()).sum(-1)` 3.0 GiB), so the
    debug report used to cost more than the mechanism it checks (review 9bacef9 E).  Key-chunked like row_masses."""
    out = torch.zeros(x.shape[:-1], dtype=torch.float64, device=x.device)
    for j0 in range(0, x.shape[-1], EXCESS_CHUNK):
        out += x[..., j0:j0 + EXCESS_CHUNK].sum(-1, dtype=torch.float64)
    return out


def _masked_excess64(Atil: torch.Tensor, A_sm: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """sum_{j in E_i.} (Atil - A_sm) in fp64, cast first, key-chunked: the MASKED form, which is not the formula the
    mechanism uses (row_masses differences two unmasked fp64 sums) -- so INV-5 recomputes the binding criterion
    independently, as it is meant to."""
    out = torch.zeros(Atil.shape[:-1], dtype=torch.float64, device=Atil.device)
    for j0 in range(0, Atil.shape[-1], EXCESS_CHUNK):
        J = slice(j0, j0 + EXCESS_CHUNK)
        out += ((Atil[..., J].double() - A_sm[..., J].double()) * E[..., J]).sum(-1)
    return out


@torch.no_grad()
def invariant_report(A: torch.Tensor, d, vis: torch.Tensor, tol: float = TOL) -> dict:
    """Returns {name: (ok: bool, worst: float)} for every invariant; never raises.  Under no_grad: the report builds
    nothing autograd should see (it used to hold a whole-row fp64 copy WITH graph -- review 9bacef9 E)."""
    is64 = d.Atil.dtype == torch.float64                                        # decide the tolerance BEFORE any cast
    cast = (lambda x: x.double()) if is64 else (lambda x: x.float())            # never narrow fp64 diagnostics to fp32
    A = cast(A)
    vis = vis.bool().expand_as(A)
    E, Atil, A_sm, a1 = d.E, cast(d.Atil), cast(d.A_sm), cast(d.a1)
    Ef = E.float()
    out = {}
    real_row = vis.any(-1)

    # INV-1: relation carries exactly cbar_j.  It is a SUM of up to n_q fp32 products, so its tolerance is
    # the summation one (round-2 ambiguity #30: fp32 re-ordered sums differ by up to 8e-6 on Atil):
    # 1e-5 absolute-or-relative at n_q <= 64, growing as sqrt(n_q / 64).  Exact in fp64 to 1e-12.
    # the summation tolerance scales with the number of terms SUMMED, which is not the same axis for every invariant:
    # INV-1 sums a column (n_q terms), INV-3 / INV-7 / SAN-1 sum a row (n_k).  One n_q-scaled tolerance for all of them
    # is both too tight wherever n_k > n_q -- a decode row, a logits_to_keep pass, n_q = 1 against n_k = 16384 -- and
    # ~10x too loose at square shapes, where it would swallow a real 1e-5 error (review 30372ae C).  INV-9 compares
    # elementwise and carries no summation at all.
    n_q, n_k = A.shape[-2], A.shape[-1]
    unit = 1e-12 if is64 else 1e-5
    tol_col = unit * max(1.0, (n_q / 64.0) ** 0.5)          # sums over queries
    tol_sum = unit * max(1.0, (n_k / 64.0) ** 0.5)          # fp32 sums over keys the MECHANISM produced
    tol_cap = cap_tolerance(n_k, is64)                      # fp64 sums of fp32 entries taken HERE (module docstring)
    skipped = []
    if d.cbar_j is None:
        # a decode row carries the ACCUMULATED column quota against a single query row, so the column identity is
        # not a statement about this tensor; recorded as absent rather than silently missing (review 30372ae C)
        skipped.append("INV-1 (no cbar_j: single-row emission)")
    else:
        ne = E.sum(-2) > 0
        relsum = (Atil * Ef).sum(-2)
        err1 = (relsum - cast(d.cbar_j)).abs() / cast(d.cbar_j).abs().clamp_min(1.0)
        err1 = err1[ne]
        out["INV-1"] = (bool((err1 <= tol_col).all()) if err1.numel() else True, float(err1.max()) if err1.numel() else 0.0)
    # INV-2: exact off the relation
    off = ~E
    diff2 = (Atil[off] != A_sm[off]).sum()
    out["INV-2"] = (int(diff2) == 0, float(diff2))
    # INV-3, stated against the row's OWN softmax mass and summed in fp64.  "sum_j A <= 1 + tol" cannot be checked
    # kernel-free: a non-binding row carries its fp32 softmax total, which the device's kernel puts up to n_k eps / 2
    # above one (review e16a843 B), and an fp32 check sum would add the same error again.  What the mechanism promises
    # is that a row never has MORE than the softmax gave it (plus CAP_TOL, the excess the cap leaves alone, plus the
    # guard window) unless the cap bound, in which case it was projected onto its softmax mass (to the projection's fp32
    # accuracy).  Both allowances are inside tol_cap.  No `max(1, .)`: since review 7814665 C the cap's target is the
    # row's fp64 softmax mass, not the constant 1.
    rs = _rowsum64(A)
    rs_sm = _rowsum64(A_sm)
    rs_a1 = _rowsum64(a1)
    bound = rs_sm + CAP_TOL + tol_cap + tol_sum          # + Stage 2's fp32 allowance on the relation part (INV-7's)
    over3 = (rs - bound)[real_row]
    ok3 = bool((A >= 0).all()) and (bool((over3 <= 0).all()) if over3.numel() else True)
    out["INV-3"] = (ok3, float(over3.max()) if over3.numel() else 0.0)
    # INV-3a: Stage 1's output alone.  a1 == Atil on a non-binding row (sum = softmax mass + excess <= CAP_TOL, or inside
    # the guard window), and a binding row was projected onto its fp64 softmax mass: no fp32 SUM of the mechanism enters,
    # so this is the tight statement -- a 1e-5 excess over the softmax mass trips it (review 9bacef9 E).
    over3a = (rs_a1 - (rs_sm + CAP_TOL + tol_cap))[real_row]
    out["INV-3a"] = (bool((over3a <= 0).all()) if over3a.numel() else True, float(over3a.max()) if over3a.numel() else 0.0)
    # INV-4
    diff4 = (A[off] != a1[off]).sum()
    out["INV-4"] = (int(diff4) == 0, float(diff4))
    # INV-5: unit cap identity on every row the relation did not push over one unit.  The criterion is recomputed here
    # INDEPENDENTLY of the mechanism's row_masses: the MASKED fp64 sum over the relation's entries (row_masses
    # differences two unmasked sums), key-chunked so debug mode holds no whole-row fp64 copy.
    excess = _masked_excess64(Atil, A_sm, E)
    sub = excess <= CAP_TOL
    diff5 = (a1[sub] != Atil[sub]).sum()
    out["INV-5"] = (int(diff5) == 0, float(diff5))
    # and the converse, INV-5b: where the cap DID bind, the row was projected DOWN onto its softmax mass -- never up
    # (a1 <= Atil, so no Stage-1 zero is lifted; review 7814665 C).  Emitted whether or not anything bound: an absent key
    # reads as a pass (the 30372ae vacuity trap), so "nothing bound" is (True, 0.0) with the count in the value.
    bnd = d.cap_binds.bool() & real_row if d.cap_binds is not None else (~sub) & real_row
    if bnd.any():
        e5b = (rs_a1[bnd] - rs_sm[bnd]).abs()
        lifted = int(((a1 > Atil) & bnd.unsqueeze(-1)).sum())
        out["INV-5b"] = (bool((e5b <= tol_cap).all()) and lifted == 0, float(e5b.max()) if lifted == 0 else float(lifted))
    else:
        out["INV-5b"] = (True, 0.0)
    # INV-5c: the cap's OWN dual (Diagnostics.cap_theta; `theta` is Stage 2's, which is non-negative by construction and
    # says nothing about the cap -- review 9bacef9 E).  Positive on every row that bound; on a row whose excess is over
    # CAP_TOL but which did NOT bind, the guard must be the reason (theta <= 0), so the recorded set is the decision.
    if getattr(d, "cap_theta", None) is not None and d.cap_binds is not None:
        th = cast(d.cap_theta)
        bad_pos = bnd & ~(th > 0)
        bad_neg = (~sub) & real_row & ~bnd & (th > 0)
        out["INV-5c"] = (int(bad_pos.sum()) + int(bad_neg.sum()) == 0, float(int(bad_pos.sum()) + int(bad_neg.sum())))
    else:
        skipped.append("INV-5c (no cap_theta recorded)")
    # INV-6: Stage-1 zero permanent
    z = Atil == 0
    bad6 = (A[z] != 0).sum()
    out["INV-6"] = (int(bad6) == 0, float(bad6))
    # INV-7: relation row mass <= cbar_i
    if d.cbar_i is None:
        skipped.append("INV-7 (no cbar_i)")
    else:
        ner = E.sum(-1) > 0
        rel_row = (A * Ef).sum(-1)
        err7 = (rel_row - cast(d.cbar_i))[ner]                               # a SUM of tau_i * u over E_i.
        out["INV-7"] = (bool((err7 <= tol_sum).all()) if err7.numel() else True, float(err7.max()) if err7.numel() else 0.0)
    # INV-8: masking total
    nv = ~vis
    bad8 = int(E[nv].sum()) + int((A_sm[nv] != 0).sum()) + int((Atil[nv] != 0).sum()) + int((a1[nv] != 0).sum()) + int((A[nv] != 0).sum())
    out["INV-8"] = (bad8 == 0, float(bad8))
    # INV-9: empty relation IS standard attention.  Elementwise, so no summation scaling.
    if not bool(E.any()):
        e9 = (A - A_sm).abs().max()
        out["INV-9"] = (bool(e9 <= tol), float(e9))
    # SAN-1.  The softmax's row sums: a statement about the DEVICE'S KERNEL, not about the mechanism (nothing in the
    # programs reads them since the relation's excess decides the cap).  The bound is the worst any lane-wise or scalar
    # reduction can do on a razor-edge row, n_k eps / 2, plus the summation tolerance: what it still catches is a
    # broken mask or a wrong dtype (O(1)), and the measured value is reported (review e16a843 B; check_unit_cap.py
    # measures the actual kernel).  Summed in fp64 so the check adds no error of its own.
    eps32 = torch.finfo(torch.float32).eps
    tol_san1 = tol_sum + (0.0 if is64 else n_k * eps32 / 2)
    okS = bool(((rs_sm[real_row] - 1).abs() <= tol_san1).all()) if real_row.any() else True
    okS = okS and bool((rs_sm[~real_row] == 0).all()) if (~real_row).any() else okS
    out["SAN-1"] = (okS, float((rs_sm[real_row] - 1).abs().max()) if real_row.any() else 0.0)
    # finiteness of everything present
    fin = torch.isfinite(A).all()
    for t in (d.tau_j, d.tau_i, d.nu):
        if torch.is_tensor(t):
            fin = fin and torch.isfinite(t).all()
    out["finite"] = (bool(fin), 0.0)
    if skipped:
        out["_not_checked"] = skipped
    return out


@torch.no_grad()
def invariant_report_sparse(d, vis: torch.Tensor, tol: float = TOL) -> dict:
    """The same list, checked on the CHUNKED path's sparse relation store.

    check_invariants used to be called from exactly one place -- normalizer._normalize_one -- and invariant_report's
    first statement reads d.Atil.dtype, so it raised AttributeError on a chunked Diagnostics rather than degrading.
    MARSEA_DEBUG=1 therefore did nothing at all on the path S2 trains (review d5bd980 F).

    INV-2 and INV-4 are structural here rather than checked: the chunked path writes A only on the relation, and off
    it Atil IS the A_sm entry it was built from.  Everything else is checked on the same quantities as the dense
    report, from the sparse store plus the row totals the path already accumulates."""
    sp = d.extra.get("sparse")
    if sp is None:
        return {}
    bhi, jj = sp["bhi"], sp["j"]
    Atil_s, a1_s, A_s = sp["Atil"], sp["a1"], sp["A"]
    B, H, n_q = d.cbar_i.shape
    n_k = d.cbar_j.shape[-1]
    dev = d.cbar_i.device
    is64 = d.cbar_i.dtype == torch.float64
    unit = 1e-12 if is64 else 1e-5
    eps32 = torch.finfo(torch.float32).eps
    tol_col = unit * max(1.0, (n_q / 64.0) ** 0.5)          # INV-1 sums a column
    tol_sum = unit * max(1.0, (n_k / 64.0) ** 0.5)          # INV-7 / INV-9 and the path's own fp32 row totals
    tol_cap = cap_tolerance(n_k, is64)                      # fp64 sums of the store's fp32 entries (module docstring)
    tol_kernel = tol_sum + (0.0 if is64 else n_k * eps32 / 2)   # an fp32 WHOLE-ROW total: the kernel's razor-edge bound
    out = {}
    n_rows, n_cols = B * H * n_q, B * H * n_k
    row_id = (bhi[:, 0] * H + bhi[:, 1]) * n_q + bhi[:, 2] if jj.numel() else torch.zeros(0, dtype=torch.long, device=dev)
    col_id = (bhi[:, 0] * H + bhi[:, 1]) * n_k + jj if jj.numel() else torch.zeros(0, dtype=torch.long, device=dev)
    z = lambda n: torch.zeros(n, dtype=d.cbar_i.dtype, device=dev)
    z64 = lambda: torch.zeros(n_rows, dtype=torch.float64, device=dev)
    # INV-1: the relation column carries exactly cbar_j
    relsum = z(n_cols).index_add(0, col_id, Atil_s.detach())
    cb = d.cbar_j.detach().reshape(-1)
    ne = torch.zeros(n_cols, dtype=torch.bool, device=dev).index_fill_(0, col_id, True) if jj.numel() else torch.zeros(n_cols, dtype=torch.bool, device=dev)
    err1 = ((relsum - cb).abs() / cb.abs().clamp_min(1.0))[ne]
    out["INV-1"] = (bool((err1 <= tol_col).all()) if err1.numel() else True, float(err1.max()) if err1.numel() else 0.0)
    nonneg = bool((A_s >= 0).all()) and bool((a1_s >= 0).all())
    sub_rows = ~d.cap_binds.detach().reshape(-1)
    real_rows = vis.bool().any(-1).reshape(B, -1, n_q).expand(B, H, n_q).reshape(-1)
    over3 = None
    if torch.is_tensor(d.extra.get("rowsum")):
        rowsum = d.extra["rowsum"].detach().reshape(-1).double()                 # the path's fp32 total of Atil, chunk by chunk
        ex_chk = z64().index_add(0, row_id, Atil_s.detach().double() - sp["A_sm"].detach().double())
        dA = z64().index_add(0, row_id, A_s.detach().double() - a1_s.detach().double())   # what Stage 2 changed on E
        # INV-3 on a non-binding row: sum_j A - sum_j A_sm = excess + sum_E (A - a1) exactly (a1 == Atil off E and
        # Atil == A_sm off E are structural here), so it is checked from the store's fp64 sums alone, with Stage 2's
        # fp32 allowance (tol_sum, INV-7's) and no kernel term.  It used to read `rowsum - sm_mass`, i.e. an fp32
        # whole-row total against an fp64 one, whose difference is the kernel's razor-edge error (1.2e-4 at 16K on an
        # 8-lane CPU, above the 1.14e-4 it allowed -- a latent false trip) (review 9bacef9 E).  Doubling A on the
        # relation still trips it (review e982f83 F-1's regression).
        over3 = (ex_chk + dA - CAP_TOL - tol_cap - tol_sum)[sub_rows & real_rows]
        # the stored sm_mass and rowsum protect each other: rowsum - sm_mass = excess in exact arithmetic, so the fp32
        # rowsum's kernel error is the only slack.  Corrupting extra["sm_mass"] used to trip nothing, because it was the
        # bound of INV-3 and on both sides of INV-3-recon (review 9bacef9 E).
        if torch.is_tensor(d.extra.get("sm_mass")):
            sm_mass = d.extra["sm_mass"].reshape(-1).double()
            e_mass = (rowsum - sm_mass - ex_chk).abs()[real_rows]
            out["INV-3-mass"] = (bool((e_mass <= tol_kernel).all()) if e_mass.numel() else True,
                                 float(e_mass.max()) if e_mass.numel() else 0.0)
        else:
            sm_mass = rowsum - ex_chk
        # PLUMBING, not numerics: ex_chk is the same expression on the same tensors as the path's own excess, so it is
        # bitwise the same number (review 7814665 E) -- what this catches is the recorded binding set being detached from
        # the store (the C-1 class).  Two-sided where the cap's own dual was recorded: a row over CAP_TOL that did not
        # bind must have been left alone BY THE GUARD (theta <= 0); without cap_theta the check is one-sided, because
        # such a row is legitimate (review 9bacef9 E).
        over_rows = ex_chk > CAP_TOL
        bad_plumb = ~sub_rows & ~over_rows                                        # bound without excess: never
        if getattr(d, "cap_theta", None) is not None:
            th = d.cap_theta.detach().reshape(-1)
            bad_plumb = bad_plumb | (~sub_rows & ~(th > 0)) | (over_rows & sub_rows & real_rows & (th > 0))
            out["INV-5c"] = (int((~sub_rows & ~(th > 0)).sum()) == 0, float((~sub_rows & ~(th > 0)).sum()))
        out["INV-5-plumbing"] = (int(bad_plumb.sum()) == 0, float(bad_plumb.sum()))
    ok3 = nonneg and (bool((over3 <= 0).all()) if (over3 is not None and over3.numel()) else True)
    out["INV-3"] = (ok3, float(over3.max()) if (over3 is not None and over3.numel()) else 0.0)
    ar = d.extra.get("A_rowsum")
    if torch.is_tensor(ar) and torch.is_tensor(d.extra.get("rowsum")):
        # a self-consistency check on the E8 reconstruction, not on the cap: A_rowsum is built from the fp32 `rowsum` on
        # non-binding rows, so it carries the kernel's whole-row error and is bounded with tol_kernel, not tol_cap
        rs = ar.reshape(-1)[real_rows].double()
        smr = sm_mass[real_rows]
        out["INV-3-recon"] = (bool((rs <= smr + CAP_TOL + tol_kernel).all()) if rs.numel() else True,
                              float((rs - smr).max()) if rs.numel() else 0.0)
    # INV-5: the unit cap is the identity on rows that did not exceed one unit
    if jj.numel():
        sub = ~d.cap_binds.detach().reshape(-1)[row_id]
        diff5 = (a1_s[sub] != Atil_s[sub]).sum()
        out["INV-5"] = (int(diff5) == 0, float(diff5))
        # INV-6: a Stage-1 zero is permanent
        zz = Atil_s == 0
        out["INV-6"] = (int((A_s[zz] != 0).sum()) == 0, float((A_s[zz] != 0).sum()))
        # INV-7: the relation row's mass never exceeds the quota
        relrow = z(n_rows).index_add(0, row_id, A_s.detach())
        ner = torch.zeros(n_rows, dtype=torch.bool, device=dev).index_fill_(0, row_id, True)
        err7 = (relrow - d.cbar_i.detach().reshape(-1))[ner]
        out["INV-7"] = (bool((err7 <= tol_sum).all()) if err7.numel() else True, float(err7.max()) if err7.numel() else 0.0)
        # INV-8: every relation pair is visible
        visb = vis.bool().expand(B, vis.shape[1], n_q, n_k)
        vsel = visb[bhi[:, 0], (bhi[:, 1] if visb.shape[1] > 1 else torch.zeros_like(bhi[:, 1])), bhi[:, 2], jj]
        out["INV-8"] = (bool(vsel.all()), float((~vsel).sum()))
    elif torch.is_tensor(ar):
        # INV-9: an empty relation IS standard attention, i.e. every real row still carries its full softmax mass
        real = vis.bool().any(-1).reshape(B, -1, n_q).expand(B, H, n_q)
        e9 = (ar[real] - 1).abs().max() if real.any() else torch.zeros((), device=dev)
        # a row TOTAL, i.e. a sum over n_k: tol_sum.  The dense INV-9 compares A with A_sm entry by entry and so uses
        # the elementwise tol; the two bounds differ because the quantities do (review e982f83 F-2), and the
        # `max(tol, tol_sum)` this read was tol_sum in every case anyway.
        out["INV-9"] = (bool(e9 <= tol_sum), float(e9))
    fin = all(bool(torch.isfinite(t).all()) for t in (d.tau_j, d.tau_i, d.nu, A_s, a1_s, Atil_s) if torch.is_tensor(t))
    out["finite"] = (fin, 0.0)
    # what the sparse store cannot see, recorded rather than silently absent (review 30372ae C)
    out["_not_checked"] = [
        "INV-2 (Atil == A_sm off E: the off-relation entries are never materialised)",
        "INV-4 (A == a1 off E_i.: the same)",
        "SAN-1 (A_sm row sums: only the relation's entries are stored)",
        "INV-3 on cap-BINDING rows (a1's row sum there is not recoverable from the relation alone)",
        "INV-5b (a binding row projected DOWN onto its softmax mass: the off-relation a1 is not stored)",
        "INV-8's off-vis zeros (only the relation pairs' visibility is checked)",
        "INV-9 checks the row TOTAL, not A == A_sm entry by entry",
    ] + ([] if getattr(d, "cap_theta", None) is not None else ["INV-5c (no cap_theta recorded)"])
    return out


def check_invariants(A: torch.Tensor, d, vis: torch.Tensor, tol: float = TOL, where: str = ""):
    rep = invariant_report(A, d, vis, tol) if getattr(d, "Atil", None) is not None else invariant_report_sparse(d, vis, tol)
    checked = {k: v for k, v in rep.items() if not k.startswith("_")}
    if not checked:
        # an empty report used to read as a pass, so MARSEA_DEBUG=1 was inert wherever the diagnostics were pruned
        # or the path was the decode one (review 30372ae C).  Silence is not evidence.
        raise InvariantError(f"MarSea invariant check was INERT {where}: nothing in this Diagnostics is checkable "
                             f"(Atil={getattr(d, 'Atil', None) is not None}, sparse={'sparse' in getattr(d, 'extra', {})})")
    bad = {k: v for k, v in checked.items() if not v[0]}
    if bad:
        raise InvariantError(f"MarSea invariant violation {where}: " + ", ".join(f"{k} (worst {v[1]:.3e})" for k, v in bad.items()))
    return rep
