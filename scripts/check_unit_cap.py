"""Check the unit cap on THIS device, and record the softmax kernel's row-sum error (review e16a843 B/C).

The unit cap `proj_le(Atil, 1)` binds iff the RELATION pushed the row over its softmax mass:
    excess_i = sum_{j in E_i.} (Atil_ij - A_sm_ij) > CAP_TOL          (normalizer.row_masses, fp64)
and never on the fp32 row total, whose error is a property of the softmax KERNEL: pairwise on CUDA (~1e-6), lane-wise
on x86 where a razor-edge row -- one spike, a flat tail just under half an ulp of it -- loses n_k eps / (2 L) of its mass
(6e-5 at 8K on an 8-lane CPU; 1e-3 for a scalar kernel).  Three rounds of slack constants were each right on one kernel.
A binding row is projected onto its own fp64 softmax mass, and the projection refuses to bind when ITS fp32 arithmetic
finds nothing above the target (review 7814665 C) -- that guard has a WINDOW, the fp32 prefix scan's error, which is a
device property too (review 9bacef9 D).

This script does three things on the device it runs on:
  1. MEASURES the kernel's row-sum error, with the razor-edge search and random rows, for torch.softmax (the dense and
     decode paths' row_softmax) and the chunked path's about-max form.  Reported, and checked against the bound SAN-1
     uses (n_k eps / 2 + the summation tolerance) -- which is what keeps SAN-1 from firing in production.
  2. MEASURES the theta guard's window by bisection: the largest relation excess a forced bind leaves alone (theta <= 0
     in fp32).  1.3e-7 on this CPU (torch's cumsum accumulates in higher precision), 1.1e-6 at 8K and 2.0e-6 at 16K on
     the dev GPU -- ABOVE CAP_TOL -- and up to n_k eps / 4 for a sequential fp32 scan.  Gated against the tolerance INV-3
     allows a row (invariants.cap_tolerance): a row inside the window keeps less than the window above its softmax mass.
  3. GATES the mechanism: through the real MarSeaNormalizer, on razor-edge rows at every length, the cap must be the
     identity bitwise with the relation forced empty (dense, chunked and decode); on a live relation at rho 0.05 and 0.5
     a binding row must be projected DOWN (a1 <= Atil, the cap's own theta > 0, no Stage-1 zero lifted, sum a1 within tol
     of the row's softmax mass), on dense, chunked AND decode; a row over CAP_TOL that did not bind must have been left
     alone by the guard; the recorded excess must equal the masked fp64 reference; dense and chunked must agree on every
     row whose excess is not within their disagreement (excess differs by <= 4e-8 between the two softmax formulas --
     the 6e-6 figure earlier rounds quoted was the SM_MASS difference, review 9bacef9 B-2 -- plus the guard window) of
     CAP_TOL; a relation adding 4 max(CAP_TOL, window) to a row must bind on every path and 5e-7 must never.
Exit 3 on any failure.  The record goes to --out with the device name.

  python scripts/check_unit_cap.py --lengths 2048 8192 16384 --out runs/unit_cap_device.json
"""
import argparse, json, os, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch


def razor_rows(n: int, gaps, dev, jitter: float = 0.0, g=None) -> torch.Tensor:
    S = torch.zeros(len(gaps), n, device=dev)
    if jitter:
        S = S + torch.randn(S.shape, generator=g, device=dev) * jitter
    S[:, 0] = torch.tensor(gaps, device=dev)
    return S


def random_rows(shape: str, R: int, n: int, scale: float, g, dev) -> torch.Tensor:
    x = torch.randn(R, n, generator=g, device=dev)
    if shape == "gauss":
        return x * scale
    if shape == "heavy":
        u = torch.rand(R, n, generator=g, device=dev).clamp_min(1e-6)
        return x * scale / u.sqrt().clamp_min(0.05) * 0.25
    x = x * scale; x[torch.arange(R, device=dev), torch.randint(0, n, (R,), generator=g, device=dev)] += 2.0 * scale
    return x


def chunked_about_max(S: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    from marsea.normalizer import softmax_about_max
    m = S.amax(-1); acc = torch.zeros_like(m)
    for j0 in range(0, S.shape[-1], chunk):
        acc = acc + torch.exp(S[..., j0:j0 + chunk] - m.unsqueeze(-1)).sum(-1)
    return softmax_about_max(S, m, torch.log(acc))


def measure_kernel(n: int, gaps, g, dev, elements: float) -> dict:
    from marsea.normalizer import row_softmax
    worst = {"torch.softmax": (0.0, None), "about-max (chunked)": (0.0, None)}
    def score(S, at):
        for kind, A in (("torch.softmax", row_softmax(S, torch.ones_like(S, dtype=torch.bool))),
                        ("about-max (chunked)", chunked_about_max(S))):
            e = (A.sum(-1, dtype=torch.float64) - 1).abs()
            k = int(e.argmax()); v = float(e[k])
            if v > worst[kind][0]:
                worst[kind] = (v, at)
    for jitter in (0.0, 0.01, 0.3):
        score(razor_rows(n, gaps, dev, jitter, g), dict(shape="razor", jitter=jitter))
    for shape in ("gauss", "heavy", "peaked"):
        for sc in (0.25, 0.5, 1, 2, 4, 8, 16):
            R = max(1, int(elements // n))
            for s0 in range(0, R, max(1, (1 << 26) // n)):
                score(random_rows(shape, min(R - s0, max(1, (1 << 26) // n)), n, sc, g, dev), dict(shape=shape, scale=sc))
    return {k: dict(error=v[0], at=v[1]) for k, v in worst.items()}


@torch.no_grad()
def measure_guard_window(n: int, gaps, dev, iters: int = 60) -> dict:
    """the largest relation excess a FORCED bind leaves alone (theta <= 0 in the projection's fp32 arithmetic), on
    razor rows whose fp32 total sits on either side of one; bisected per row in log space, the maximum reported."""
    from marsea.normalizer import row_softmax
    from marsea.primitives import proj_le_masked
    S = razor_rows(n, gaps, dev)
    Asm = row_softmax(S, torch.ones_like(S, dtype=torch.bool))
    sm = Asm.sum(-1, dtype=torch.float64)
    vis = torch.ones_like(Asm, dtype=torch.bool)
    R = len(gaps)
    lo = torch.full((R,), 1e-10, dtype=torch.float64, device=dev)
    hi = torch.full((R,), 1e-2, dtype=torch.float64, device=dev)
    ones = torch.ones(R, dtype=torch.bool, device=dev)
    def suppressed(e):
        Atil = Asm.clone(); Atil[:, 1:9] += (e / 8).float().unsqueeze(-1)
        _, th = proj_le_masked(Atil, sm.float(), vis, binds=ones)
        return th <= 0
    for _ in range(iters):
        mid = (lo * hi).sqrt()
        sup = suppressed(mid)
        lo = torch.where(sup, mid, lo); hi = torch.where(sup, hi, mid)
    still = int(suppressed(hi).sum())                    # nothing may be suppressed at the top of the bracket
    return dict(window=float(lo.max()), window_over_cap_tol=float(lo.max()) / 1e-6, rows=R, unresolved_rows=still,
                kernel_total_minus_one=[float((sm - 1).min()), float((sm - 1).max())])


def _decode_live(norm, S, vis, K, Q, n, tol_mass, force_empty: bool, prior_quota: float = 5e-8) -> dict:
    """decode_step on every razor row with the calibrated relation (or forced empty): the cap's Sec. C properties.

    The cache starts with a column quota of `prior_quota` on every key, as if earlier queries had put it there: with an
    EMPTY cache cbar_j is exactly this row's own A_sm on E, Atil == A_sm on the relation and the cap can never bind, so
    the earlier "decode ok" was vacuous (0 binding rows).  With ~|E_i.| x 5e-8 of excess the razor rows bind."""
    from marsea.causal import FrozenPrefixCache, decode_step, NEG_PAD
    n_q = S.shape[-2]
    binds = adds = lifted = 0; worst_mass = 0.0; theta_min = float("inf")
    for i in range(n_q):
        cache = FrozenPrefixCache(K_ret=8)
        cache.tau_j = torch.ones(1, 1, n, device=S.device)
        cache.cbar_j = torch.full((1, 1, n), 0.0 if force_empty else prior_quota, device=S.device)
        cache.nu = torch.ones(1, 1, n, device=S.device)
        cache.rel_scores = torch.full((1, 1, n, 8), NEG_PAD, device=S.device); cache.rel_valid = torch.zeros(1, 1, n, 8, dtype=torch.bool, device=S.device)
        cache.rel_qidx = torch.full((1, 1, n, 8), -1, dtype=torch.long, device=S.device); cache.n_seen = n
        lr = torch.full((1, 1, 1, n), -1e4, device=S.device) if force_empty else None
        _, info = decode_step(norm, cache, S[:, :, i:i + 1], vis[:, :, i:i + 1], K, Q[:, :, i:i + 1], logits_row=lr)
        b = info["cap_binds"]
        binds += int(b.sum())
        adds += int((info["a1"] > info["Atil"]).sum())
        lifted += int(((info["Atil"] == 0) & (info["a1"] > 0)).sum())
        if b.any():
            worst_mass = max(worst_mass, float((info["a1"].sum(-1, dtype=torch.float64) - info["sm_mass"])[b].abs().max()))
            theta_min = min(theta_min, float(info["cap_theta"][b].min()))
    return dict(binding_rows=binds, a1_above_Atil=adds, zeros_lifted=lifted, worst_mass_err=worst_mass,
                min_cap_theta_on_binding=(theta_min if binds else None),
                ok=(adds == 0 and lifted == 0 and worst_mass <= tol_mass
                    and ((binds == 0) if force_empty else (binds > 0 and theta_min > 0))))     # live: NOT vacuous


def gate_mechanism(n: int, gaps, dev, rho: float, window: float) -> dict:
    """the real normaliser on razor-edge rows: forced empty (dense, chunked, decode) and live at `rho` (all three)."""
    from marsea.normalizer import MarSeaNormalizer, State, row_masses, unit_cap, CAP_TOL
    from marsea.invariants import cap_tolerance
    from marsea.chunked import marsea_chunked_attention
    torch.manual_seed(0)
    D = 16
    norm = MarSeaNormalizer(D, 8).to(dev)
    n_q = len(gaps)
    K = torch.zeros(1, 1, n, D, device=dev); K[0, 0, 0, 0] = 1.0
    K[0, 0, 1:, 1:] = torch.randn(n - 1, D - 1, device=dev) * 1e-3
    Q = torch.zeros(1, 1, n_q, D, device=dev); Q[0, 0, :, 0] = torch.tensor(gaps, device=dev) * D ** 0.5
    vis = torch.ones(1, 1, n_q, n, dtype=torch.bool, device=dev)
    tol_mass = cap_tolerance(n)
    out = dict(rho=rho, tol_mass=tol_mass)
    with torch.no_grad():
        S = torch.matmul(Q, K.transpose(-1, -2)) * D ** -0.5
        A, d = norm.normalize(S, vis, K, Q, State(), logits_override=torch.full((1, 1, 1, 1), -1e4, device=dev))
        out["dense_forced_empty_binds"] = int(d.cap_binds.sum()); out["dense_forced_empty_identity"] = bool(torch.equal(A, d.A_sm))
        _, dc = marsea_chunked_attention(norm, Q, K, K, vis, State(), chunk=1024, force_empty=True)
        out["chunked_forced_empty_binds"] = int(dc.cap_binds.sum())
        out["decode_forced_empty_binds"] = _decode_live(norm, S, vis, K, Q, n, tol_mass, force_empty=True)["binding_rows"]
        norm.relation.calibrate_b0(K, Q, vis, rho)
        A, d = norm.normalize(S, vis, K, Q, State())
        # the masked fp64 reference: the formula the mechanism does NOT use (row_masses differences two unmasked sums)
        ex = torch.zeros_like(d.cap_binds, dtype=torch.float64)
        for j0 in range(0, n, 1024):
            J = slice(j0, j0 + 1024)
            ex += ((d.Atil[..., J].double() - d.A_sm[..., J].double()) * d.E[..., J]).sum(-1)
        ex_mech, sm_mech = row_masses(d.Atil, d.A_sm, d.E)
        out["excess_exact_vs_reference"] = float((ex_mech - ex).abs().max())      # fp64 regrouping only (D-1)
        out["min_abs_excess_minus_cap_tol"] = float((ex - CAP_TOL).abs().min())    # the real safety margin, recorded
        b = d.cap_binds; th = d.cap_theta
        over = ex > CAP_TOL
        # the recorded binding set is the excess rule minus the rows the projection's own arithmetic left alone --
        # and those rows must show it (theta <= 0), so the decision is exactly the excess rule plus the guard
        out["guard_suppressed_rows"] = int((over & ~b).sum())
        out["dense_live_decision_is_excess"] = bool((b <= over).all() and (th[over & ~b] <= 0).all() and torch.equal(d.a1[~b], d.Atil[~b]))
        out["dense_live_identity_on_nonbinding"] = bool(torch.equal(d.a1[~b], d.Atil[~b]))
        tot_sm = d.A_sm.sum(-1, dtype=torch.float64); tot_a1 = d.a1.sum(-1, dtype=torch.float64)
        out["binding_rows"] = int(b.sum())
        # the CAP's theta (Diagnostics.cap_theta), not Stage 2's `theta`, which is non-negative by construction and was
        # what this asserted before (review 9bacef9 E)
        out["cap_never_adds_mass"] = bool((d.a1 <= d.Atil).all()) and bool((th[b] > 0).all()) \
            and int(((d.Atil == 0) & (d.a1 > 0)).sum()) == 0
        out["min_cap_theta_on_binding"] = float(th[b].min()) if b.any() else None
        out["binding_rows_sum_to_softmax_mass_err"] = float((tot_a1 - tot_sm)[b].abs().max()) if b.any() else 0.0
        out["kernel_rowsum_error_live"] = float((tot_sm - 1).abs().max())
        _, dc = marsea_chunked_attention(norm, Q, K, K, vis, State(), chunk=1024)
        exc = dc.extra["excess"]
        out["dense_chunked_excess_diff"] = float((exc - ex).abs().max())
        out["dense_chunked_sm_mass_diff"] = float((dc.extra["sm_mass"] - sm_mech).abs().max())
        # the two paths may decide differently on a row whose excess is within their excess disagreement of CAP_TOL, or
        # inside the guard window (their fp32 Atil differ, so the guard may fall on different sides); nowhere else
        near_w = 2 * out["dense_chunked_excess_diff"] + window
        near = (ex - CAP_TOL).abs() <= near_w
        out["path_agreement_window"] = near_w
        out["dense_chunked_same_decision_off_boundary"] = bool(torch.equal(dc.cap_binds[~near], d.cap_binds[~near]))
        out["rows_at_boundary"] = int(near.sum())
        out["chunked_binding_rows"] = int(dc.cap_binds.sum())
        out["chunked_min_cap_theta_on_binding"] = float(dc.cap_theta[dc.cap_binds].min()) if dc.cap_binds.any() else None
        out["chunked_cap_theta_positive_on_binding"] = bool((dc.cap_theta[dc.cap_binds] > 0).all())
        out["decode_live"] = _decode_live(norm, S, vis, K, Q, n, tol_mass, force_empty=False)
        # sharpness THROUGH unit_cap (the excess rule AND the guard): a relation adding 4 max(CAP_TOL, window) binds on
        # every row; 5e-7 never.  Testing cap_binds_from_excess alone never exercised the guard (review 9bacef9 D).
        Asm = d.A_sm; E = torch.zeros_like(vis); E[..., :, 1:9] = True
        e_hi = 4 * max(CAP_TOL, window)
        res = {}
        for excess in (e_hi, 5e-7):
            Atil = Asm.clone(); Atil[E] = Asm[E] + excess / 8
            exs, sms = row_masses(Atil, Asm, E)
            _, _, bnd = unit_cap(Atil, vis, exs, sms)
            res[excess] = bnd
        out["binds_above_window_excess"] = e_hi
        out["binds_above_window"] = bool(res[e_hi].all()); out["binds_at_5e-7"] = bool(res[5e-7].any())   # .any(): NO row may bind
    out["passed"] = (out["dense_forced_empty_binds"] == 0 and out["dense_forced_empty_identity"] and out["chunked_forced_empty_binds"] == 0
                     and out["decode_forced_empty_binds"] == 0 and out["dense_live_decision_is_excess"]
                     and out["dense_live_identity_on_nonbinding"] and out["dense_chunked_same_decision_off_boundary"]
                     and out["cap_never_adds_mass"] and out["binding_rows_sum_to_softmax_mass_err"] <= tol_mass
                     and out["chunked_cap_theta_positive_on_binding"] and out["decode_live"]["ok"]
                     and out["excess_exact_vs_reference"] <= 1e-11 and out["binding_rows"] > 0      # fp64 regrouping only
                     and out["binds_above_window"] and not out["binds_at_5e-7"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="*", default=[2048, 4096, 8192, 16384])
    ap.add_argument("--elements", type=float, default=1e8, help="random softmax entries measured per (length, shape, scale)")
    ap.add_argument("--gaps", type=float, nargs="*", default=[15.5 + 0.05 * i for i in range(41)])
    ap.add_argument("--rhos", type=float, nargs="*", default=[0.05, 0.5], help="live-relation coverages gated")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/unit_cap_device.json")
    args = ap.parse_args()
    from marsea.invariants import cap_tolerance, validate_tol_cap_override
    from marsea.normalizer import CAP_TOL
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    g = torch.Generator(device=dev); g.manual_seed(args.seed)
    eps = torch.finfo(torch.float32).eps
    res = dict(device=name, torch=torch.__version__, created=time.strftime("%Y-%m-%dT%H:%M:%S"), CAP_TOL=CAP_TOL,
               MARSEA_TOL_CAP=os.environ.get("MARSEA_TOL_CAP") or None, lengths={})
    ok = True
    for n in args.lengths:
        kern = measure_kernel(n, args.gaps, g, dev, args.elements)
        san1 = 1e-5 * max(1.0, (n / 64.0) ** 0.5) + n * eps / 2
        gw = measure_guard_window(n, args.gaps, dev)
        tol_cap = cap_tolerance(n)
        window_ok = gw["window"] <= tol_cap and gw["unresolved_rows"] == 0
        # an override is bounded by the arithmetic (n eps / 4) and by the window just measured HERE (8 x), or the gate
        # fails: with the override inside cap_tolerance the comparison above would otherwise pass by construction, a
        # stale value surviving a reschedule onto a different GPU included (review 4f754ed 2)
        try:
            gw["override"] = validate_tol_cap_override(n, gw["window"])
        except ValueError as e:
            print(f"n_k = {n}: {e}"); gw["override_refused"] = str(e); window_ok = False
        if not window_ok:
            # the defined failure branch (review e6ca44f E): the tolerance is bound by MEASUREMENT, not by the log law
            print(f"n_k = {n}: theta-guard window {gw['window']:.2e} EXCEEDS the INV-3 tolerance {tol_cap:.2e}.  Set "
                  f"MARSEA_TOL_CAP={4 * gw['window']:.3e} (4 x the measured window) for preflight and the queue scripts, "
                  f"record it (unit_cap_record does), and re-run this gate.")
        gates = {rho: gate_mechanism(n, args.gaps, dev, rho, gw["window"]) for rho in args.rhos}
        kern_ok = all(v["error"] <= san1 for v in kern.values())
        ok &= all(gt["passed"] for gt in gates.values()) and kern_ok and window_ok
        res["lengths"][n] = dict(kernel_rowsum_error=kern, san1_bound=san1, kernel_within_san1=kern_ok,
                                 theta_guard_window=gw, inv3_tolerance=tol_cap, window_within_inv3=window_ok,
                                 mechanism={str(rho): gt for rho, gt in gates.items()})
        print(f"n_k = {n:6d}: kernel row-sum error torch.softmax {kern['torch.softmax']['error']:.2e} at {kern['torch.softmax']['at']}, "
              f"about-max {kern['about-max (chunked)']['error']:.2e}  (SAN-1 bound {san1:.1e}: {'ok' if kern_ok else 'EXCEEDED'})  |  "
              f"theta-guard window {gw['window']:.2e} = {gw['window_over_cap_tol']:.2f} x CAP_TOL (INV-3 allows {tol_cap:.1e}: "
              f"{'ok' if window_ok else 'EXCEEDED'})")
        for rho, gt in gates.items():
            dl = gt["decode_live"]
            print(f"    rho = {rho}: forced-empty binds dense/chunked/decode {gt['dense_forced_empty_binds']}/{gt['chunked_forced_empty_binds']}/"
                  f"{gt['decode_forced_empty_binds']}; live: dense {gt['binding_rows']} binding rows (guard left {gt['guard_suppressed_rows']}), "
                  f"never adds mass {gt['cap_never_adds_mass']}, min cap theta {gt['min_cap_theta_on_binding']}, "
                  f"|sum a1 - sum A_sm| {gt['binding_rows_sum_to_softmax_mass_err']:.1e} (<= {gt['tol_mass']:.1e}); "
                  f"chunked {gt['chunked_binding_rows']} binding, theta > 0 {gt['chunked_cap_theta_positive_on_binding']}; "
                  f"decode {dl['binding_rows']} binding, ok {dl['ok']} (mass err {dl['worst_mass_err']:.1e}); "
                  f"excess exact {gt['excess_exact_vs_reference']:.1e}, min|excess - CAP_TOL| {gt['min_abs_excess_minus_cap_tol']:.1e}, "
                  f"dense-chunked excess diff {gt['dense_chunked_excess_diff']:.1e} (sm_mass diff {gt['dense_chunked_sm_mass_diff']:.1e}; "
                  f"same decision off boundary {gt['dense_chunked_same_decision_off_boundary']}, {gt['rows_at_boundary']} rows at it); "
                  f"binds at {gt['binds_above_window_excess']:.1e} {gt['binds_above_window']}, any at 5e-7 {gt['binds_at_5e-7']}  "
                  f"-> {'ok' if gt['passed'] else 'FAILED'}")
    res["passed"] = bool(ok)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(res, indent=1))
    print("->", args.out)
    if not ok:
        print(f"GATE FAILED on {name}: see above"); sys.exit(3)
    print(f"GATE PASSED on {name}: the unit cap's decision is independent of this device's softmax kernel, and the theta "
          f"guard's window is inside INV-3's tolerance")


if __name__ == "__main__":
    main()
