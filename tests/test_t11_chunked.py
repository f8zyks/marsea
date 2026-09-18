"""T11: chunked == dense on every output tensor and diagnostic at T <= 4K, random E and tau, fp64 to 1e-9 then fp32
to a RELATIVE 1e-5 (spec Sec. 11 [v4]); plus gradient agreement between the two paths."""
import torch
import pytest
from marsea.normalizer import MarSeaNormalizer, State
from marsea.chunked import marsea_chunked_attention
from marsea.relation import repeat_kv

D, R = 16, 8


def _case(T, B=1, Hkv=1, g=2, dtype=torch.float32, pad=False, seed=0, rho=0.3):
    torch.manual_seed(seed)
    H = Hkv * g
    Q = torch.randn(B, H, T, D, dtype=dtype) * 1.5
    K_kv = torch.randn(B, Hkv, T, D, dtype=dtype) * 1.5
    V_kv = torch.randn(B, Hkv, T, D, dtype=dtype)
    i = torch.arange(T)[:, None]; j = torch.arange(T)[None, :]
    vis = (j <= i)[None, None].expand(B, 1, T, T).clone()
    if pad and B > 1:
        cut = T * 3 // 4
        vis[1, :, cut:, :] = False; vis[1, :, :, cut:] = False
    return Q, K_kv, V_kv, vis


def _dense(norm, Q, K_kv, V_kv, vis, state):
    g = Q.shape[1] // K_kv.shape[1]
    k_rep = repeat_kv(K_kv, g); v_rep = repeat_kv(V_kv, g)
    S = torch.einsum("bhid,bhjd->bhij", Q, k_rep) * D ** -0.5
    S = S.masked_fill(~vis, float("-inf"))
    A, d = norm.normalize(S, vis, K_kv, Q, state)
    O = torch.matmul(A, v_rep)
    return O, A, d


@pytest.mark.parametrize("dtype,tol,T,chunk", [(torch.float64, 1e-9, 200, 32), (torch.float64, 1e-9, 513, 128),
                                                (torch.float32, 1e-5, 200, 32), (torch.float32, 1e-5, 1024, 256)])
def test_t11_chunked_equals_dense(dtype, tol, T, chunk):
    torch.manual_seed(0)
    norm = MarSeaNormalizer(D, R).to(dtype)
    Q, K_kv, V_kv, vis = _case(T, B=2, dtype=dtype, pad=True)
    with torch.no_grad():
        norm.relation.calibrate_b0(K_kv, Q, vis, 0.3)
        nu_prev = torch.rand(2, 2, T, dtype=dtype) * 3 + 1
        O_d, A_d, dd = _dense(norm, Q, K_kv, V_kv, vis, State(nu_prev=nu_prev))
        O_c, dc = marsea_chunked_attention(norm, Q, K_kv, V_kv, vis, State(nu_prev=nu_prev), chunk=chunk, dense_outputs=True)

    def close(a, b, name):
        a = a.to(dtype); b = b.to(dtype)
        err = (a - b).abs()
        scale = b.abs().clamp_min(1.0) if dtype == torch.float32 else torch.ones_like(b)
        worst = float((err / scale).max()) if err.numel() else 0.0
        assert worst <= tol, f"{name}: worst {worst:.3e} > {tol}"
    close(O_c, O_d, "O")
    close(dc.A, A_d, "A"); close(dc.Atil, dd.Atil, "Atil"); close(dc.a1, dd.a1, "a1"); close(dc.A_sm, dd.A_sm, "A_sm")
    close(dc.cbar_j, dd.cbar_j, "cbar_j"); close(dc.cbar_i, dd.cbar_i, "cbar_i"); close(dc.tau_j, dd.tau_j, "tau_j")
    close(dc.tau_i, dd.tau_i, "tau_i"); close(dc.nu, dd.nu, "nu"); close(dc.Rtil, dd.Rtil, "Rtil"); close(dc.theta, dd.theta, "theta")
    assert (dc.E == dd.E).all()
    assert (dc.kstar == dd.kstar).all() and (dc.supp_rel == dd.supp_rel).all()
    assert (dc.cap_binds == dd.cap_binds).all()
    assert dc.extra["pass2_rebuilt_row_frac"] > 0, "no row went over one unit: pass 2's dense branch was not exercised"


def test_t11_gradients_agree():
    dtype = torch.float64
    torch.manual_seed(1)
    norm = MarSeaNormalizer(D, R).to(dtype)
    Q, K_kv, V_kv, vis = _case(96, B=1, dtype=dtype, seed=3)
    with torch.no_grad():
        norm.relation.calibrate_b0(K_kv, Q, vis, 0.35)
    w = torch.randn(1, 2, 96, D, dtype=dtype)
    grads = {}
    for name, fn in (("dense", lambda: _dense(norm, Q, K_kv, V_kv, vis, State())[0]),
                     ("chunked", lambda: marsea_chunked_attention(norm, Q, K_kv, V_kv, vis, State(), chunk=16)[0])):
        norm.zero_grad()
        Qg = Q.clone().requires_grad_(True); Kg = K_kv.clone().requires_grad_(True)
        if name == "dense":
            O = _dense(norm, Qg, Kg, V_kv, vis, State())[0]
        else:
            O = marsea_chunked_attention(norm, Qg, Kg, V_kv, vis, State(), chunk=16)[0]
        (O * w).sum().backward()
        grads[name] = {n: p.grad.clone() for n, p in norm.named_parameters() if p.grad is not None}
        grads[name]["Q"] = Qg.grad.clone(); grads[name]["K"] = Kg.grad.clone()
    for k in grads["dense"]:
        a, b = grads["dense"][k], grads["chunked"][k]
        scale = max(1e-12, float(a.abs().max()))
        assert float((a - b).abs().max()) / scale < 1e-8, f"grad mismatch on {k}: {(a - b).abs().max()} (scale {scale})"


def test_t11_checkpointed_backward_runs_fp32():
    torch.manual_seed(2)
    norm = MarSeaNormalizer(D, R)
    Q, K_kv, V_kv, vis = _case(300, B=1)
    Qg = Q.clone().requires_grad_(True)
    O, dc = marsea_chunked_attention(norm, Qg, K_kv, V_kv, vis, State(), chunk=64)
    O.sum().backward()
    assert torch.isfinite(Qg.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in norm.parameters() if p.grad is not None)


def test_chunked_is_fp32_under_autocast():
    """H1 / read-through follow-up: the chunked path is called straight from the patched layer, so it must disable
    autocast itself.  Without the guard every matmul and einsum in it is autocast-eligible and runs in bf16 -- the dtype
    that loses the exact zeros the mechanism is -- and the result differs from the un-autocast run."""
    if not torch.cuda.is_available():
        import pytest; pytest.skip("needs a GPU for bf16 autocast")
    torch.manual_seed(0)
    norm = MarSeaNormalizer(D, R).cuda()
    Q, K_kv, V_kv, vis = _case(192, B=1, seed=5)
    Q, K_kv, V_kv, vis = Q.cuda().bfloat16(), K_kv.cuda().bfloat16(), V_kv.cuda().bfloat16(), vis.cuda()
    with torch.no_grad():
        norm.relation.calibrate_b0(K_kv.float(), Q.float(), vis, 0.3)
        O_plain, d_plain = marsea_chunked_attention(norm, Q, K_kv, V_kv, vis, State(), chunk=64)
        O_again, _ = marsea_chunked_attention(norm, Q, K_kv, V_kv, vis, State(), chunk=64)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            O_auto, d_auto = marsea_chunked_attention(norm, Q, K_kv, V_kv, vis, State(), chunk=64)
    assert O_auto.dtype == torch.float32 and d_auto.tau_j.dtype == torch.float32, "the programs must stay fp32"
    # the path's own run-to-run spread (index_add over the sparse relation entries is order-nondeterministic on GPU) is
    # the only difference autocast may make: anything larger means a program ran in bf16
    # index_add over the sparse relation entries is order-nondeterministic on GPU and its spread varies run to run, so
    # the bound is the measured spread or a fixed 1e-5 -- still a thousandfold below what running a program in bf16
    # would produce (~1e-2 relative), which is what this test exists to catch
    nondet = float((O_plain - O_again).abs().max())
    assert float((O_plain - O_auto).abs().max()) <= max(nondet, 1e-5), "autocast changed the chunked result"
    assert torch.equal(d_plain.cbar_j, d_auto.cbar_j) and torch.equal(d_plain.kstar, d_auto.kstar)
    assert torch.equal(d_plain.tau_j, d_auto.tau_j) and torch.equal(d_plain.nu, d_auto.nu)


def test_chunked_dense_head_matches_all_heads():
    """The chunked path's dense diagnostics must be rebuildable for ONE head (the fidelity statistics only ever read
    (l*, h*)): at 16K a single [1, H, T, T] fp32 tensor is 12.9 GB, so E3 -- the load-bearing experiment, run at 16K --
    would OOM on its own diagnostics.  The single-head rebuild must equal the corresponding slice of the all-head one."""
    torch.manual_seed(0)
    norm = MarSeaNormalizer(D, R)
    Q, K_kv, V_kv, vis = _case(96, B=1, seed=7)
    with torch.no_grad():
        norm.relation.calibrate_b0(K_kv, Q, vis, 0.3)
        _, d_all = marsea_chunked_attention(norm, Q, K_kv, V_kv, vis, State(), chunk=32, dense_outputs=True)
        _, d_one = marsea_chunked_attention(norm, Q, K_kv, V_kv, vis, State(), chunk=32, dense_outputs=True, dense_head=1)
    assert d_one.extra["head_sub"] == 1 and d_one.A.shape[1] == 1 and d_all.A.shape[1] == Q.shape[1]
    for name in ("E", "supp_rel", "kstar"):                      # discrete quantities must match exactly
        a, b = getattr(d_all, name), getattr(d_one, name)
        if a is None or b is None: continue
        assert torch.equal(a[:, 1:2], b), f"{name} differs between the all-head and single-head rebuild"
    for name in ("A", "Atil", "a1", "A_sm", "p", "cbar_j", "tau_j", "nu", "cbar_i", "tau_i", "theta", "Rtil"):
        a, b = getattr(d_all, name), getattr(d_one, name)
        if a is None or b is None: continue
        # the single-head rebuild runs a differently-shaped GEMM, so it rounds differently in the last bit
        assert torch.allclose(a[:, 1:2], b, atol=1e-7, rtol=1e-5), \
            f"{name} differs beyond rounding: {(a[:, 1:2] - b).abs().max():.2e}"


@pytest.mark.parametrize("row_block", [16, 64, 100000])
def test_pass2_densification_is_row_blocked_and_invariant(row_block):
    """Pass 2 rebuilds the rows the unit cap binds.  It used to gather k_rep_full[b_r, h_r] and v_rep_full[b_r, h_r] --
    [n_over, n_k, d], i.e. d times the score tensor beside them -- so its cost scaled with frac_over: ~8 GB at 8K when
    1 % of rows bind and ~82 GB at the 10 % the training procedure calls healthy, arriving first at 16K where E3 lives.
    Rows are now grouped by (b, h) (plain matmuls against [n_k, d], no gather) and processed in bounded blocks.
    The result must not depend on the block size, and pass 2 must actually be exercised here."""
    torch.manual_seed(0)
    norm = MarSeaNormalizer(D, R).double()
    Q, K_kv, V_kv, vis = _case(160, B=1, dtype=torch.float64, seed=11)
    with torch.no_grad():
        norm.relation.calibrate_b0(K_kv, Q, vis, 0.6)         # a dense relation, so columns concentrate and rows bind
        norm.tauK.set_init_tau(20.0)
        O, d = marsea_chunked_attention(norm, Q, K_kv, V_kv, vis, State(), chunk=32, row_block=row_block)
    assert d.extra["pass2_rebuilt_row_frac"] > 0.02, "pass 2 never fired: the test does not exercise the branch"
    if row_block == 16:
        test_pass2_densification_is_row_blocked_and_invariant.ref = (O, d)
    else:
        O0, d0 = test_pass2_densification_is_row_blocked_and_invariant.ref
        assert torch.allclose(O, O0, atol=1e-12), f"row_block changed the output by {(O - O0).abs().max():.2e}"
        for n in ("cbar_i", "tau_i", "theta", "Rtil", "supp_rel"):
            a, b = getattr(d0, n), getattr(d, n)
            assert torch.allclose(a.double(), b.double(), atol=1e-12), f"row_block changed {n}"


def test_chunked_backward_is_fp32_under_autocast():
    """The guard inside _chunk_fanout exists for exactly one scenario -- the CHECKPOINTED RECOMPUTE under ambient
    autocast, since torch.utils.checkpoint restores the ambient autocast state when it recomputes.  The forward-only
    test above never enters it (review 2026-09-12).  Here the backward runs under autocast(bf16) and must produce the
    same gradients as the un-autocast backward."""
    if not torch.cuda.is_available():
        import pytest; pytest.skip("needs a GPU for bf16 autocast")
    torch.manual_seed(0)
    norm = MarSeaNormalizer(D, R).cuda()
    Q, K_kv, V_kv, vis = _case(160, B=1, seed=9)
    Q, K_kv, V_kv, vis = Q.cuda(), K_kv.cuda(), V_kv.cuda(), vis.cuda()
    with torch.no_grad():
        norm.relation.calibrate_b0(K_kv, Q, vis, 0.4)
    w = torch.randn(1, 2, 160, D, device="cuda")
    grads = {}
    for tag, auto in (("plain", False), ("autocast", True)):
        norm.zero_grad(set_to_none=True)
        Qg = Q.clone().requires_grad_(True)
        ctxm = torch.autocast("cuda", dtype=torch.bfloat16) if auto else torch.autocast("cuda", enabled=False)
        with ctxm:                                   # forward under autocast, as the patched layer runs it ...
            O, d = marsea_chunked_attention(norm, Qg, K_kv, V_kv, vis, State(), chunk=32)
            assert O.dtype == torch.float32 and d.tau_j.dtype == torch.float32
            loss = (O.float() * w).sum()
        loss.backward()                              # ... and backward OUTSIDE it, exactly as train_step does
        grads[tag] = {n: p.grad.clone() for n, p in norm.named_parameters() if p.grad is not None}
        grads[tag]["Q"] = Qg.grad.clone()
        grads[tag]["_cbar_j"] = d.cbar_j.clone(); grads[tag]["_tau_j"] = d.tau_j.clone()
    assert torch.equal(grads["plain"]["_cbar_j"], grads["autocast"]["_cbar_j"]), "cbar_j differs under autocast"
    assert torch.equal(grads["plain"]["_tau_j"], grads["autocast"]["_tau_j"]), "tau_j differs under autocast"
    for k in grads["plain"]:
        if k.startswith("_"): continue
        a, b = grads["plain"][k], grads["autocast"][k]
        scale = max(1e-12, float(a.abs().max()))
        assert float((a - b).abs().max()) / scale < 1e-4, f"gradient of {k} changed under autocast (bf16 recompute?)"
    # and the ordering train_step forbids really is the dangerous one: backward INSIDE autocast leaks bf16 into the
    # gradients (~3e-2 relative), which is why train_step asserts torch.is_autocast_enabled() is False
    norm.zero_grad(set_to_none=True)
    Qg = Q.clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        O, _ = marsea_chunked_attention(norm, Qg, K_kv, V_kv, vis, State(), chunk=32)
        (O.float() * w).sum().backward()
    leak = max(float((grads["plain"][n] - p.grad).abs().max()) / max(1e-12, float(grads["plain"][n].abs().max()))
               for n, p in norm.named_parameters() if p.grad is not None)
    assert leak > 1e-4, "expected the backward-under-autocast ordering to degrade gradients; the guard's premise changed"


@pytest.mark.parametrize("hb", [1, 2, 3])
def test_chunked_head_blocking_is_identical(hb):
    """Head blocking must work on the CHUNKED path too -- it is the path S2 runs, and it could not head-block at all
    (review 2026-09-12)."""
    torch.manual_seed(0)
    full = MarSeaNormalizer(D, R).double()
    blk = MarSeaNormalizer(D, R, head_block=hb).double(); blk.load_state_dict(full.state_dict())
    Q, K_kv, V_kv, vis = _case(128, B=1, dtype=torch.float64, seed=5)
    with torch.no_grad():
        full.relation.calibrate_b0(K_kv, Q, vis, 0.4)
        blk.relation.calibrate_b0(K_kv, Q, vis, 0.4)
        s1, s2 = State(), State()
        O1, d1 = marsea_chunked_attention(full, Q, K_kv, V_kv, vis, s1, chunk=32)
        O2, d2 = marsea_chunked_attention(blk, Q, K_kv, V_kv, vis, s2, chunk=32)
    assert torch.equal(O1, O2) and torch.equal(d1.supp_rel, d2.supp_rel) and torch.equal(d1.kstar, d2.kstar)
    assert torch.equal(s1.nu_next, s2.nu_next)
    for n in ("cbar_j", "tau_j", "nu", "cbar_i", "tau_i", "theta", "Rtil"):
        assert torch.equal(getattr(d1, n), getattr(d2, n)), f"{n} differs under head blocking on the chunked path"


def test_e8_agrees_between_the_dense_and_chunked_paths():
    """The chunked path never materialises E, so the dense E8 summary cannot run on it -- and without a sparse
    equivalent E8 would be empty for the whole of S2 (which trains chunked): no coverage, no |E_.j| histogram, no k*
    histogram, hence none of the three pre-registered degeneracy readings.  The two summaries must agree."""
    from marsea.fidelity import e8_summary_from_diag, e8_summary_from_sparse
    torch.manual_seed(0)
    norm = MarSeaNormalizer(D, R)
    Q, K_kv, V_kv, vis = _case(96, B=1, seed=3)
    with torch.no_grad():
        norm.relation.calibrate_b0(K_kv, Q, vis, 0.25)
        from marsea.relation import repeat_kv
        k_rep = repeat_kv(K_kv, Q.shape[1] // K_kv.shape[1])
        S = (torch.einsum("bhid,bhjd->bhij", Q, k_rep) * D ** -0.5).masked_fill(~vis, float("-inf"))
        _, d_dense = norm.normalize(S, vis, K_kv, Q, State())
        _, d_chunk = marsea_chunked_attention(norm, Q, K_kv, V_kv, vis, State(), chunk=32)
    a = e8_summary_from_diag(d_dense, vis)
    b = e8_summary_from_sparse(d_chunk, vis)
    assert b, "the chunked path produced no E8 block"
    assert abs(a["rho"] - b["rho"]) < 1e-6, f"coverage differs: {a['rho']} vs {b['rho']}"
    for k in ("rho_head", "frac_Ecol_singleton", "tau_j_median", "var_tau_j"):
        for x, y in zip(a[k], b[k]):
            if x == x and y == y:
                assert abs(x - y) < 1e-5, f"{k} differs between the paths: {x} vs {y}"
    for ha, hb_ in zip(a["hist_Ecol"], b["hist_Ecol"]):
        for lab in ha:
            assert abs(ha[lab] - hb_[lab]) < 1e-6, f"|E_.j| histogram differs in bin {lab}"
    # EVERY key the dense summary reports must have a sparse counterpart that agrees.  The first version of this test
    # compared 6 of 18, and five dense keys had no sparse counterpart at all -- frac_rows_zero_mass (the D-31
    # diagnostic), kstar_by_Ecol, row_trigger_rate, col_induced_rate, row_trigger_violation (review d5bd980).
    missing = [k for k in a if k not in b]
    assert not missing, f"the sparse summary has no counterpart for {missing}"
    for k, va in a.items():
        vb = b[k]
        if isinstance(va, float) and isinstance(vb, float):
            assert (va != va and vb != vb) or abs(va - vb) < 1e-5, f"{k}: {va} vs {vb}"
        elif isinstance(va, bool) or va is None:
            assert va == vb, f"{k}: {va} vs {vb}"
        elif isinstance(va, list) and va and isinstance(va[0], (int, float)):
            for x, y in zip(va, vb):
                if x == x and y == y:
                    assert abs(x - y) < 1e-5, f"{k} differs between the paths: {x} vs {y}"
        elif isinstance(va, list) and va and isinstance(va[0], dict):
            for ha_, hb2 in zip(va, vb):
                for lab in ha_:
                    x, y = ha_[lab], hb2[lab]
                    if x == x and y == y:
                        assert abs(x - y) < 1e-5, f"{k}[{lab}]: {x} vs {y}"


def test_invariants_run_on_the_chunked_path_and_catch_violations():
    """check_invariants was called from the dense normaliser only, and invariant_report's first statement reads
    d.Atil.dtype -- so it RAISED on a chunked Diagnostics instead of degrading, and MARSEA_DEBUG=1 had no effect at
    all on the path S2 trains (review d5bd980 F).  The sparse report must run, and it must not be inert."""
    from marsea.invariants import invariant_report_sparse, check_invariants, InvariantError
    torch.manual_seed(0)
    norm = MarSeaNormalizer(D, R).double()
    Q, K_kv, V_kv, vis = _case(64, B=1, seed=5)
    Q, K_kv, V_kv = Q.double(), K_kv.double(), V_kv.double()
    with torch.no_grad():
        norm.relation.calibrate_b0(K_kv, Q, vis, 0.3)
        _, d = marsea_chunked_attention(norm, Q, K_kv, V_kv, vis, State(), chunk=16)
    rep = invariant_report_sparse(d, vis)
    assert {"INV-1", "INV-3", "INV-3-recon", "INV-5", "INV-6", "INV-7", "INV-8", "finite"} <= set(rep), rep
    checked = {k: v for k, v in rep.items() if not k.startswith("_")}
    assert all(ok for ok, _ in checked.values()), checked
    # what the sparse store cannot see is RECORDED, not silently missing
    absent = " ".join(rep["_not_checked"])
    for name in ("INV-2", "INV-4", "SAN-1"):
        assert name in absent, f"{name} is absent from the sparse report without being declared so"
    # and an empty report is not a pass
    from marsea.normalizer import Diagnostics
    try:
        check_invariants(None, Diagnostics(), vis)
        raise AssertionError("an empty Diagnostics passed the invariant check")
    except InvariantError as e:
        assert "INERT" in str(e)
    # the row's total final mass is reconstructed without any dense row: it must equal the dense path's A.sum(-1)
    from marsea.relation import repeat_kv
    with torch.no_grad():
        k_rep = repeat_kv(K_kv, Q.shape[1] // K_kv.shape[1])
        S = (torch.einsum("bhid,bhjd->bhij", Q, k_rep) * D ** -0.5).masked_fill(~vis, float("-inf"))
        A, _ = norm.normalize(S, vis, K_kv, Q, State())
    assert (d.extra["A_rowsum"] - A.sum(-1)).abs().max() < 1e-12
    for field, inv in (("A", "INV-7"), ("A_rowsum", "INV-3")):
        import copy
        dc = copy.copy(d); dc.extra = dict(d.extra)
        if field == "A":
            dc.extra["sparse"] = dict(d.extra["sparse"]); dc.extra["sparse"]["A"] = d.extra["sparse"]["A"] + 1.0
        else:
            dc.extra[field] = d.extra[field] + 0.5
        try:
            check_invariants(None, dc, vis)
            raise AssertionError(f"a corrupted {field} passed the sparse invariant check")
        except InvariantError as e:
            assert inv in str(e), f"corrupting {field} did not trip {inv}: {e}"


@pytest.mark.parametrize("hb", [1, 2, 3])
def test_per_head_parameters_compose_with_head_blocking(hb):
    """run_s2.sh put --head_block in COMMON, i.e. on every job including the per_head = 12 E9 arm.  With per_head
    the relation's U/V and the tau MLPs are indexed in GLOBAL head space, so a block carrying `hb` heads either
    crashed the einsum or -- in pass 2's `norm.relation.V[h]` -- silently used the WRONG head's V on every
    cap-binding row (review d5bd980 B-2).  head_offset makes the two compose, exactly."""
    torch.manual_seed(0)
    H = 6
    Q, K_kv, V_kv, vis = _case(96, B=1, Hkv=2, g=H // 2, dtype=torch.float64, seed=2)
    norm = MarSeaNormalizer(D, R, per_head=H).double()
    with torch.no_grad():
        norm.relation.calibrate_b0(K_kv, Q, vis, 0.35)
    Q = Q.clone().requires_grad_(True); K = K_kv.clone().requires_grad_(True)
    norm.head_block = None
    O0, d0 = marsea_chunked_attention(norm, Q, K, V_kv, vis, State(), chunk=32)
    (O0 * torch.arange(O0.numel(), dtype=O0.dtype).view_as(O0)).sum().backward()
    g0 = (Q.grad.clone(), K.grad.clone(), norm.relation.V.grad.clone(), norm.tauK.w1.grad.clone())
    Q.grad = None; K.grad = None; norm.zero_grad()
    norm.head_block = hb
    O1, d1 = marsea_chunked_attention(norm, Q, K, V_kv, vis, State(), chunk=32)
    (O1 * torch.arange(O1.numel(), dtype=O1.dtype).view_as(O1)).sum().backward()
    g1 = (Q.grad, K.grad, norm.relation.V.grad, norm.tauK.w1.grad)
    # exact to fp64 rounding, not bitwise: blocking changes the GEMM shapes (and, for the per-head parameters, the
    # gradient REDUCTION ORDER), which is the scope note the review records in section G.  Measured here: values
    # 4e-15, gradients 4e-11 in fp64; expect ~1e-7 relative drift in fp32.
    assert (O0 - O1).abs().max() < 1e-12, f"per_head + head_block={hb} changed the output"
    for n in ("cbar_j", "tau_j", "nu", "cbar_i", "tau_i", "theta", "Rtil"):
        a, b = getattr(d0, n), getattr(d1, n)
        assert (a - b).abs().max() < 1e-11, f"{n} differs at head_block={hb}"
    for a, b, name in zip(g0, g1, ("dQ", "dK", "dV_rel", "dw1_tauK")):
        den = a.abs().max().clamp_min(1e-12)
        assert ((a - b).abs().max() / den) < 1e-9, f"{name} differs at head_block={hb}: {(a - b).abs().max():.3e}"


def test_chunked_prefill_seeds_the_decode_cache_like_the_dense_one():
    """D-28's frozen prefix was unreachable under --mode chunked: the chunked branch returns before the dense cache
    seeding, so every decode step recomputed tau_j live at n_vis = 1 instead of using the value sealed at prefill,
    in the generation mode of the load-bearing experiment (review d5bd980 B-3)."""
    from marsea.causal import FrozenPrefixCache
    torch.manual_seed(0)
    T, K_RET = 64, 8
    norm = MarSeaNormalizer(D, R).double()
    Q, K_kv, V_kv, vis = _case(T, B=1, Hkv=1, g=2, dtype=torch.float64, seed=7)
    with torch.no_grad():
        norm.relation.calibrate_b0(K_kv, Q, vis, 0.3)
        O_c, d_c = marsea_chunked_attention(norm, Q, K_kv, V_kv, vis, State(), chunk=16)
        _, A, d_d = _dense(norm, Q, K_kv, V_kv, vis, State())
    B, H = Q.shape[0], Q.shape[1]
    dense_cache = FrozenPrefixCache(K_ret=K_RET)
    S = (torch.einsum("bhid,bhjd->bhij", Q, repeat_kv(K_kv, H)) * D ** -0.5).masked_fill(~vis, float("-inf"))
    dense_cache.init_from_prefill(S, d_d)
    sparse_cache = FrozenPrefixCache(K_ret=K_RET)
    sparse_cache.init_from_sparse(d_c, B, H, T, T, Q.device, dtype=torch.float64)
    assert sparse_cache.n_seen == dense_cache.n_seen == T
    for name in ("tau_j", "cbar_j", "nu"):
        a, b = getattr(dense_cache, name), getattr(sparse_cache, name)
        assert (a - b).abs().max() < 1e-12, f"{name} differs between the dense and chunked prefills"
    assert torch.equal(dense_cache.rel_valid, sparse_cache.rel_valid), "the relation's per-column top-K differs"
    dv, sv = dense_cache.rel_scores[dense_cache.rel_valid], sparse_cache.rel_scores[sparse_cache.rel_valid]
    assert (dv - sv).abs().max() < 1e-12, "the stored relation scores differ"
    assert torch.equal(dense_cache.rel_qidx[dense_cache.rel_valid], sparse_cache.rel_qidx[sparse_cache.rel_valid])
    # truncation_events is a REPORTED D-28/E8 reading, so it must not depend on which prefill built the cache
    # (review 30372ae B-2): the dense side counted only the kstar term, the sparse side that plus the columns whose
    # relation exceeds K_ret -- the event _insert_scores counts at decode time.
    for nm in ("eviction_events", "near_truncation_events"):
        assert getattr(dense_cache, nm) == getattr(sparse_cache, nm), (
            f"{nm}: {getattr(dense_cache, nm)} (dense prefill) vs {getattr(sparse_cache, nm)} (chunked prefill)")
    assert sparse_cache.eviction_events > 0, "the test does not reach a column bigger than K_ret"


def test_zero_mass_rows_survive_the_fp32_reconstruction():
    """frac_rows_zero_mass is the D-31 diagnostic.  Dense computes (A.sum(-1) == 0) exactly; sparse reads a
    reconstruction whose fp32 residual at maximal cancellation -- which IS the zero-mass case -- is ~1e-7, so an
    exact `<= 0` decided at random (review 30372ae B-3)."""
    from marsea.fidelity import e8_summary_from_sparse, _zero_mass_tol
    torch.manual_seed(0)
    norm = MarSeaNormalizer(D, R)
    Q, K_kv, V_kv, vis = _case(96, B=1, seed=11)
    with torch.no_grad():
        norm.relation.calibrate_b0(K_kv, Q, vis, 0.3)
        _, d = marsea_chunked_attention(norm, Q, K_kv, V_kv, vis, State(), chunk=32)
    n_k = Q.shape[2]
    tol = _zero_mass_tol(d.extra["A_rowsum"], n_k)
    assert 0 < tol < 1e-4, tol
    base = e8_summary_from_sparse(d, vis)
    assert base["zero_mass_tol"] == pytest.approx(tol)
    # a row the mechanism emptied lands anywhere in +-tol, not at exactly 0: every sign must read as zero mass
    ar = d.extra["A_rowsum"].clone()
    ar[0, 0, 5] = 0.0; ar[0, 0, 6] = tol / 3; ar[0, 0, 7] = -tol / 3
    ar[0, 0, 8] = 1e-2                                        # real mass: neither zero nor ambiguous
    ar[0, 0, 9] = 1e-4                                        # between the tolerance and a mass a reader trusts
    d.extra["A_rowsum"] = ar
    out = e8_summary_from_sparse(d, vis)
    n_real = int(vis[0, 0].any(-1).sum())
    assert out["frac_rows_zero_mass"][0] == pytest.approx(3 / n_real), "the +-tol rows must all count as zero mass"
    assert out["frac_rows_zero_mass_ambiguous"][0] == pytest.approx(1 / n_real), "the band must be reported, not hidden"
