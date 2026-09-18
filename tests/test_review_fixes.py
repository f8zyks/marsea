"""Regression tests for the 2026-09-09 code review: D-31 (sink excluded from every relation), A5/A6 (hit defined only
where Thm. 2 speaks), A10 (right-derivative of the unit cap at Atil == 0), B13 (parameter groups), A3 (stageless arms)."""
import pytest
import torch
import numpy as np
from conftest import rand_case
from marsea.normalizer import MarSeaNormalizer, State
from marsea.relation import sink_pair_mask
from marsea.primitives import proj_le_masked, sparsemax_masked
from marsea.fidelity import column_fidelity

D, R = 16, 8


def test_d31_sink_excluded(rng):
    torch.manual_seed(0)
    norm = MarSeaNormalizer(D, R)
    for t in range(30):
        cs = rand_case(rng, n_q=int(rng.integers(3, 30)))
        with torch.no_grad():
            norm.relation.b0.fill_(+5.0)                          # nearly everything visible is in the relation ...
            A, d = norm.normalize(cs["S"], cs["vis"], cs["K_kv"], cs["Q"], State())
        n_q, n_k = d.E.shape[-2:]
        assert not d.E[..., :, 0].any(), "the sink COLUMN must never be in a relation"
        i0 = n_q - n_k
        if 0 <= i0 < n_q:
            assert not d.E[..., i0, :].any(), "the sink ROW must never be in a relation"
        vis = cs["vis"].expand_as(d.E)
        others = vis & ~sink_pair_mask(n_q, n_k, d.E.device)
        if others.any():
            assert d.E[others].float().mean() > 0.9              # ... except the sink pairs
    # the sink pairs receive no straight-through gradient
    cs = rand_case(rng, n_q=12, n_k=12)
    K = cs["K_kv"].clone().requires_grad_(True)
    A, d = norm.normalize(cs["S"], cs["vis"], K, cs["Q"], State())
    (A * torch.randn_like(A)).sum().backward()
    assert torch.isfinite(K.grad).all()
    # chunk-aware mask: a chunk starting at key 5 has no sink column; a decode row (n_q = 1, n_k = 9) has no sink row
    assert not sink_pair_mask(4, 6, "cpu", key_offset=5, n_k_total=20).any()
    m = sink_pair_mask(1, 9, "cpu"); assert m[0, 0] and m.sum() == 1
    # b0 calibration ignores the sink pairs
    cs = rand_case(rng, n_q=20, n_k=20, B=1, Hkv=1, H=1)
    norm.relation.calibrate_b0(cs["K_kv"], cs["Q"], cs["vis"], 0.3)
    with torch.no_grad():
        _, d = norm.normalize(cs["S"], cs["vis"], cs["K_kv"], cs["Q"], State())
    vis = cs["vis"].expand_as(d.E); others = vis & ~sink_pair_mask(20, 20, d.E.device)
    assert abs(d.E[others].float().mean() - 0.3) < 0.08


def test_a10_unit_cap_right_derivative():
    v = torch.tensor([[0.0, 0.3, 0.0, 0.2]], requires_grad=True)
    a, _ = proj_le_masked(v, torch.ones(1), torch.ones(1, 4, dtype=torch.bool), binds=torch.zeros(1, dtype=torch.bool))
    a.sum().backward()
    assert torch.equal(v.grad, torch.ones(1, 4)), "the slack cap must be the identity in the backward too, at exact zeros"


def test_a5_a6_hit_defined_only_where_thm2_speaks():
    s = torch.tensor([2.0, 1.0, -1.0, -2.0]); Tm = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
    E = torch.ones(4, dtype=torch.bool); p = sparsemax_masked(s[None], E[None])[0]
    c = column_fidelity(s, E, p, p, 0.6, Tm)
    assert c["hit"] is not None and c["hit_excluded"] is None
    # |E_.j| = 1 containing its target: a hit at every tau in the old code -> now excluded
    E1 = torch.tensor([1, 0, 0, 0], dtype=torch.bool); p1 = sparsemax_masked(s[None], E1[None])[0]
    for tau in (0.05, 1.0, 100.0):
        c = column_fidelity(s, E1, p1, p1, tau, Tm)
        assert c["hit"] is None and c["hit_excluded"] == "|E_.j| < 2"
    # delta <= 0: not a miss, an excluded column (reported through frac_columns_delta_pos)
    s2 = torch.tensor([2.0, -1.0, 1.5, -2.0]); p2 = sparsemax_masked(s2[None], E[None])[0]
    c = column_fidelity(s2, E, p2, p2, 1.0, Tm)
    assert c["hit"] is None and c["hit_excluded"] == "delta <= 0" and not c["delta_pos"]
    # empty complement with W = 0: the degenerate interval [0, inf)
    s3 = torch.tensor([1.0, 1.0, 0.0, 0.0]); E3 = torch.tensor([1, 1, 0, 0], dtype=torch.bool); p3 = sparsemax_masked(s3[None], E3[None])[0]
    c = column_fidelity(s3, E3, p3, p3, 3.0, Tm)
    assert c["hit"] is None and c["hit_excluded"].startswith("degenerate")


def test_b13_param_groups_by_name():
    norm = MarSeaNormalizer(D, R, per_head=3)
    w, o = norm.new_module_params()
    names_o = {n for n, p in norm.named_parameters() if any(p is q for q in o)}
    assert all(x in names_o for x in ("relation.b0", "tauK.b1", "tauK.b2", "tauQ.b1", "tauQ.b2", "tauK.ln.weight", "tauK.ln.bias"))
    assert all(("ln." not in n and not n.endswith(("b1", "b2", "b0", "bias"))) for n, p in norm.named_parameters() if any(p is q for q in w))


def test_a3_stageless_arm_scoring():
    from marsea.evaluate import _stageless_attention_level
    from marsea.data.ruler import RULERExample
    n = 10
    A = torch.zeros(n, n); A[8, 3] = 0.5; A[8, 5] = 0.5; A[9, 5] = 1.0
    ex = RULERExample(prompt="", gold="", outputs=["x", "y"], query_key="k", prompt_ids=list(range(8)), gold_ids=[0, 0],
                      answer_rows=[8, 9], value_spans=[(3, 4), (5, 6)], sentence_spans=[], key_mentions=[1], question_mention=None,
                      T_j={1: [7, 8, 9]})
    E = torch.zeros(n, n, dtype=torch.bool); E[8, [3, 4, 5]] = True; E[9, [5, 6]] = True; E[7:, 1] = True
    out = _stageless_attention_level(A, None, ex, E, lambda i: torch.arange(n) <= i, n)
    r8 = out["rows"][0]; assert r8["P_after"] == 1.0 and r8["P_before"] == r8["P_after"] and out["stageless"]
    assert "P_rel" in out["cols"][0] and out["cols"][0]["E_size"] == 3 and out["cols"][0]["P_rel"] == 1.0   # empty support -> 1.0 by convention


# ----------------------------------------------------------------- read-through 2026-09-11
def test_f2_e8_snapshot_survives_the_accumulation_window():
    """F-2: the model's forward pre-hook clears ctx.diags on EVERY forward, so the E8 payload collected on micro-batch 0
    must be snapshotted; otherwise e8_summary() sees {} and E8 is empty for the whole of S2."""
    from marsea.backbone import MarSeaContext
    from marsea.train import e8_summary
    from marsea.normalizer import Diagnostics
    ctx = MarSeaContext()
    ctx.collect = True
    ctx.diags = {14: Diagnostics(extra={"e8": {"rho": 0.05, "rho_head": [0.05]}})}
    # what train_step does after micro 0
    if not hasattr(ctx, "extra_log"): ctx.extra_log = {}
    ctx.extra_log["e8"] = {l: d.extra.get("e8") for l, d in ctx.diags.items() if d is not None and "e8" in d.extra}
    ctx.reset_forward(None)                                  # micro-batches 1..15 wipe ctx.diags
    assert ctx.diags == {}
    e8 = e8_summary(ctx)
    assert e8 and e8[14]["rho"] == 0.05, "the E8 block must survive the accumulation window"


def test_f3_chunked_refuses_the_uniform_quota():
    from marsea.chunked import marsea_chunked_attention
    from marsea.normalizer import MarSeaNormalizer, State
    torch.manual_seed(0)
    norm = MarSeaNormalizer(D, R, quota_mode="uniform")
    q = torch.randn(1, 2, 16, D); k = torch.randn(1, 1, 16, D); v = torch.randn(1, 1, 16, D)
    i = torch.arange(16)[:, None]; j = torch.arange(16)[None, :]
    vis = (j <= i)[None, None]
    with pytest.raises(AssertionError, match="inherited quota only"):
        marsea_chunked_attention(norm, q, k, v, vis, State(), chunk=8)


def test_f5_replication_M_excludes_the_sink():
    from marsea.sweep import replication_M
    S = torch.full((6,), float("-inf")); S[0] = 5.0; S[2] = 1.0; S[3] = 0.9; S[4] = 0.2
    M = replication_M(S, j=2)                                  # the sink (query 0) scores highest but is not a candidate
    assert M[0.5] == 2 and M[1.0] == 3, M                      # {1.0, 0.9} within 0.5 of max 1.0; +0.2 within 1.0
    M0 = replication_M(S, j=0)                                 # the key IS position 0: the column has nothing else
    assert M0[0.5] == 1


def test_hierarchical_topk_deployed_and_flag_keyed_to_it(rng):
    """the tournament is exact when K_ret >= k*, and the E8 truncation flag is None when no hierarchy ran."""
    from marsea.normalizer import MarSeaNormalizer, State
    from marsea.fidelity import e8_summary_from_diag, column_fidelity
    from conftest import rand_case
    torch.manual_seed(0)
    flat = MarSeaNormalizer(D, R)
    hier = MarSeaNormalizer(D, R, K_ret=64, block_size=8)
    hier.load_state_dict(flat.state_dict())
    cs = rand_case(rng, n_q=40, n_k=12)
    with torch.no_grad():
        A_f, d_f = flat.normalize(cs["S"], cs["vis"], cs["K_kv"], cs["Q"], State(), logits_override=cs["logits"], tau_j_override=cs["tau_j"], tau_i_override=cs["tau_i"])
        A_h, d_h = hier.normalize(cs["S"], cs["vis"], cs["K_kv"], cs["Q"], State(), logits_override=cs["logits"], tau_j_override=cs["tau_j"], tau_i_override=cs["tau_i"])
    assert torch.allclose(A_f, A_h, atol=1e-6) and torch.equal(d_f.kstar, d_h.kstar)     # K_ret >= k*: exact
    e8_flat = e8_summary_from_diag(d_f, cs["vis"], K_ret=None)
    assert e8_flat["truncation_flag"] is None, "no hierarchy ran: the flag must not claim truncation"
    e8_hier = e8_summary_from_diag(d_h, cs["vis"], K_ret=64)
    assert e8_hier["truncation_flag"] in (True, False)
    s = torch.tensor([3.0, 2.0, -1.0]); T = torch.tensor([1, 1, 0], dtype=torch.bool); E = torch.ones(3, dtype=torch.bool)
    p = sparsemax_masked(s[None], E[None])[0]
    assert column_fidelity(s, E, p, p, 1.0, T, kstar=60, K_ret=None)["truncated"] is False
    assert column_fidelity(s, E, p, p, 1.0, T, kstar=60, K_ret=64)["truncated"] is True


def test_f4_hotpot_joint_uses_one_index_space():
    """F-4: the joint metrics compared predicted PASSAGE SLOTS against range(len(gold_slots)).  Unless the gold slots
    happened to be 0 and 1 the intersection was empty, so the joint cell read 0 for every arm even on a perfect
    prediction -- and the `if sup[3] else []` collapse threw away all partial credit."""
    from marsea.data.qa import hotpot_joint
    gold = {3, 7}
    perfect = hotpot_joint("paris", "Paris", {3, 7}, gold)
    assert perfect["sp_em"] == 1.0 and perfect["sp_f1"] == 1.0 and perfect["joint_em"] == 1.0 and perfect["joint_f1"] == 1.0
    half = hotpot_joint("paris", "Paris", {3}, gold)                       # partial credit survives
    assert half["sp_em"] == 0.0 and 0 < half["sp_f1"] < 1 and 0 < half["joint_f1"] < 1
    wrong = hotpot_joint("paris", "Paris", {1, 2}, gold)
    assert wrong["sp_f1"] == 0.0 and wrong["joint_f1"] == 0.0
    # the old form: gold passed as range(len(gold_slots)) = {0, 1} against a correct prediction {3, 7}
    assert hotpot_joint("paris", "Paris", {3, 7}, {0, 1})["sp_f1"] == 0.0


def test_f6_one_b0_bisection_and_one_candidate_rule(rng):
    """F-6: phase_b_init re-implemented the b0 bisection and did not drop the D-31 sink pairs, so realised coverage sat
    below rho_0 and the README's coverage number was not the run's.  Both callers now share candidate_logits/bisect_b0."""
    from marsea.relation import candidate_logits, bisect_b0, sink_pair_mask, RelationHead
    from marsea.normalizer import MarSeaNormalizer, State
    torch.manual_seed(0)
    n_q = n_k = 24
    head = RelationHead(D, R)
    K = torch.randn(1, 1, n_k, D); Q = torch.randn(1, 1, n_q, D)
    i = torch.arange(n_q)[:, None]; j = torch.arange(n_k)[None, :]
    vis = (j <= i)[None, None]
    raw = head.raw(K, Q)
    vals = candidate_logits(raw, vis)
    assert vals.numel() == int((vis.expand_as(raw) & ~sink_pair_mask(n_q, n_k, raw.device)).sum())
    # the two calibration paths agree exactly on b0 ...
    b_direct = head.calibrate_b0(K, Q, vis, rho0=0.05)
    b_pooled = head.calibrate_b0(None, None, None, rho0=0.05, raw=vals)     # the phase_b_init path
    # b0 is an fp32 Parameter, so the stored value is the fp32 rounding of the bisection's double: agreement is
    # exact between the two callers and to fp32 precision against the raw bisection
    assert b_direct == b_pooled and abs(bisect_b0(vals, 0.05) - b_direct) < 1e-5
    # ... and the realised coverage of the running model matches the calibration target
    norm = MarSeaNormalizer(D, R); norm.relation.load_state_dict(head.state_dict())
    norm.relation.calibrate_b0(K, Q, vis, rho0=0.05)
    with torch.no_grad():
        _, d = norm.normalize(torch.randn(1, 1, n_q, n_k).masked_fill(~vis, float("-inf")), vis, K, Q, State())
    cand = vis.expand_as(d.E) & ~sink_pair_mask(n_q, n_k, d.E.device)
    assert abs(float(d.E[cand].float().mean()) - 0.05) < 0.02
    # counting the sink pairs (the old rule) would have solved for a different b0 on this batch
    all_vis = raw[vis.expand_as(raw)]
    assert abs(bisect_b0(all_vis, 0.05) - b_direct) > 1e-6


def test_f7_spec_version_matches_what_the_code_implements():
    """F-7: every checkpoint and run README stamps SPEC_VERSION; it read 4.6 while the code implements D-31, D-27 and
    D-9a (v4.9)."""
    from marsea import SPEC_VERSION
    assert SPEC_VERSION == "4.9"
    from marsea.relation import sink_pair_mask                      # D-31 is in
    assert sink_pair_mask(4, 4, "cpu")[0].all()
    from marsea.data.ruler import RULERExample
    assert "T_j_kind" in RULERExample.__dataclass_fields__          # D-9a is in


def test_f8_head_subselection_and_field_filter_preserve_the_numbers(rng):
    """F-8: teacher_forced_pass kept nine [1, H, T, T] fp32 tensors per example (~29 GB at T = 8K).  Keeping one head --
    or only the fields the consumer reads -- must not change any reported quantity."""
    from marsea.backbone import MarSeaContext
    from marsea.normalizer import Diagnostics
    from marsea.evaluate import _kept_head
    H, T = 12, 7
    full = Diagnostics(A=torch.randn(1, H, T, T), E=torch.rand(1, H, T, T) > 0.5, tau_i=torch.randn(1, H, T),
                       supp_rel=torch.randint(0, 3, (1, H, T)))
    h_star = 5
    assert _kept_head(full, h_star) == h_star                        # nothing sub-selected: index as before
    sub = Diagnostics(A=full.A[:, h_star:h_star + 1], E=full.E[:, h_star:h_star + 1],
                      tau_i=full.tau_i[:, h_star:h_star + 1], supp_rel=full.supp_rel[:, h_star:h_star + 1],
                      extra={"head_sub": h_star})
    hh = _kept_head(sub, h_star)
    assert hh == 0
    assert torch.equal(sub.A[0, hh], full.A[0, h_star]) and torch.equal(sub.E[0, hh], full.E[0, h_star])
    assert torch.equal(sub.tau_i[0, hh], full.tau_i[0, h_star])
    assert sub.A.numel() * H == full.A.numel()                       # H-fold smaller
    ctx = MarSeaContext()
    assert ctx.keep_dense_head is None and ctx.keep_dense_fields is None     # off by default


def test_e6_driver_scores_by_tier_as_a_slope(tmp_path):
    """E6 has a driver and reports accuracy against conflicting-tier count as a SLOPE (paper Sec. 5: never a pooled
    mean).  The loader keeps only records that carry a tier count."""
    import json
    from marsea.data.iheval import load_iheval, accuracy_by_tier
    recs = [dict(system="S", turns=[{"role": "user", "content": "u"}], reference="alpha", num_tiers=1),
            dict(system="S", user="u2", answer="beta", metadata={"n_conflicts": 3}),
            dict(system="S", user="u3", answer="gamma")]                      # no tier count: dropped
    f = tmp_path / "ih.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in recs))
    exs = load_iheval(f)
    assert len(exs) == 2 and {e.tiers for e in exs} == {1, 3}
    assert exs[0].prompt.startswith("System: S\n\n") and exs[0].prompt.rstrip().endswith("Assistant:")
    out = accuracy_by_tier([dict(tiers=1, correct=1.0), dict(tiers=1, correct=1.0),
                            dict(tiers=3, correct=0.0), dict(tiers=3, correct=1.0)])
    assert out["accuracy_by_tier"] == {1: 1.0, 3: 0.5}
    assert abs(out["slope"] - (-0.25)) < 1e-9                                 # the reported quantity is the slope


@pytest.mark.parametrize("hb", [1, 2, 3])
def test_head_blocking_is_arithmetically_identical(hb, rng):
    """Every program is per Q-head (spec Sec. 6.2), so processing the heads in blocks is the head-axis analogue of the
    key chunking of Sec. 6.6: identical numbers, and it divides the [B, H, n_q, n_k] transients that dominate memory.
    Measured payoff at T = 1536 with four patched layers: 8.0 -> 4.2 GB of activations (per-block checkpointing on)."""
    from marsea.normalizer import MarSeaNormalizer, State
    torch.manual_seed(0)
    full = MarSeaNormalizer(D, R).double()
    blk = MarSeaNormalizer(D, R, head_block=hb).double(); blk.load_state_dict(full.state_dict())
    worst = 0.0
    for _ in range(8):
        cs = rand_case(rng, n_q=18, n_k=10, B=2, Hkv=2, H=6)
        s1, s2 = State(), State()
        with torch.no_grad():
            A1, d1 = full.normalize(cs["S"].double(), cs["vis"], cs["K_kv"].double(), cs["Q"].double(), s1)
            A2, d2 = blk.normalize(cs["S"].double(), cs["vis"], cs["K_kv"].double(), cs["Q"].double(), s2)
        assert torch.equal(d1.E, d2.E) and torch.equal(d1.kstar, d2.kstar) and torch.equal(d1.supp_rel, d2.supp_rel)
        worst = max(worst, float((A1 - A2).abs().max()), float((s1.nu_next - s2.nu_next).abs().max()))
        for n in ("cbar_j", "tau_j", "nu", "cbar_i", "tau_i", "Atil", "a1", "p", "theta", "Rtil"):
            worst = max(worst, float((getattr(d1, n) - getattr(d2, n)).abs().max()))
    assert worst < 1e-12, f"head blocking changed the numbers by {worst:.2e}"


def test_head_blocking_gradients_agree(rng):
    from marsea.normalizer import MarSeaNormalizer, State
    torch.manual_seed(1)
    full = MarSeaNormalizer(D, R).double()
    blk = MarSeaNormalizer(D, R, head_block=2).double(); blk.load_state_dict(full.state_dict())
    cs = rand_case(rng, n_q=14, n_k=8, B=1, Hkv=2, H=4)
    w = torch.randn(1, 4, 14, 8, dtype=torch.float64)
    grads = {}
    for name, mod in (("full", full), ("blk", blk)):
        mod.zero_grad()
        S = cs["S"].double().clone().requires_grad_(True)
        A, _ = mod.normalize(S, cs["vis"], cs["K_kv"].double(), cs["Q"].double(), State())
        (A * w).sum().backward()
        grads[name] = {n: p.grad.clone() for n, p in mod.named_parameters() if p.grad is not None}
        grads[name]["S"] = S.grad.clone()
    for k in grads["full"]:
        a, b = grads["full"][k], grads["blk"][k]
        assert torch.allclose(a, b, atol=1e-12, rtol=1e-9), f"gradient of {k} differs: {(a - b).abs().max():.2e}"


def test_inv3_is_stated_against_the_rows_own_softmax_mass():
    """INV-3 used to read `sum_j A <= 1 + tol` with tol a slack constant: that cannot be checked kernel-free, because a
    non-binding row carries its fp32 softmax total, which the device's kernel puts up to n_k eps / 2 above one (review
    e16a843).  It now reads sum_j A <= sum_j A_sm + CAP_TOL + tol_cap + tol_sum in fp64 (and INV-3a on a1 without
    Stage 2's tol_sum): a row with a softmax total 1e-4 over one (the CPU razor edge) passes, a row carrying 10 CAP_TOL
    MORE than its softmax gave it fails."""
    from marsea.invariants import invariant_report
    from marsea.normalizer import Diagnostics, CAP_TOL
    n_k = 64
    A_sm = torch.zeros(1, 1, 2, n_k, dtype=torch.float64)
    A_sm[0, 0, 0, :] = (1.0 + 1e-4) / n_k                      # what a lane-wise kernel can hand back for a softmax row
    A_sm[0, 0, 1, :] = 0.5 / n_k
    def diag(A):
        return Diagnostics(E=torch.zeros(1, 1, 2, n_k, dtype=torch.bool), Atil=A.clone(), A_sm=A_sm.clone(), a1=A.clone(),
                           cbar_j=torch.zeros(1, 1, n_k, dtype=torch.float64), cbar_i=torch.zeros(1, 1, 2, dtype=torch.float64),
                           tau_j=torch.ones(1, 1, n_k, dtype=torch.float64), tau_i=torch.ones(1, 1, 2, dtype=torch.float64),
                           nu=torch.ones(1, 1, n_k, dtype=torch.float64))
    vis = torch.ones(1, 1, 2, n_k, dtype=torch.bool)
    rep = invariant_report(A_sm.clone(), diag(A_sm), vis)
    assert rep["INV-3"][0], f"a row carrying exactly its (kernel-inflated) softmax mass must pass INV-3 ({rep['INV-3'][1]:.3e})"
    A2 = A_sm.clone(); A2[0, 0, 0, :] += 10 * CAP_TOL / n_k     # ten times the excess the cap leaves alone
    rep2 = invariant_report(A2, diag(A2), vis)
    assert not rep2["INV-3"][0] and not rep2["INV-3a"][0], "INV-3 must still catch a row carrying more than its softmax mass"


def test_inv5_recomputes_the_binding_set_independently_and_the_cap_uses_no_slack():
    """INV-5 used to hard-code 1.0 + 1e-6, then the cap's slack constant; both drifted from the cap.  The cap now binds on
    relation_excess (sum_E (Atil - A_sm) in fp64) and INV-5 recomputes that criterion from the FULL row in fp64 -- a
    different code path, the same mathematics (Atil == A_sm off E)."""
    import inspect
    from marsea import invariants, normalizer
    src = inspect.getsource(invariants.invariant_report)
    # the check uses the MASKED fp64 form; the mechanism (row_masses) differences two unmasked fp64 sums (review 9bacef9 B-1)
    assert "_masked_excess64(Atil, A_sm, E)" in src and "row_masses(" not in src and "1.0 + 1e-6" not in src and "cap_slack" not in src
    assert "* E[..., J]" in inspect.getsource(invariants._masked_excess64)
    assert not hasattr(normalizer, "cap_slack") and normalizer.CAP_TOL == 1e-6
    src_n = inspect.getsource(normalizer.MarSeaNormalizer._normalize_one)
    assert "row_masses(Atil, A_sm, E)" in src_n and "unit_cap(Atil, vis, excess, sm_mass)" in src_n and "Atil.sum(-1) > 1" not in src_n


def test_the_unit_cap_is_the_identity_on_a_real_softmax_row_at_length():
    """THE reason 5(a) started failing at 2K: a softmax row's fp32 total is not one, and a cap that tested the total bound
    on rows nothing pushed over a unit.  With the decision taken from the relation, an EMPTY relation gives excess exactly
    0.0 and the cap is the identity bitwise -- whatever the kernel's row sums are."""
    import torch
    from marsea.primitives import proj_le_masked
    from marsea.normalizer import relation_excess, cap_binds_from_excess
    torch.manual_seed(0)
    for n_k in (2048, 8192):
        S = (torch.randn(1, 2, 256, n_k) * 2).cuda() if torch.cuda.is_available() else torch.randn(1, 2, 256, n_k) * 2
        A = torch.softmax(S, -1)
        vis = torch.ones_like(A, dtype=torch.bool)
        err = float((A.sum(-1) - 1).abs().max())
        ex = relation_excess(A, A, torch.zeros_like(vis))
        assert torch.equal(ex, torch.zeros_like(ex))
        a1, _ = proj_le_masked(A, torch.ones(1, 2, 256, device=A.device), vis, binds=cap_binds_from_excess(ex))
        assert torch.equal(a1, A), f"the cap bound on a plain softmax row at n_k = {n_k} (row-sum error {err:.2e})"
        a1_old, _ = proj_le_masked(A, torch.ones(1, 2, 256, device=A.device), vis)      # the sum-based test, for contrast
        if err > 0:
            assert not torch.equal(a1_old, A), f"n_k = {n_k} no longer exercises the old failure (error {err:.2e})"


def test_e5_task_metrics_reach_the_table():
    """aggregate_ruler read only exact_set / ruler_recall plus the attention block, so ans_f1, support_* and joint
    were written per row and never aggregated -- the E5 table reported exact_set_acc = nan and nothing else
    (review d5bd980 F)."""
    from marsea.evaluate import aggregate_ruler
    rows = [dict(m=2, ans_f1=0.5, ans_em=0.0, support_P=0.9, support_R=0.7, support_F1=0.8, support_EM=1.0, joint=0.4),
            dict(m=2, ans_f1=0.7, ans_em=1.0, support_P=0.5, support_R=0.5, support_F1=0.6, support_EM=0.0, joint=0.2)]
    t = aggregate_ruler(rows, stratify="m")[2]
    assert abs(t["ans_f1"] - 0.6) < 1e-9 and abs(t["ans_em"] - 0.5) < 1e-9
    assert abs(t["support_F1"] - 0.7) < 1e-9 and abs(t["joint"] - 0.3) < 1e-9
    assert abs(t["support_EM"] - 0.5) < 1e-9


def test_the_gates_are_gates():
    """preflight's "peak < 72 GB" compared nothing, profile_memory swallowed OOM and exited 0, time_step printed the
    D-23 rule and exited 0, and verify_all had no verdict at all (review d5bd980 F)."""
    import pathlib, subprocess, sys
    root = pathlib.Path(__file__).resolve().parents[1]
    pf = (root / "scripts/preflight.sh").read_text()
    assert "--gate_GB $GATE_GB" in pf and "GATE FAILED" in pf
    assert "|| true" not in pf, "preflight still swallows a failure"
    pm = (root / "scripts/profile_memory.py").read_text()
    assert 'sys.exit(3)' in pm and "--head_block" in pm and "--gate_GB" in pm
    assert "marginal_per_patched_layer_GB" in pm, "the per-layer term is still activation / n_layers"
    ts = (root / "scripts/time_step.py").read_text()
    assert "sys.exit(2)" in ts and "--no_gate" in ts
    va = (root / "scripts/verify_all.py").read_text()
    assert "TOTAL VIOLATIONS" in va and "sys.exit(1)" in va
    es = (root / "scripts/run_evalsuite.sh").read_text()
    assert "B1 B4" in es and "runs/e9_*/" in es and "run_e6.py" in es, "B1/B4/E9/E6 still have no evaluation"


def test_iheval_reports_its_drops():
    import json, tempfile, pathlib
    from marsea.data.iheval import load_iheval
    recs = [dict(id="a", system="s", user="u", reference="r", num_tiers=1), dict(id="b", system="s", user="u", reference="r")]
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "ih.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in recs))
        st = {}
        exs = load_iheval(p, st)
    assert len(exs) == 1 and st == dict(n_records=2, n_kept=1, n_dropped_no_tier=1)


def test_san1_and_inv9_carry_the_summation_tolerance():
    """SAN-1 kept a fixed 1e-6 while INV-1/3/7 were relaxed to a summation tolerance, so a softmax row of 2048
    entries failed it by rounding alone -- invisible to the acceptance tests, which run at T <= 1024, and it stopped
    the first real 2K training step."""
    import torch
    from marsea.invariants import invariant_report
    from marsea.normalizer import MarSeaNormalizer, State
    from marsea.relation import repeat_kv
    torch.manual_seed(0)
    D_, R_, T = 32, 8, 2048
    norm = MarSeaNormalizer(D_, R_)
    Q = torch.randn(1, 2, T, D_) * 1.5; K = torch.randn(1, 1, T, D_) * 1.5
    i = torch.arange(T)[:, None]; j = torch.arange(T)[None, :]
    vis = (j <= i)[None, None]
    with torch.no_grad():
        norm.relation.calibrate_b0(K, Q, vis, 0.05)
        S = (torch.einsum("bhid,bhjd->bhij", Q, repeat_kv(K, 2)) * D_ ** -0.5).masked_fill(~vis, float("-inf"))
        A, d = norm.normalize(S, vis, K, Q, State())
    rep = invariant_report(A, d, vis)
    assert rep["SAN-1"][0], f"SAN-1 failed at T = {T} by fp32 rounding alone: worst {rep['SAN-1'][1]:.3e}"
    assert rep["SAN-1"][1] > 1e-7, "the test is not exercising the regime it was written for"
    assert all(ok for ok, _ in rep.values()), {k: v for k, v in rep.items() if not v[0]}


def test_queue_scripts_observe_failures_and_run_every_arm():
    """A bare `wait` returns 0 whatever the jobs did, so `set -e` never fired and run_s2 ran its whole Phase-B
    section after every arm had crashed; two E9 arms were off by default; E3depth was generated and evaluated by
    nothing; E6's --out named a file where run_e6 makes a directory (review 30372ae D)."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    s2 = (root / "scripts/run_s2.sh").read_text()
    ev = (root / "scripts/run_evalsuite.sh").read_text()
    assert "barrier()" in s2 and 'wait "${PIDS[$i]}"' in s2, "run_s2.sh still cannot observe a failed job"
    # the eval queue waits per job with `wait -n -p` since e16a843 (work-conserving slots); the simulated queue in
    # test_review_e982f83.py checks the behaviour, not just the text
    assert "barrier()" in ev and 'wait -n -p pid' in ev, "run_evalsuite.sh still cannot observe a failed job"
    for src, name in ((s2, "run_s2.sh"), (ev, "run_evalsuite.sh")):
        assert "FAILED_JOBS" in src, name
    assert 'per_head\\":12' in s2 and "EXTENDED_E9" not in s2.split('per_head\\":12')[1].split("\n")[0], \
        "the per_head arm is still behind EXTENDED_E9"
    assert "e9_hierarchical_K64" in s2.split("DENSE_COMMON=")[1], "the K_ret arm is not in the dense group"
    assert "UNIFORM_L:-$L" in s2, "the dense E9 arms still train at a different length from their evaluation"
    assert "E3depth" in ev and "--stratify depth" in ev, "E3depth is still generated and never evaluated"
    assert '--tag "e6_${arm}" --out runs/eval' in ev, "E6's --out still names a file where run_e6 creates a directory"
    pf = (root / "scripts/preflight.sh").read_text()
    assert "--modes train --chunk $CHUNK \\\n    --head_block $HEAD_BLOCK --targets $L --gate_GB" in pf, \
        "preflight still gates training on a 16K extrapolation nothing trains at"
    assert "detector.json.superseded" in pf, "preflight still keeps a stale detector"
    ts = (root / "scripts/time_step.py").read_text()
    assert "--head_block" in ts, "D-23 is still measured on a forward the queue does not run"


def test_verify_all_reports_evaluations_not_rows():
    """The paper quotes 1,058,450 checks; the harness printed "0 over 62 counted checks", naming two quantities a
    factor of 17,000 apart with the same word.  Structural rows (no trial count) must gate too (review 30372ae D)."""
    import pathlib
    va = (pathlib.Path(__file__).resolve().parents[1] / "scripts/verify_all.py").read_text()
    assert "evaluations in" in va and "_uncounted" in va
    assert "sum(v for _, v, _ in _counted) + sum(v for _, v, _ in _uncounted)" in va


def test_detector_is_stamped_and_its_loader_tolerates_the_s0_keys():
    from marsea.detector import DetectorResult
    import json, pathlib, tempfile
    r = DetectorResult(scores=[[0.5]], patched_layers=[0], l_star=0, h_star=0, n_prompts=1, threshold=0.1,
                       ranking=[[0.5, 0, 0]])
    r.stamp("Qwen/Qwen2.5-1.5B")
    assert r.spec_version and r.created and r.backbone == "Qwen/Qwen2.5-1.5B"
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "d.json"
        r.save(p)
        raw = json.loads(p.read_text()); raw["s0_licence"] = {"coref": {"site": [1, 2]}}; p.write_text(json.dumps(raw))
        back = DetectorResult.load(p)           # used to raise: run_s0_sweep adds keys the dataclass does not declare
    assert back.spec_version == r.spec_version and back.extra["s0_licence"]["coref"]["site"] == [1, 2]


def test_both_T_j_kinds_can_name_their_own_site():
    """`value` had no way through, so a kind licensed only by usable_anywhere was still scored at (l*, h*) and
    reported as licensed -- the gate was half-fixed (review 30372ae D)."""
    from marsea.evaluate import kind_site_map
    assert kind_site_map(coref_site=(14, 5)) == {"coref": (14, 5)}
    assert kind_site_map(value_site=(19, 7)) == {"value": (19, 7)}
    assert kind_site_map((14, 5), (19, 7)) == {"coref": (14, 5), "value": (19, 7)}
    assert kind_site_map() == {}
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "scripts/run_eval.py").read_text()
    assert '("coref", args.coref_site), ("value", args.value_site)' in src, "run_eval still reads only the coref site"
    ev = (pathlib.Path(__file__).resolve().parents[1] / "marsea/evaluate.py").read_text()
    assert "def attention_level_qa" in ev and "kind_sites=sites" in ev, "E5 still has no per-kind provenance"
