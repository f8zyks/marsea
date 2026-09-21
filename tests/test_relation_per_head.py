"""The 2026-09-20 relation head: one (U_h, V_h, b0_h) per Q-head, calibrated per head, with an annealed straight-through
temperature.  The shared head's single b0 met rho_0 on average and on no head (layer 24: 7-13 % on one KV group, under
0.6 % on the other, at calibration), and one U could not serve two key spaces."""
import copy, torch, pytest
from marsea.normalizer import MarSeaNormalizer, State
from marsea.chunked import marsea_chunked_attention
from marsea.relation import RelationHead, straight_through
from test_t11_chunked import _case, _dense, D, R


def _norm(H, dtype=torch.float64, **kw):
    torch.manual_seed(0)
    return MarSeaNormalizer(D, R, relation_per_head=H, **kw).to(dtype)


def test_every_head_sits_at_rho0_after_calibration_and_the_shared_head_does_not():
    Q, K_kv, V_kv, vis = _case(160, B=1, Hkv=2, g=3, dtype=torch.float64)
    Q[:, 3:] *= 4.0                                                        # the second KV group's queries live on another scale
    per = _norm(6); shared = MarSeaNormalizer(D, R).double()
    with torch.no_grad():
        per.relation.calibrate_b0(K_kv, Q, vis, 0.10); shared.relation.calibrate_b0(K_kv, Q, vis, 0.10)
        cov = lambda nm: [nm.relation.coverage(nm.relation(K_kv, Q)[:, h:h + 1], vis) for h in range(6)]
        cp, cs = cov(per), cov(shared)
    assert per.relation.b0.shape == (6,) and shared.relation.b0.dim() == 0
    assert max(abs(c - 0.10) for c in cp) < 0.01, cp
    assert max(cs) - min(cs) > 0.05, cs                                    # one threshold: the heads are nowhere near each other


@pytest.mark.parametrize("hb", [None, 2])
def test_chunked_equals_dense_with_the_per_head_relation(hb):
    norm = _norm(6, head_block=hb); norm.st_temperature = 4.0
    Q, K_kv, V_kv, vis = _case(200, B=1, Hkv=2, g=3, dtype=torch.float64)
    with torch.no_grad():
        norm.relation.calibrate_b0(K_kv, Q, vis, 0.3)
        nu = torch.rand(1, 6, 200, dtype=torch.float64) * 3 + 1
        O_d, A_d, dd = _dense(norm, Q, K_kv, V_kv, vis, State(nu_prev=nu))
        O_c, dc = marsea_chunked_attention(norm, Q, K_kv, V_kv, vis, State(nu_prev=nu), chunk=32, dense_outputs=True)
    assert (dc.E == dd.E).all() and float((O_c - O_d).abs().max()) < 1e-9 and float((dc.A - A_d).abs().max()) < 1e-9
    assert dc.extra["pass2_rebuilt_row_frac"] > 0                           # the rebuilt rows read b0_of(h): exercised


def test_gradients_agree_between_the_paths_at_temperature_4_and_reach_every_b0():
    Q, K_kv, V_kv, vis = _case(96, B=1, Hkv=2, g=3, dtype=torch.float64, seed=3)
    grads = []
    for path in ("dense", "chunked"):
        norm = _norm(6); norm.st_temperature = 4.0
        with torch.no_grad():
            norm.relation.calibrate_b0(K_kv, Q, vis, 0.3)
        if path == "dense":
            O, _, _ = _dense(norm, Q, K_kv, V_kv, vis, State())
        else:
            O, _ = marsea_chunked_attention(norm, Q, K_kv, V_kv, vis, State(), chunk=32)
        (O * torch.linspace(0.5, 1.5, O.numel(), dtype=O.dtype).view_as(O)).sum().backward()
        grads.append({n: p.grad.clone() for n, p in norm.named_parameters() if p.grad is not None})
    for n in grads[0]:
        assert torch.allclose(grads[0][n], grads[1][n], atol=1e-8, rtol=1e-6), n
    assert (grads[0]["relation.b0"].abs() > 0).all()                        # one gradient per head


def test_temperature_changes_the_backward_and_never_the_forward():
    logits = torch.tensor([[-6.0, -1.0, 0.5, 3.0]], dtype=torch.float64, requires_grad=True)
    vis = torch.ones_like(logits, dtype=torch.bool); E = (logits > 0) & vis
    out = {}
    for T in (1.0, 4.0):
        g = straight_through(E, logits, vis, T)
        assert torch.equal(g.detach(), E.double())
        (gr,) = torch.autograd.grad(g.sum(), logits); out[T] = gr
    assert out[4.0][0, 0] > 10 * out[1.0][0, 0]                              # at logit -6: 0.037 against 0.0025


def test_state_dict_round_trip_and_the_e9_arm_keeps_its_scalar():
    a = _norm(6); b = _norm(6)
    with torch.no_grad(): a.relation.b0.copy_(torch.arange(6.0))
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    assert torch.equal(b.relation.b0, a.relation.b0)
    assert MarSeaNormalizer(D, R, per_head=6).relation.b0.dim() == 0        # 2026-09's e9_per_head checkpoints still load
    with pytest.raises(AssertionError):
        RelationHead(D, R, per_head=0, per_head_b0=True)


# ------------------------------------------------------------------------------------------ the trainer's side of the recipe
def test_recipe_knobs_are_off_by_default_and_the_schedules_are_what_they_say():
    from marsea.train import TrainConfig, st_temperature, module_warmup
    off = TrainConfig()
    assert (off.relation_warmstart_steps, off.recal_every, off.st_T0, off.st_anneal_steps, off.module_warmup_steps,
            off.gate_min_relation_recall) == (0, 0, 1.0, 0, 0, 0.0)
    assert st_temperature(700, off) == 1.0 and module_warmup(500, off) == 1.0          # the 2026-09 grid, unchanged
    cfg = TrainConfig(phase_a_steps=500, st_T0=4.0, st_anneal_steps=500, module_warmup_steps=50)
    assert st_temperature(500, cfg) == 4.0 and abs(st_temperature(750, cfg) - 2.5) < 1e-12 and st_temperature(1000, cfg) == 1.0
    assert st_temperature(2400, cfg) == 1.0
    assert abs(module_warmup(500, cfg) - 1 / 50) < 1e-12 and module_warmup(549, cfg) == 1.0 and module_warmup(2000, cfg) == 1.0


def test_warmstart_targets_are_high_mass_pairs_without_the_sink_column():
    from marsea.train import _warmstart_targets
    T_ = 12
    S = torch.full((1, 2, T_, T_), -5.0); vis = torch.tril(torch.ones(T_, T_, dtype=torch.bool))[None, None]
    S[0, 0, 9, 4] = 5.0; S[0, 1, 9, 0] = 9.0                                # head 0 attends key 4; head 1 attends the SINK
    S = S.masked_fill(~vis, float("-inf"))
    T, cand = _warmstart_targets(S, vis, torch.tensor([9]), 0.10)
    assert T.shape == (1, 2, 1, T_) and bool(T[0, 0, 0, 4]) and int(T[0, 0].sum()) == 1
    assert not bool(T[0, 1].any()) and not bool(cand[..., 0].any())        # the sink is neither a target nor a candidate


def test_quick_eval_reports_the_relation_at_the_answer_rows(monkeypatch):
    import marsea.evaluate as ev
    rows = [dict(tf_loss=0.1, exact_set=True, ruler_recall=0.75, m=4, e8={}, attention=dict(rows=[dict(rel_recall=1.0, E_size=40, Rtil=0.6, supp_rel=3),
                                                                        dict(rel_recall=0.0, E_size=0, Rtil=0.0, supp_rel=0)]))]
    monkeypatch.setattr(ev, "evaluate_ruler", lambda *a, **k: rows)
    monkeypatch.setattr(ev, "aggregate_ruler", lambda r, stratify=None: {None: dict(interval_hit_rate=None, row_precision=1.0, row_recall=None,
                                                                                   excluded_by_stage1=None, rejected_by_row=None)})
    res = ev.quick_eval_factory([object()], [], None, 23, 2, "marsea")(None, None, 750)
    assert res["relation_recall_row"] == 0.5 and res["relation_size_row"] == 20 and res["relation_mass_row"] == 0.3
    assert res["relation_support_nonempty"] == 0.5
    assert res["ruler_recall"] == 0.75 and res["ruler_recall_by_m"] == {"4": 0.75}


def test_per_head_tau_heads_with_the_per_head_relation_and_their_per_head_calibration():
    """the owner's 2026-09-20 decision: TauK and TauQ per Q-head too -- arm_kwargs {"relation_per_head": H, "per_head": H}."""
    from marsea.heads import TauK, column_stats
    norm = MarSeaNormalizer(D, R, relation_per_head=6, per_head=6).double()
    assert norm.relation.per_head_b0 and norm.relation.b0.shape == (6,) and norm.tauK.per_head == 6 and norm.tauQ.per_head == 6
    Q, K_kv, V_kv, vis = _case(160, B=1, Hkv=2, g=3, dtype=torch.float64)
    Q[:, 3:] *= 4.0                                                        # heads 3-5: four times the score spread
    S = (torch.einsum("bhid,bhjd->bhij", Q, K_kv.repeat_interleave(3, dim=1)) * D ** -0.5).masked_fill(~vis, float("-inf"))
    targets = norm.tauK.calibrate(S, vis)
    assert isinstance(targets, list) and len(targets) == 6 and min(targets[:3]) > 2.5 * max(targets[3:])   # 1/std per head
    tau = norm.tauK(K_kv.repeat_interleave(3, dim=1), column_stats(S, vis), torch.ones(1, 6, 160, dtype=torch.float64))
    for h in range(6):
        assert abs(float(tau[0, h].median()) - targets[h]) < 0.05 * targets[h], (h, float(tau[0, h].median()), targets[h])
    # pooled (a tensor of stds) still sets every head alike: the E9 per_head arm of 2026-09 is reproduced
    t = norm.tauK.calibrate(None, None, stds=torch.tensor([0.5, 0.5, 2.0]))
    assert isinstance(t, float) and torch.allclose(norm.tauK.b2, norm.tauK.b2[0].expand_as(norm.tauK.b2))
    # and both paths still agree
    with torch.no_grad():
        norm.relation.calibrate_b0(K_kv, Q, vis, 0.3); norm.tauK.calibrate(S, vis)
        O_d, A_d, dd = _dense(norm, Q, K_kv, V_kv, vis, State())
        O_c, dc = marsea_chunked_attention(norm, Q, K_kv, V_kv, vis, State(), chunk=32, dense_outputs=True)
    assert (dc.E == dd.E).all() and float((O_c - O_d).abs().max()) < 1e-9
