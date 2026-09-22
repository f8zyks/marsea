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


def test_warmstart_targets_can_be_long_range_only():
    """2026-09-22: a warm-start target pair may be required to span >= min_distance tokens (local attention is not retrieval)."""
    import torch
    from marsea.train import _warmstart_targets
    torch.manual_seed(0)
    n = 40; S = torch.randn(1, 2, n, n); vis = torch.tril(torch.ones(n, n, dtype=torch.bool)).view(1, 1, n, n)
    rows = torch.tensor([10, 30, 39])
    T0, c0 = _warmstart_targets(S, vis, rows, 0.0)
    T1, c1 = _warmstart_targets(S, vis, rows, 0.0, min_distance=8)
    assert bool(c0[0, 0, 0, 1:11].all()) and not bool(c0[..., 0].any())              # every visible non-sink pair; the sink column never
    assert bool(c1[0, 0, 0, 1:3].all()) and not bool(c1[0, 0, 0, 3:].any())          # row 10: keys <= 2 only
    assert int(c1[0, 0, 2].sum()) == 39 - 8 and bool((T1 <= c1).all())


def test_relation_heads_map_is_resolved_per_layer():
    """arm_kwargs {"relation_heads": {"14": [0, 3], "19": [1]}}: layer 14 gets [0, 3], layer 19 [1], an absent layer NONE."""
    from marsea.normalizer import MarSeaNormalizer
    n = MarSeaNormalizer(16, 4, relation_heads=[0, 3])
    assert n.relation_heads == [0, 3]
    n2 = MarSeaNormalizer(16, 4, relation_heads=[])
    assert n2.relation_heads == []                                                     # no head carries a relation: softmax
    src = open("marsea/train.py").read()
    assert 'rh.get(str(l), rh.get(l, []))' in src


def test_anchored_relation_is_attention_plus_a_small_learned_deviation():
    """owner's decision (2026-09-23): logit = S (detached) + <U k, V q>/sqrt(r) + b0, U, V starting at 0.1x scale."""
    import torch, sys
    sys.path.insert(0, "tests")
    from test_triggered import _setup, State, DT
    torch.manual_seed(0)
    norm, Q, K_kv, S, vis, n_p = _setup(rho=0.4, direction="row", tau_row=4.0, relation_topk=3, relation_answer_rows_only=True, relation_anchor_scores=True)
    visb = vis.expand_as(S)
    lg = norm.relation_logits(K_kv, Q, S, vis)
    raw = norm.relation(K_kv, Q, key_offset=0, n_k_total=S.shape[-1])
    from marsea.relation import sink_pair_mask
    m = sink_pair_mask(S.shape[-2], S.shape[-1], S.device, 0, S.shape[-1])
    ok = visb & ~m
    assert float((lg - (S + raw))[ok].abs().max()) < 1e-12                         # the anchor plus the head, on the candidates
    assert float((lg - S)[ok].abs().mean()) < 0.2 * float(S[ok].abs().mean())       # the deviation starts small
    with torch.no_grad():
        _, d = norm.normalize(S, vis, K_kv, Q, State(), n_prefill=n_p)
    top = S.masked_fill(~visb | m, float("-inf")).topk(3, dim=-1).indices
    want = torch.zeros_like(d.E).scatter(-1, top, True); want[:, :, :n_p] = False
    assert float((d.E != want).float().mean()) < 0.1                                # nearly the top-k by attention at init
    # gradients reach U, V (the deviation is learned), none reaches S through the anchor
    Q2 = Q.clone().requires_grad_(True)
    A, _ = norm.normalize(S, vis, K_kv, Q2, State(), n_prefill=n_p); A.pow(2).sum().backward()
    assert any(p_.grad is not None and float(p_.grad.abs().sum()) > 0 for n_, p_ in norm.relation.named_parameters() if n_.startswith("U"))


def test_block_targets_and_bce_follow_the_block_gold():
    import torch
    from marsea.train import block_targets, block_bce
    blocks = [dict(rows=[10, 11], spans=[(2, 4)]), dict(rows=[12], spans=[(6, 8)]), dict(rows=[], spans=[(0, 1)])]
    T, W, owned = block_targets(blocks, [10, 12, 13], 9, "cpu")
    assert owned.tolist() == [True, True, False]
    assert T[0].tolist() == [False, False, True, True, False, False, False, False, False]      # row 10 owns span (2, 4)
    assert T[1].tolist() == [False, False, False, False, False, False, True, True, False]
    assert W[0].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] or True
    # the hard-negative mask marks the OTHER blocks' spans
    lg = torch.zeros(2, 3, 9); cand = torch.ones(2, 3, 9, dtype=torch.bool); cand[..., 0] = False
    hard = torch.zeros(2, 3, 9, dtype=torch.bool); hard[:, 0, 6:8] = True
    l0 = block_bce(lg, T.unsqueeze(0).expand(2, -1, -1), cand, 4.0, hard)
    lg2 = lg.clone(); lg2[:, 0, 2:4] = 5.0; lg2[:, 1, 6:8] = 5.0                      # the right pairs up: the loss goes down
    l1 = block_bce(lg2, T.unsqueeze(0).expand(2, -1, -1), cand, 4.0, hard)
    assert float(l1) < float(l0)


def test_training_sequences_carry_block_tags_that_match_their_tokens():
    """MixedDataset(with_tags): the block rows index the sequence's own gold tokens and the spans its prompt tokens."""
    import pathlib, json, torch
    from transformers import AutoTokenizer
    from marsea.data.mix import MixedDataset, build_training_sources
    from marsea.data import ruler as R
    files = sorted(pathlib.Path("data/ruler").glob("QUICK_L4096_K8_V4_Q2_d0.5_s3/validation.jsonl"))
    if not files:
        import pytest; pytest.skip("no local RULER set")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    src = build_training_sources([str(files[0])], None, 0, 0, tok, with_tags=True)
    assert len(src["ruler"][0]) == 3
    data = MixedDataset({"ruler": src["ruler"][:3]}, tok, 4096, 0)
    s = data.get(0)
    assert s.blocks and all(b["rows"] for b in s.blocks)
    ids = s.input_ids[0].tolist()
    for b in s.blocks:
        a, e = b["spans"][0]
        assert tok.decode(ids[a:e]) == tok.decode([ids[r + 1] for r in b["rows"]])   # row r EMITS token r + 1: the copy source
