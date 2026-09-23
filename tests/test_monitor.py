"""The relation monitor: per answer row, the ONE key it is about to copy -- on a tiny model, both training forms."""
import math, pathlib, torch, pytest
from types import SimpleNamespace
from test_triggered import _tiny_model


def _example(n_p=30, vals=((5, 8), (12, 15))):
    """a fake RULER example: two 3-token values at prompt positions 5-7 and 12-14, answered in order."""
    g = torch.Generator().manual_seed(7)
    prompt = torch.randint(0, 97, (n_p,), generator=g).tolist()
    gold = []
    rows = []
    for (a, b) in vals:
        rows.append([n_p + len(gold) + k - 1 for k in range(b - a)]); gold += prompt[a:b]
    return SimpleNamespace(prompt_ids=prompt, gold_ids=gold, outputs=["v0", "v1"], value_spans=list(vals), answer_value_rows=rows,
                           answer_rows=[r[0] for r in rows])


def test_copy_sources_pairs_each_answer_row_with_the_token_it_copies():
    from marsea.monitor import copy_sources
    ex = _example()
    assert copy_sources(ex) == [(29, 5, 0, 0), (30, 6, 0, 1), (31, 7, 0, 2), (32, 12, 1, 0), (33, 13, 1, 1), (34, 14, 1, 2)]
    ex.value_spans[1] = (12, 14)                                          # needle and answer tokenise differently: skipped
    assert [c[2] for c in copy_sources(ex)] == [0, 0, 0]


@pytest.mark.parametrize("triggered", [False, True])
def test_monitor_records_are_consistent_with_the_diagnostics(triggered):
    from marsea.monitor import monitor_example, summarise, format_line, ROW_KEYS, COL_KEYS
    model, ctx = _tiny_model(); ctx.triggered = triggered
    recs = monitor_example(model, ctx, _example(), 2, 1, device="cpu")
    assert len(recs) == 6 and all(set(ROW_KEYS + COL_KEYS) <= set(r) for r in recs)
    for r in recs:
        assert 0 <= r["src_mass_softmax"] <= 1 and 0 <= r["tail_kept_frac"] <= 1 and r["relation_size"] >= 0
        assert r["src_in_support"] in (0.0, 1.0) and (r["src_mass_final"] > 0) == bool(r["src_in_support"])
        if r["src_in_relation"]:
            assert r["col_size"] >= 1 and r["row_standing"] <= 1e-12          # the row is a member: at or below the column's max
        else:
            assert math.isnan(r["row_standing"]) and math.isnan(r["tau_used"]) and math.isnan(r["row_share_p"])
        assert abs(r["relation_mass_final"] + r["tail_mass_final"] - (r["relation_mass_final"] + r["tail_mass_final"])) < 1e-12
    if triggered:                                                              # the per-solve tau is reported for triggered rows
        assert any(r["tau_used"] != r["tau_sealed"] for r in recs if r["src_in_relation"])
    res = summarise(recs)
    assert res["all"]["n"] == 6 and set(res["by_m"]) == {"2"} and res["first_token"]["n"] == 2
    assert "src in E" in format_line(500, res) and format_line(1, dict(n=0)).endswith("no rows")
    assert ctx.n_prefill is None and not ctx.collect                           # the pass leaves the context clean


def test_train_calls_the_monitor_on_its_cadence():
    import inspect, marsea.train as tr
    src = inspect.getsource(tr.train)
    assert "cfg.monitor_every and (step - cfg.phase_a_steps) % cfg.monitor_every == 0" in src and "log(dict(step=step, monitor=mres))" in src


def test_a_non_finite_gradient_is_named_by_parameter():
    """P7 (2026-09-21) stopped on 'non-finite gradient norm at step 751' three times with nothing to say where: the check
    now runs per micro-batch and names the parameters; the training loop keeps the weights (nan_step{step}.pt)."""
    import torch
    from marsea.train import nonfinite_grad_names
    m = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Linear(3, 1))
    assert nonfinite_grad_names(m) == []                                       # no gradients at all
    m(torch.ones(2, 3)).sum().backward()
    assert nonfinite_grad_names(m) == []
    m[1].weight.grad[0, 1] = float("nan"); m[0].bias.grad[2] = float("inf")
    assert nonfinite_grad_names(m) == ["0.bias", "1.weight"]
    src = (pathlib.Path(__file__).resolve().parents[1] / "marsea/train.py").read_text()
    assert 'nan_step{step}.pt' in src and "nonfinite_grad_names(model)" in src


@pytest.mark.parametrize("triggered", [False, True])
def test_gold_by_head_reports_every_head_and_agrees_with_the_single_head_monitor(triggered):
    """owner's request (2026-09-21): per head, the fraction of gold tokens IN the relation and OFF it."""
    from marsea.monitor import gold_by_head, monitor_example, format_heads, HEAD_KEYS
    model, ctx = _tiny_model(); ctx.triggered = triggered
    ex = _example()
    res = gold_by_head(model, ctx, [ex, ex], [1, 2], device="cpu")
    assert res["n_rows"] == 12 and set(res["layers"]) == {"1", "2"}
    for l, r in res["layers"].items():
        assert set(HEAD_KEYS) <= set(r) and all(len(r[k]) == 8 for k in HEAD_KEYS)
        for h in range(8):
            assert abs(r["src_in_relation"][h] + r["src_off_relation"][h] - 1.0) < 1e-12     # the two fractions: everything is somewhere
            assert 0.0 <= r["gold_in_relation"][h] <= 1.0 and 0.0 <= r["src_mass_softmax"][h] <= 1.0
    recs = monitor_example(model, ctx, ex, 2, 1, device="cpu")                                # the single-head monitor at (2, 1)
    r = res["layers"]["2"]
    mean = lambda k: sum(x[k] for x in recs) / len(recs)
    assert abs(r["src_in_relation"][1] - mean("src_in_relation")) < 1e-12 and abs(r["src_mass_softmax"][1] - mean("src_mass_softmax")) < 1e-12
    assert abs(r["src_mass_final"][1] - mean("src_mass_final")) < 1e-12 and abs(r["relation_size"][1] - mean("relation_size")) < 1e-9
    text = format_heads(7, res)
    assert text.count("\n") == 8 and "IN relation" in text and "OFF" in text
    for x in recs:                                                                            # which stage zeroed a member source
        if x["src_in_relation"]:
            assert x["src_zero_column"] + x["src_zero_cap"] + x["src_zero_step2"] == float(x["src_mass_final"] == 0)
        else:
            assert math.isnan(x["src_zero_column"])



@pytest.mark.parametrize("triggered", [False, True])
def test_block_metrics_partition_the_row_and_agree_with_the_monitor(triggered):
    """owner's decision (2026-09-22): the gold is a partition.  own + other + rest = the row's mass; the own-block softmax mass
    at (l, h) equals the single-head monitor's copy-source-block reading; a row-less block is 'other', never 'own'."""
    from marsea.blockgold import block_metrics, block_metrics_example, ruler_blocks, format_blocks, list_errors
    model, ctx = _tiny_model(); ctx.triggered = triggered
    ex = _example(); ex.value_owner = [0, 0]; ex.meta = {}
    bl = ruler_blocks(ex)
    assert [b["spans"] for b in bl] == [[(5, 8)], [(12, 15)]] and [b["rows"] for b in bl] == [[29, 30, 31], [32, 33, 34]]
    res = block_metrics(model, ctx, [ex, ex], [1, 2], device="cpu")
    assert res["n_rows"] == 12 and set(res["layers"]) == {"1", "2"}
    r = res["layers"]["2"]
    for h in range(8):
        tot = r["mass_own_sm"][h] + r["mass_other_sm"][h] + r["mass_rest_sm"][h]
        assert abs(tot - 1.0) < 1e-6                                              # the three regions partition the visible row
        assert r["mass_own_final"][h] + r["mass_other_final"][h] + r["mass_rest_final"][h] <= 1.0 + 1e-6   # final: the tau_i discount may lose mass
        assert 0 <= r["block_precision_rel"][h] <= 1 and 0 <= r["block_recall"][h] <= 1
        assert abs(r["mass_stale_final"][h] + r["mass_remaining_final"][h] - r["mass_other_final"][h]) < 1e-9   # other = stale + remaining
    # value 1's rows come after value 0's: from their side value 0 is STALE, never remaining
    o = block_metrics_example(model, ctx, ex, [2], device="cpu")[2]
    assert float(o["mass_remaining_sm"].sum()) > 0 and float(o["mass_stale_sm"].sum()) > 0
    # a row-less block is 'other'
    ex2 = _example(); ex2.value_owner = [0, 0]; ex2.meta = dict(task="variable_tracking", chains=[dict(value="1", vars=["Z"], queried=False)], var_mentions={"Z": [(20, 22)]})
    bl2 = ruler_blocks(ex2); assert bl2[-1]["rows"] == [] and bl2[-1]["spans"] == [(20, 22)]
    o2 = block_metrics_example(model, ctx, ex2, [2], device="cpu", blocks=bl2)[2]
    assert float(o2["mass_other_sm"].sum()) > float(o["mass_other_sm"].sum()) - 1e-12
    assert "col purity" in format_blocks(0, res) and format_blocks(0, res).count("\n") == 8
    # B0 has no relation: relation columns are None, masses still there
    e = list_errors(" 111, 222, 222, 999 mentioned", ["111", "222", "333"])
    assert abs(e.pop("recall") - 2 / 3) < 1e-12 and abs(e.pop("precision") - 2 / 4) < 1e-12 and e == dict(n_listed=4, n_gold=3, repeated=1, omitted=1, extra=1, exact=False)
    assert list_errors(" 333, 111, 222", ["111", "222", "333"])["exact"]


def test_qa_blocks_own_the_supporting_facts_and_distractors_are_row_less():
    """QA block gold (2026-09-22): the answer rows own the supporting sentences (Hotpot) / paragraphs (MuSiQue); every
    distractor passage is a row-less block, so its mass reads as 'other'."""
    from marsea.blockgold import qa_blocks, block_metrics
    from marsea.data.qa import QAExample
    model, ctx = _tiny_model()
    ex = QAExample(prompt="", gold="", answer="x", question="q", passages=[("t0", "", False), ("t1", "", True), ("t2", "", False), ("t3", "", True)],
                   gold_slots=[1, 3], m=2, prompt_ids=list(range(40)), gold_ids=[5, 6, 7],
                   passage_spans=[(0, 10), (10, 20), (20, 30), (30, 38)], sentence_spans=[(1, 0, (12, 15)), (3, 1, (33, 36))], answer_rows=[39, 40, 41])
    bl = qa_blocks(ex)
    assert bl[0]["rows"] == [39, 40, 41] and bl[0]["spans"] == [(12, 15), (33, 36)]
    assert [(b["name"], b["rows"], b["spans"]) for b in bl[1:]] == [("passage0", [], [(0, 10)]), ("passage2", [], [(20, 30)])]
    ex.sentence_spans = []                                                     # MuSiQue: paragraph-level gold
    assert qa_blocks(ex)[0]["spans"] == [(10, 20), (30, 38)]
    res = block_metrics(model, ctx, [ex], [1, 2], device="cpu")
    r = res["layers"]["2"]
    assert res["n_rows"] == 3 and all(r["mass_stale_sm"][h] == 0 and r["mass_remaining_sm"][h] == 0 for h in range(8))   # one owner: nothing stale
    assert all(abs(r["mass_own_sm"][h] + r["mass_other_sm"][h] + r["mass_rest_sm"][h] - 1) < 1e-6 for h in range(8))


def test_aux_block_loss_runs_on_a_tiny_model_and_moves_the_relation():
    """the Phase-B auxiliary: relation logits at the owned answer rows vs the block gold, from the captured (K, Q)."""
    import torch
    from types import SimpleNamespace
    from marsea.train import aux_block_loss, TrainConfig
    from marsea.blockgold import ruler_blocks
    model, ctx = _tiny_model(); ctx.triggered = True
    for l in (1, 2):
        nm = model.model.layers[l].self_attn.normalizer
        nm.relation_anchor_scores = False; nm.relation_heads = [0, 1, 2]
    ex = _example(); ex.value_owner = [0, 0]; ex.meta = {}
    blocks = ruler_blocks(ex)
    ids = torch.tensor([ex.prompt_ids + ex.gold_ids]); labels = torch.tensor([[-100] * len(ex.prompt_ids) + ex.gold_ids])
    seq = SimpleNamespace(input_ids=ids, labels=labels, source="ruler", index=0, n_label_tokens=len(ex.gold_ids), blocks=blocks)
    cfg = TrainConfig(aux_block_weight=1.0, aux_anneal_steps=100, phase_a_steps=0)
    ctx.aux_capture = True; ctx.aux_qk = {}; ctx.n_prefill = len(ex.prompt_ids)
    model.train()
    model(input_ids=ids, use_cache=False, logits_to_keep=1)
    assert set(ctx.aux_qk) == {1, 2}
    loss, w = aux_block_loss(model, ctx, seq, cfg, step=0)
    assert loss is not None and w == 1.0 and float(loss) > 0
    loss.backward()
    nm = model.model.layers[1].self_attn.normalizer
    assert float(nm.relation.U.grad.abs().sum()) > 0 if hasattr(nm.relation.U, "grad") else True
    l2, w2 = aux_block_loss(model, ctx, seq, cfg, step=100)                            # annealed to the floor, not to zero
    assert l2 is not None and abs(w2 - 0.2) < 1e-9
    cfg.aux_floor = 0.0
    assert aux_block_loss(model, ctx, seq, cfg, step=100)[0] is None
