"""The relation monitor: per answer row, the ONE key it is about to copy -- on a tiny model, both training forms."""
import math, torch, pytest
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
