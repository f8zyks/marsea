"""CPU tests of the instrumentation (Sec. 12) and the data utilities: interval / hit rate against verify_all.py's
planted columns, the three-way distractor split, the dense-arm two-domain precision, aggregation, QA metrics, and the
1:1:1 mix's deterministic cursor."""
import numpy as np
import torch
import pytest
from conftest import planted_column, interval
from marsea.fidelity import (interval_from_scores, column_fidelity, dense_column_precision, row_fidelity, passage_support,
                             coverage_residual, e8_summary_from_diag)
from marsea.primitives import sparsemax_masked
from marsea.evaluate import aggregate_ruler
from marsea.data.qa import f1_score, em_score, hotpot_joint, normalize_answer
from marsea.data.ruler import exact_set_accuracy, ruler_recall_score
from marsea.data.mix import MixedDataset


def test_interval_matches_harness(rng):
    for _ in range(300):
        nq = int(rng.integers(4, 40)); m = int(rng.integers(1, min(8, nq)))
        s, T = planted_column(rng, nq, m)
        lo, hi, delta, W = interval(s, T)
        Tm = torch.zeros(nq, dtype=torch.bool); Tm[list(T)] = True
        iv = interval_from_scores(torch.as_tensor(s, dtype=torch.float32), Tm, torch.ones(nq, dtype=torch.bool))
        assert iv["m"] == m and abs(iv["lo"] - lo) < 1e-6 * max(1, lo) and (iv["hi"] == hi if not np.isfinite(hi) else abs(iv["hi"] - hi) < 1e-5 * hi)
        assert abs(iv["delta"] - delta) < 1e-5


def test_column_fidelity_hit_and_pr(rng):
    """the SHARP lower endpoint scores a hit where 1/(m delta) would record a miss; P/R read the three regimes."""
    n_sharp_only = 0
    for _ in range(300):
        nq = int(rng.integers(6, 40)); m = int(rng.integers(2, min(8, nq)))
        s, T = planted_column(rng, nq, m)
        lo, hi, delta, W = interval(s, T)
        Tm = torch.zeros(nq, dtype=torch.bool); Tm[list(T)] = True
        st = torch.as_tensor(s, dtype=torch.float32)
        E = torch.ones(nq, dtype=torch.bool)
        for tau in (lo * 0.5, lo + 0.5 * (min(hi, 50 * lo) - lo), (hi * 1.5 if np.isfinite(hi) else None)):
            if tau is None: continue
            p = sparsemax_masked(tau * st[None], E[None])[0]
            c = column_fidelity(st, E, p, p, tau, Tm)
            inside = lo <= tau < hi
            assert c["hit"] == inside
            if inside:
                assert c["P_before"] == 1.0 and c["R_before"] == 1.0
            elif tau < lo:
                assert c["R_before"] == 1.0 and c["P_before"] < 1.0 and c["dist"] < 0
            else:
                assert c["P_before"] == 1.0 and c["R_before"] < 1.0 and c["dist"] > 0
        naive_lo = 1.0 / (m * delta)                      # the NON-sharp endpoint; it may even exceed 1/W
        if naive_lo > lo * 1.05 and lo < min(naive_lo, hi):
            tau = 0.5 * (lo + min(naive_lo, hi))
            p = sparsemax_masked(tau * st[None], E[None])[0]
            c = column_fidelity(st, E, p, p, tau, Tm)
            assert c["hit"], "sharp endpoint must score a hit where 1/(m delta) would not"
            n_sharp_only += 1
    assert n_sharp_only > 10


def test_column_relation_recall_and_empty_complement():
    s = torch.tensor([3.0, 2.0, 1.0, -1.0, -2.0])
    T = torch.tensor([1, 1, 1, 0, 0], dtype=torch.bool)
    E = torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool)                 # target 2 excluded by the relation: recall miss
    iv = interval_from_scores(s, T, E)
    assert iv["m"] == 2 and iv["m_total"] == 3 and abs(iv["rel_recall"] - 2 / 3) < 1e-9
    assert iv["delta"] == float("inf") and iv["lo"] == 0.0                  # complement empty -> lower endpoint 0
    E2 = torch.tensor([0, 0, 0, 1, 1], dtype=torch.bool)
    iv2 = interval_from_scores(s, T, E2)
    assert iv2["m"] == 0                                                    # excluded from column averages


def test_row_split_and_dense_domains():
    A = torch.tensor([0.0, 0.3, 0.0, 0.2, 0.1, 0.4])
    Atil = torch.tensor([0.0, 0.3, 0.05, 0.2, 0.1, 0.4])
    E = torch.tensor([1, 1, 1, 1, 0, 1], dtype=torch.bool)
    K = torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.bool)
    r = row_fidelity(A, Atil, E, K, 0.02, 1.0, 0.95, 0.9, True)
    # distractors on the relation: 0 (excluded by stage 1), 2 (rejected by row), 3 and 5 (surviving)
    assert abs(r["excluded_by_stage1"] - 0.25) < 1e-9 and abs(r["rejected_by_row"] - 0.25) < 1e-9 and abs(r["surviving"] - 0.5) < 1e-9
    assert r["P_after"] == 1 / 3 and r["R_after"] == 1.0 and r["supp_rel"] == 3
    Acol = torch.softmax(torch.randn(10), 0); vis = torch.ones(10, dtype=torch.bool)
    T = torch.zeros(10, dtype=torch.bool); T[:3] = True
    Ep = torch.zeros(10, dtype=torch.bool); Ep[[0, 1, 5, 6]] = True
    d = dense_column_precision(Acol, vis, T, Ep)
    assert abs(d["P_full"] - 0.3) < 1e-9 and abs(d["P_rel"] - 0.5) < 1e-9        # m/n and m_j/|E_.j|, identities
    assert passage_support(A, E, [(0, 2), (2, 4), (4, 6)]) == [True, True, True]
    assert passage_support(torch.tensor([0.0, 0.0, 0.1, 0.0]), torch.ones(4, dtype=torch.bool), [(0, 2), (2, 4)]) == [False, True]
    assert coverage_residual(10, [3, 3, 2]) == 2


def test_e8_summary_shapes(rng):
    from conftest import rand_case
    from marsea.normalizer import MarSeaNormalizer, State
    torch.manual_seed(0)
    norm = MarSeaNormalizer(16, 8)
    cs = rand_case(rng, n_q=30, n_k=10, B=2, Hkv=1, H=2)
    with torch.no_grad():
        A, d = norm.normalize(cs["S"], cs["vis"], cs["K_kv"], cs["Q"], State(), logits_override=cs["logits"])
    e8 = e8_summary_from_diag(d, cs["vis"])
    assert 0 <= e8["rho"] <= 1 and len(e8["rho_col"]) == 2 and len(e8["hist_Ecol"]) == 2
    assert not e8["row_trigger_violation"]
    assert set(e8["hist_kstar"][0].keys()) == {"0", "1", "2", "3-4", "5-8", "9-16", "17+"}


def test_aggregate_ruler():
    rows = [dict(m=2, exact_set=True, ruler_recall=1.0, attention=dict(rows=[dict(P_after=1.0, R_after=1.0, rel_recall=1.0, excluded_by_stage1=0.5, rejected_by_row=0.25, surviving=0.25)],
                                                                    cols=[dict(m=2, hit=True, delta_pos=True, P_after=1.0, rel_recall=1.0), dict(m=0, hit=None, delta_pos=False, P_after=0.0, rel_recall=0.0)])),
            dict(m=2, exact_set=False, ruler_recall=0.5, attention=dict(rows=[dict(P_after=0.5, R_after=1.0, rel_recall=1.0)],
                                                                     cols=[dict(m=1, hit=False, delta_pos=False, P_after=0.5, rel_recall=0.5)]))]
    t = aggregate_ruler(rows, "m")[2]
    assert t["n"] == 2 and abs(t["exact_set_acc"] - 0.5) < 1e-9 and abs(t["interval_hit_rate"] - 0.5) < 1e-9 and t["n_hit_columns"] == 2
    assert abs(t["frac_columns_delta_pos"] - 0.5) < 1e-9 and abs(t["col_precision_delta_pos"] - 1.0) < 1e-9 and abs(t["row_precision"] - 0.75) < 1e-9


def test_task_metrics():
    assert exact_set_accuracy(" 123, 456\nsomething", ["456", "123"]) and not exact_set_accuracy(" 123, 456, 789", ["123", "456"])
    assert not exact_set_accuracy(" 123", ["123", "456"]) and ruler_recall_score("123 and 456", ["123", "456", "789"]) == 2 / 3
    assert em_score("The Eiffel Tower.", "eiffel tower") == 1.0 and abs(f1_score("Eiffel Tower in Paris", "eiffel tower")[0] - 2 / 3) < 1e-9
    j = hotpot_joint("paris", "Paris", {0, 1}, {0, 1})
    assert j["joint_em"] == 1.0 and j["joint_f1"] == 1.0
    j2 = hotpot_joint("paris", "Paris", {0}, {0, 1})
    assert j2["sp_em"] == 0.0 and abs(j2["joint_f1"] - 2 * 1 * 0.5 / 1.5) < 1e-9


class _Tok:
    eos_token_id = 0
    def __call__(self, text, add_special_tokens=False, **kw):
        return {"input_ids": [ord(c) % 50 + 1 for c in text]}


def test_mix_deterministic_and_drops_long():
    src = {"ruler": [("a" * 5, "b"), ("c" * 50, "d"), ("e" * 7, "f")], "musique": [("g" * 3, "h"), ("i" * 4, "j")]}
    ds = MixedDataset(src, _Tok(), L=20, seed=3)
    ds2 = MixedDataset(src, _Tok(), L=20, seed=3)
    assert len(ds.perms["ruler"]) == 2                                      # the 50-char sequence is dropped, never truncated
    seqs = [ds.get(c) for c in range(6)]
    assert [s.source for s in seqs] == ["ruler", "musique"] * 3            # lock-step 1:1
    assert all(ds.get(c).index == ds2.get(c).index for c in range(20))      # a function of the seed alone
    s = seqs[0]
    assert (s.labels[0, :len(s.input_ids[0]) - s.n_label_tokens] == -100).all() and s.labels[0, -1] == 0


def test_sweep_routing_ratios_span_and_sink():
    """marsea/sweep.py: mass on the key SPAN against the row max with the sink (position 0) excluded; any query token."""
    from marsea.sweep import routing_ratios
    T = 8; A = torch.zeros(2, T, T)
    A[:, :, 0] = 0.6                                   # a sink on the first token in every row
    A[0, 6, 3] = 0.15; A[0, 6, 4] = 0.15               # head 0: query 6 puts 0.30 on the span (3,5), max candidate 0.15
    A[1, 7, 4] = 0.35                                  # head 1: query 7 hits the +1 token of the span only
    r = routing_ratios(A, (3, 5), [6, 7], exclude_sink=True)
    assert abs(float(r[0]) - 2.0) < 1e-6              # 0.30 / 0.15: the span carries twice the best single candidate
    assert abs(float(r[1]) - 1.0) < 1e-6              # the offset token is inside the span
    r2 = routing_ratios(A, (3, 5), [6], exclude_sink=False)
    assert abs(float(r2[0]) - 0.5) < 1e-6             # against the sink it would read as 0.5: the depressor the sweep removes
    assert routing_ratios(A, (3, 5), [2], exclude_sink=True) is None   # a query before the span sees none of it


def test_key_phrase_matching_is_word_bounded_and_seeds_are_process_stable():
    """key_mentions[0] anchors the whole D-9a coreference column, so a HAYSTACK hit inside a longer word relocates
    it (review d5bd980 D-4); and Python's hash() is salted per process, so every arm laid its passages out
    differently and a resume did not reproduce its own order (D-3)."""
    import subprocess, sys, pathlib
    from marsea.data.ruler import word_bounded_mentions
    from marsea.data.qa import _stable_hash
    hay = "a cat sat. concatenate. the cat."
    assert word_bounded_mentions("cat", hay) == [2, 28], word_bounded_mentions("cat", hay)
    assert word_bounded_mentions("glossy-antelope", "x glossy-antelopes y glossy-antelope.") == [21]
    assert word_bounded_mentions("", hay) == []
    root = str(pathlib.Path(__file__).resolve().parents[1])
    code = f"import sys;sys.path.insert(0,{root!r});from marsea.data.qa import _stable_hash;print(_stable_hash('q_123'))"
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip() for _ in range(3)}
    assert len(outs) == 1 and outs != {""}, f"per-record seed is not stable across processes: {outs}"
    assert str(_stable_hash("q_123")) in outs


def test_ruler_feasibility_is_measured_not_guessed():
    """The old guard (16 tokens per needle against L // 4) rejected E3 at n = 128, which spec Sec. 13.1 sizes as
    legal -- and under `set -e` that aborted gen_data.sh before the training pool generated (review d5bd980 D-1/D-2)."""
    from marsea.data import ruler as R
    for fn in (R.grid_E2, R.grid_E3, R.grid_E3_depth, R.grid_E4, R.grid_training, R.grid_detector):
        for cfg in fn():
            assert R.feasible(cfg), f"{cfg.save_name()} rejected: {R.needle_budget(cfg)}"
    big = R.NIAHConfig("X", 4096, 64, 4, 1)              # 256 needles, ~5.6K tokens: genuinely does not fit at 4K
    assert not R.feasible(big)
    assert R.QUICK_SEED not in R.EVAL_SEEDS and R.QUICK_SEED not in R.TRAIN_SEEDS


def test_detector_threshold_rejects_and_the_sink_is_excluded():
    """Sec. 6.7's "a head retrieves if score > threshold" was stored and printed, never applied; and the row argmax
    included position 0, the attention sink, which sweep.py already excludes (review d5bd980 D-6)."""
    import inspect
    from marsea import detector as det
    src = inspect.getsource(det.run_detector)
    assert 'wr[..., 0] = float("-inf")' in src, "the detector's argmax still includes the sink"
    assert "above = [r for r in ranking if r[0] > threshold]" in src, "threshold is still not a rejection"
    r = det.DetectorResult(scores=[[0.0]], patched_layers=[0], l_star=0, h_star=0, n_prompts=1, threshold=0.1, ranking=[])
    assert r.usable is None and r.layers_below_threshold == []      # old JSON still loads
