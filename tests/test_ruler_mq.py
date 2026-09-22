"""RULER multi-query x multi-value NIAH (2026-09-22): the gold is a PARTITION -- queried key q owns its V values, value v's
answer rows own the tokens of its needle -- and the parser must say who owns what.  Q = 1 is byte-identical to before."""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import pytest
from marsea.data import ruler as R


def test_split_query_matches_niah_py_join():
    assert R.split_query("psychedelic-walker") == ["psychedelic-walker"]
    assert R.split_query("a-b, and c-d") == ["a-b", "c-d"]
    assert R.split_query("a-b, c-d, e-f, and g-h") == ["a-b", "c-d", "e-f", "g-h"]
    assert R.split_query("") == []


def test_mq_grids_are_feasible_and_budget_the_answer():
    tr = R.grid_training_mq(seeds=(100,))
    assert all(c.num_needle_k >= c.num_needle_q for c in tr) and all(R.feasible(c) for c in tr)
    assert {(c.num_needle_q, c.num_needle_v) for c in tr} == {(1, 4), (1, 8), (1, 16), (2, 4), (2, 8), (2, 16), (4, 4), (4, 8)}   # Q4 x V16 does not fit 4K
    assert R.answer_tokens(4, 8) == 9 * 32 + 16 and R.answer_tokens(1, 4) == 128
    assert all(c.seed == R.QUICK_SEED and c.gold_depth == 0.5 for c in R.grid_quick_mq())
    assert len(R.grid_quick_mq(L=8192)) == 9                                              # Q4 x V16 fits 8K


def _fake_record(keys, vals_per_key, filler="The grass is green. " * 40):
    """a niah.py-shaped record: needle sentences in the input, the joined query in the answer prefix."""
    needles = [f"One of the special magic numbers for {k} is: {v}." for k, vs in zip(keys, vals_per_key) for v in vs]
    inp = ("Some special magic numbers are hidden within the following text. Make sure to memorize it.\n" + filler
           + " ".join(f"{n} {filler}" for n in needles) + f"\nWhat are all the special magic numbers for {R_join(keys)} mentioned in the provided text?")
    return dict(input=inp, answer_prefix=f" The special magic numbers for {R_join(keys)} mentioned in the provided text are",
                outputs=[v for vs in vals_per_key for v in vs], length=0, index=0)


def R_join(keys):
    return ", ".join(keys[:-1]) + ", and " + keys[-1] if len(keys) > 1 else keys[0]


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")


def test_owner_blocks_for_two_queried_keys(tok):
    rec = _fake_record(["sweet-chem", "splendid-battery"], [["1865242", "2162303"], ["8003414", "8138361", "7013622"]])
    ex = R.build_example(rec, tok)
    assert ex.query_keys == ["sweet-chem", "splendid-battery"] and ex.query_key == "sweet-chem, and splendid-battery"
    assert ex.value_owner == [0, 0, 1, 1, 1] and len(ex.value_spans) == 5 == len(ex.answer_value_rows)
    blocks = R.owner_blocks(ex)
    for b, v in zip(blocks, ex.outputs):
        a, e = b["span"]
        assert tok.decode(ex.prompt_ids[a:e]) == v                                          # the block's span IS the needle value
        assert tok.decode([ex.gold_ids[r - len(ex.prompt_ids) + 1] for r in b["rows"]]) == v   # and its rows emit exactly that value
    assert [b["key"] for b in blocks] == [0, 0, 1, 1, 1]
    # one coreference column per key (needles + question + prefix = 2 + 2 and 3 + 2 mentions), one value column per value
    assert sum(1 for k in ex.T_j_kind.values() if k == "coref") == 2 and sum(1 for k in ex.T_j_kind.values() if k == "value") == 5
    assert [len(m) for m in ex.key_mentions_by_key] == [4, 5]
    assert ex.key_mentions == sorted(ex.key_mentions) and len(ex.key_mentions) == 9      # merged, in text order


def test_single_query_fields_are_unchanged(tok):
    rec = _fake_record(["psychedelic-walker"], [["8682697", "1234567"]])
    ex = R.build_example(rec, tok)
    assert ex.query_keys == ["psychedelic-walker"] and ex.value_owner == [0, 0]
    assert ex.key_mention_spans == ex.key_mentions_by_key[0] and len(ex.key_mentions) == 4
    assert list(ex.T_j_kind.values()).count("coref") == 1


# --------------------------------------------------------------------------- variable tracking
def _fake_vt_record(chains, query_idx=0, filler="The grass is green. " * 30):
    """chains: [(value, [vars])]; statements interleaved chain by chain like RULER's heap shuffle (within-chain order kept)."""
    stmts = []
    for val, vs in chains:
        stmts.append([f"VAR {vs[0]} = {val}"] + [f"VAR {vs[i + 1]} = VAR {vs[i]}" for i in range(len(vs) - 1)])
    flat = [s for group in zip(*stmts) for s in group] if len({len(s) for s in stmts}) == 1 else [s for g in stmts for s in g]
    ctx = filler + " ".join(f"{s}. {filler}" for s in flat)
    val, vs = chains[query_idx]
    return dict(input=f"Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{ctx}\nQuestion: Find all variables that are assigned the value {val} in the text above.",
                answer_prefix=f" Answer: According to the chain(s) of variable assignment in the text above, {len(vs)} variables are assigned the value {val}, they are:",
                outputs=list(vs), length=0, index=0, _config=dict(num_chains=len(chains), num_hops=len(vs) - 1, seed=3))


def test_vt_grids_respect_the_forty_depth_slots():
    assert all(R.vt_feasible(c) for c in R.grid_training_vt(seeds=(100,)))
    assert {(c.num_chains, c.num_hops) for c in R.grid_training_vt(seeds=(100,))} == {(1, 4), (1, 8), (2, 4), (2, 8), (4, 4), (4, 8)}
    assert not R.vt_feasible(R.VTConfig("x", 4096, 8, 8))                                    # 72 statements
    assert R.VTConfig("VT", 4096, 2, 4, seed=100).save_name() == "VT_L4096_C2_H4_s100"


def test_vt_example_recovers_the_chains_and_owner_blocks(tok):
    rec = _fake_vt_record([("48213", ["TEJEK", "HLZNV", "XQATL"]), ("17713", ["WZCRV", "UKEWS", "YXGZY"])], query_idx=0)
    assert R.is_vt_record(rec)
    ex = R.build_example(rec, tok)                                                            # dispatches to the VT builder
    assert ex.meta["task"] == "variable_tracking" and ex.query_key == "48213" and ex.query_keys == ["48213"]
    assert ex.outputs == ["TEJEK", "HLZNV", "XQATL"] and ex.value_owner == [0, 0, 0]
    ch = {c["value"]: (c["vars"], c["queried"]) for c in ex.meta["chains"]}
    assert ch["48213"] == (["TEJEK", "HLZNV", "XQATL"], True) and ch["17713"] == (["WZCRV", "UKEWS", "YXGZY"], False)
    for b, v in zip(R.owner_blocks(ex), ex.outputs):
        a, e = b["span"]
        assert tok.decode(ex.prompt_ids[a:e]).strip() == v                                     # the DEFINITION mention of v
        assert tok.decode([ex.gold_ids[r - len(ex.prompt_ids) + 1] for r in b["rows"]]).strip() == v
    assert [len(ex.meta["var_mentions"][v]) for v in ex.outputs] == [2, 2, 1]               # defined once, used once (the last: never used)
    assert [tok.decode(ex.prompt_ids[a:e]) for a, e in ex.sentence_spans][0].strip() == "VAR TEJEK = 48213"
    assert len(ex.key_mentions) == 3 and list(ex.T_j_kind.values()).count("coref") == 1 and list(ex.T_j_kind.values()).count("value") == 3
    assert ex.gold == " TEJEK, HLZNV, XQATL"
