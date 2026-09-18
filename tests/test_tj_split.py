"""The split T_j (spec v4.7 / paper Sec. 2.3, 2026-09-09): on an MV-NIAH example with m needles the key phrase's first
mention has m_j = m + 1 targets (the other m - 1 needles, the question, the answer prefix) and NO answer positions; each
value column has exactly its emitting answer row.  Needs the backbone tokenizer in the HF cache."""
import pytest
import torch
from marsea.data.ruler import build_example
from marsea.sweep import constructions

try:
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", local_files_only=True)
except Exception:
    TOK = None

NEEDLE = "One of the special magic numbers for {key} is: {value}."


def synthetic_record(m: int, key: str = "gamy-drain", filler: int = 40):
    import random
    rng = random.Random(0)
    values = [str(rng.randint(1000000, 9999999)) for _ in range(m)]
    dis_key = "salty-harbor"; dis_values = [str(rng.randint(1000000, 9999999)) for _ in range(m)]
    sents = ["The quick brown fox jumps over the lazy dog."] * filler
    for k in range(m):
        sents.insert(5 + 3 * k, NEEDLE.format(key=key, value=values[k]))
        sents.insert(9 + 3 * k, NEEDLE.format(key=dis_key, value=dis_values[k]))
    ctx = " ".join(sents)
    plural = m > 1
    inp = (f"Some special magic {'numbers' if plural else 'number'} are hidden within the following text. Make sure to memorize it. "
           f"I will quiz you about the {'numbers' if plural else 'number'} afterwards.\n{ctx}\nWhat {'are all the' if plural else 'is the'} special magic "
           f"{'numbers' if plural else 'number'} for {key} mentioned in the provided text?")
    prefix = f" The special magic {'numbers' if plural else 'number'} for {key} mentioned in the provided text {'are' if plural else 'is'}"
    return dict(index=0, input=inp, outputs=values, length=0, answer_prefix=prefix)


@pytest.mark.skipif(TOK is None, reason="backbone tokenizer not cached")
@pytest.mark.parametrize("m", [1, 2, 4, 8])
def test_mv_niah_split_m_plus_one(m):
    ex = build_example(synthetic_record(m), TOK)
    assert len(ex.outputs) == m and len(ex.value_spans) == m and len(ex.answer_rows) == m
    j0 = ex.key_mentions[0]
    assert ex.T_j_kind[j0] == "coref"
    assert len(ex.T_j[j0]) == m + 1, f"m_j on the key phrase must be m + 1, got {len(ex.T_j[j0])}"
    assert not (set(ex.T_j[j0]) & set(ex.answer_rows)), "answer positions must not be targets of the coreference column"
    assert all(t < len(ex.prompt_ids) for t in ex.T_j[j0])                    # every target is a prompt position
    value_cols = [j for j, k in ex.T_j_kind.items() if k == "value"]
    assert len(value_cols) == m
    for (a, b), row in zip(ex.value_spans, ex.answer_rows):
        assert ex.T_j[a] == [row] and ex.T_j_kind[a] == "value"
    # the sweep's constructions agree with the annotator's split
    cons = constructions(ex)
    assert len(cons["coref_mentions"]) == m + 1 and len(cons["C-a"]) == m
    assert cons["coref_mentions"][0][0][0] == j0                                # key span starts at the first mention
    assert "coref_full" not in cons


@pytest.mark.skipif(TOK is None, reason="backbone tokenizer not cached")
def test_d9a_scores_each_kind_at_its_own_site():
    """D-9a is only half done if T_j_kind is a TAG: coreference is the induction pattern and lives at a different
    (layer, head) from the detector's retrieval (l*, h*), so measuring both there while the tables split by kind
    makes the tables look split when they are not (review d5bd980 E)."""
    from marsea.evaluate import attention_level_ruler
    from marsea.normalizer import Diagnostics
    ex = build_example(synthetic_record(2), TOK)
    n = len(ex.prompt_ids) + len(ex.gold_ids)

    def fake(seed):
        torch.manual_seed(seed)
        i = torch.arange(n)[:, None]; j = torch.arange(n)[None, :]
        vis = (j <= i)
        S = torch.randn(1, 1, n, n)
        A = torch.softmax(S.masked_fill(~vis, float("-inf")), -1)
        E = (torch.rand(1, 1, n, n) < 0.15) & vis
        d = Diagnostics(E=E, A=A, A_sm=A, Atil=A, a1=A, p=A * E, tau_j=torch.ones(1, 1, n), tau_i=torch.ones(1, 1, n),
                        kstar=E.sum(-2), nu=torch.ones(1, 1, n), psi_j=torch.zeros(1, 1, n), theta=torch.zeros(1, 1, n),
                        Rtil=(A * E).sum(-1), cbar_i=(A * E).sum(-1), cbar_j=(A * E).sum(-2),
                        supp_rel=((A > 0) & E).sum(-1), cap_binds=torch.zeros(1, 1, n, dtype=torch.bool))
        d.extra["S"] = S.masked_fill(~vis, float("-inf")); d.extra["head_sub"] = 0
        return d

    diags = {5: fake(1), 9: fake(2)}
    out_one = attention_level_ruler(diags, 5, 0, ex, arm_is_dense=False)
    out_two = attention_level_ruler(diags, 5, 0, ex, arm_is_dense=False, coref_site=(9, 0))
    kinds = {c["col"]: c["kind"] for c in out_one["cols"]}
    assert set(kinds.values()) == {"coref", "value"}
    for c in out_two["cols"]:
        assert c["site"] == ([9, 0] if c["kind"] == "coref" else [5, 0]), c
    for c in out_one["cols"]:
        assert c["site"] == [5, 0]
    coref_one = {c["col"]: c["P_after"] for c in out_one["cols"] if c["kind"] == "coref"}
    coref_two = {c["col"]: c["P_after"] for c in out_two["cols"] if c["kind"] == "coref"}
    val_one = {c["col"]: c["P_after"] for c in out_one["cols"] if c["kind"] == "value"}
    val_two = {c["col"]: c["P_after"] for c in out_two["cols"] if c["kind"] == "value"}
    assert val_one == val_two, "the value columns must be unaffected by the coreference site"
    assert coref_one != coref_two, "the coreference columns were read from the same tensors at both sites"
    # the residual is reported per kind, not pooled into one uninterpretable number
    assert set(out_one["coverage_residual"]) == {"coref", "value"}


@pytest.mark.skipif(TOK is None, reason="backbone tokenizer not cached")
def test_a_coref_site_in_l_stars_own_layer_does_not_collapse():
    """`sites` keyed by LAYER collapsed when the coreference head lives in l*'s own layer: the second entry
    overwrote the first, one head was kept, and the coreference columns were measured at h* while being labelled
    h' -- the same class as last round's paired_E head-0 defect (review 30372ae B-1)."""
    from marsea.evaluate import attention_level_ruler, site_map, _kept_head
    from marsea.normalizer import Diagnostics
    ex = build_example(synthetic_record(2), TOK)
    n = len(ex.prompt_ids) + len(ex.gold_ids)
    assert site_map([(19, 3), (19, 7)]) == {19: [3, 7]}, "two heads of one layer must both survive"
    assert site_map([(19, 3), (14, 5)]) == {19: [3], 14: [5]}
    assert site_map([(19, 3), (19, 3)]) == {19: [3]}

    torch.manual_seed(3)
    i = torch.arange(n)[:, None]; j = torch.arange(n)[None, :]
    vis = (j <= i)
    def head(seed, rho):
        torch.manual_seed(seed)
        S = torch.randn(1, 1, n, n)
        A = torch.softmax(S.masked_fill(~vis, float("-inf")), -1)
        E = (torch.rand(1, 1, n, n) < rho) & vis
        return S, A, E
    S0, A0, E0 = head(11, 0.15); S1, A1, E1 = head(22, 0.60)        # the layer kept heads [3, 7] in that order
    S = torch.cat([S0, S1], 1); A = torch.cat([A0, A1], 1); E = torch.cat([E0, E1], 1)
    d = Diagnostics(E=E, A=A, A_sm=A, Atil=A, a1=A, p=A * E, tau_j=torch.ones(1, 2, n), tau_i=torch.ones(1, 2, n),
                    kstar=E.sum(-2), nu=torch.ones(1, 2, n), psi_j=torch.zeros(1, 2, n), theta=torch.zeros(1, 2, n),
                    Rtil=(A * E).sum(-1), cbar_i=(A * E).sum(-1), cbar_j=(A * E).sum(-2),
                    supp_rel=((A > 0) & E).sum(-1), cap_binds=torch.zeros(1, 2, n, dtype=torch.bool))
    d.extra["S"] = S.masked_fill(~vis, float("-inf")); d.extra["head_sub"] = [3, 7]
    assert _kept_head(d, 3) == 0 and _kept_head(d, 7) == 1
    out = attention_level_ruler({19: d}, 19, 3, ex, arm_is_dense=False, coref_site=(19, 7))
    for c in out["cols"]:
        assert c["site"] == ([19, 7] if c["kind"] == "coref" else [19, 3]), c
    # and the coreference columns must actually read head 7's tensors, not h*'s
    one = attention_level_ruler({19: d}, 19, 3, ex, arm_is_dense=False)
    # repr, not ==: these dicts carry NaNs, and nan != nan would make every comparison below vacuously true
    strip = lambda cols, kind: [repr(sorted((k, repr(v)) for k, v in c.items() if k != "site"))
                                for c in cols if c["kind"] == kind]
    assert strip(out["cols"], "value") == strip(one["cols"], "value"), "the value columns must be unaffected"
    assert strip(out["cols"], "coref") != strip(one["cols"], "coref"), "the coreference columns were still read at h*"
