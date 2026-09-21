"""The triggered teacher-forced form == prefill + frozen-prefix decoding, row by row (owner's decision, 2026-09-20).
`causal.decode_step` is the reference: the prompt is prefilled with the ordinary normaliser, the cache is seeded as
generation seeds it, and every answer row is a decode step.  The triggered form must give the same attention rows."""
import torch, pytest
from marsea.normalizer import MarSeaNormalizer, State
from marsea.causal import FrozenPrefixCache, decode_step
from test_t11_chunked import _case, D, R

DT = torch.float64


def _setup(T=44, n_p=30, H=(2, 3), rho=0.3, seed=0, **kw):
    torch.manual_seed(seed)
    norm = MarSeaNormalizer(D, R, **kw).to(DT)
    Q, K_kv, V_kv, vis = _case(T, B=1, Hkv=H[0], g=H[1], dtype=DT, seed=seed)
    g = Q.shape[1] // K_kv.shape[1]
    S = (torch.einsum("bhid,bhjd->bhij", Q, K_kv.repeat_interleave(g, dim=1)) * D ** -0.5).masked_fill(~vis, float("-inf"))
    with torch.no_grad():
        norm.relation.calibrate_b0(K_kv, Q, vis, rho)
        for p_ in norm.tauK.parameters(): p_.add_(0.05 * torch.randn_like(p_))      # tau heads that actually vary
        for p_ in norm.tauQ.parameters(): p_.add_((0.3 if norm.tauQ.sized else 0.05) * torch.randn_like(p_))
        if norm.tauK_trig is not None:
            for p_ in norm.tauK_trig.parameters(): p_.add_(0.3 * torch.randn_like(p_))   # a trigger-time tau that really varies
    return norm, Q, K_kv, S, vis, n_p


def _by_decoding(norm, Q, K_kv, S, vis, n_p, K_ret):
    """the reference: prefill the prompt, then one decode_step per answer row."""
    T = S.shape[-1]; rows = []
    with torch.no_grad():
        A_p, d_p = norm.normalize(S[:, :, :n_p, :n_p], vis[:, :, :n_p, :n_p], K_kv[:, :, :n_p], Q[:, :, :n_p], State())
        cache = FrozenPrefixCache(K_ret=K_ret); cache.init_from_prefill(S[:, :, :n_p, :n_p].masked_fill(~vis[:, :, :n_p, :n_p], float("-inf")), d_p)
        for t in range(n_p, T):
            A_row, info = decode_step(norm, cache, S[:, :, t:t + 1, :t + 1], vis[:, :, t:t + 1, :t + 1], K_kv[:, :, :t + 1], Q[:, :, t:t + 1])
            rows.append(torch.nn.functional.pad(A_row, (0, T - t - 1)))
    return A_p, torch.cat(rows, -2)


@pytest.mark.parametrize("K_ret,kw", [(64, {}), (3, {}), (64, dict(relation_per_head=6, per_head=6)), (64, dict(head_block=2)),
                                      (64, dict(tau_i_pinned=True)), (64, dict(trigger_tau=True)), (3, dict(trigger_tau=True)),
                                      (64, dict(trigger_tau=True, relation_per_head=6, per_head=6, head_block=2)),
                                      (64, dict(size_aware_tau_i=True)), (3, dict(size_aware_tau_i=True, trigger_tau=True)),
                                      (64, dict(size_aware_tau_i=True, trigger_tau=True, relation_per_head=6, per_head=6, head_block=2)),
                                      (64, dict(cap_mode="relation")), (3, dict(cap_mode="relation", cap_fallback_tail_dominates=True)),
                                      (64, dict(cap_mode="relation", size_aware_tau_i=True, trigger_tau=True, relation_per_head=6, per_head=6, head_block=2))])
def test_triggered_equals_prefill_plus_decode(K_ret, kw):
    norm, Q, K_kv, S, vis, n_p = _setup(**kw)
    A_p, A_dec = _by_decoding(norm, Q, K_kv, S, vis, n_p, K_ret)
    with torch.no_grad():
        A, d = norm.normalize(S, vis, K_kv, Q, State(), n_prefill=n_p, decode_K_ret=K_ret)
    assert float((A[:, :, :n_p, :n_p] - A_p).abs().max()) < 1e-12 and float(A[:, :, :n_p, n_p:].abs().max()) == 0.0
    err = float((A[:, :, n_p:] - A_dec).abs().max())
    assert err < 1e-9, err
    assert bool((d.E[:, :, n_p:].sum(-1) > 0).any()) and float(d.p[:, :, n_p:].sum()) > 0       # the answer rows DID trigger columns
    if K_ret == 3:
        assert int((d.E.sum(-2) > 3).sum()) > 0                                                  # and lists really were truncated


def test_it_differs_from_the_full_sequence_form_and_no_row_sees_a_later_one():
    norm, Q, K_kv, S, vis, n_p = _setup()
    with torch.no_grad():
        A_full, _ = norm.normalize(S, vis, K_kv, Q, State())
        A_trig, _ = norm.normalize(S, vis, K_kv, Q, State(), n_prefill=n_p)
        assert float((A_full - A_trig).abs().max()) > 1e-6                          # the full form couples rows to later rows
        # perturb the LAST answer row's query: in the triggered form no earlier row may move; in the full form some do
        Q2 = Q.clone(); Q2[:, :, -1] += 3.0
        g = Q.shape[1] // K_kv.shape[1]
        S2 = (torch.einsum("bhid,bhjd->bhij", Q2, K_kv.repeat_interleave(g, dim=1)) * D ** -0.5).masked_fill(~vis, float("-inf"))
        B_trig, _ = norm.normalize(S2, vis, K_kv, Q2, State(), n_prefill=n_p)
        B_full, _ = norm.normalize(S2, vis, K_kv, Q2, State())
    assert float((A_trig[:, :, :-1] - B_trig[:, :, :-1]).abs().max()) == 0.0
    assert float((A_full[:, :, :-1] - B_full[:, :, :-1]).abs().max()) > 1e-9        # the leak the triggered form removes


def test_gradients_reach_every_module_and_forced_empty_is_softmax():
    norm, Q, K_kv, S, vis, n_p = _setup(relation_per_head=6)
    norm.st_temperature = 4.0
    A, d = norm.normalize(S, vis, K_kv, Q, State(), n_prefill=n_p)
    (A[:, :, n_p:] * torch.linspace(0.5, 1.5, A[:, :, n_p:].numel(), dtype=DT).view_as(A[:, :, n_p:])).sum().backward()
    for name in ("relation.U", "relation.V", "relation.b0", "tauK.fc2.weight", "tauQ.fc2.weight"):
        gp = dict(norm.named_parameters())[name].grad
        assert gp is not None and float(gp.abs().sum()) > 0, name
    with torch.no_grad():
        A0, _ = norm.normalize(S, vis, K_kv, Q, State(), n_prefill=n_p, logits_override=torch.full((1, 1, 1, 1), -1e4, dtype=DT))
        sm = torch.softmax(S, -1)
    assert float((A0 - sm).abs().max()) < 1e-12


def test_nu_handoff_is_the_prefills_for_prompt_keys_and_one_for_answer_keys():
    norm, Q, K_kv, S, vis, n_p = _setup()
    with torch.no_grad():
        st = State(); norm.normalize(S, vis, K_kv, Q, st, n_prefill=n_p)
        sp = State(); norm.normalize(S[:, :, :n_p, :n_p], vis[:, :, :n_p, :n_p], K_kv[:, :, :n_p], Q[:, :, :n_p], sp)
    assert torch.equal(st.nu_next[..., :n_p], sp.nu_next) and bool((st.nu_next[..., n_p:] == 1).all())


# ------------------------------------------------------------------------------------------ the whole model
def _tiny_model(head_block=None, sized=True, cap_mode="relation"):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from marsea.backbone import patch_model, MarSeaContext
    from marsea.baselines import make_normalizer
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=8,
                      num_key_value_heads=2, max_position_embeddings=512, attn_implementation="eager")
    model = Qwen2ForCausalLM(cfg).double()
    ctx = MarSeaContext(mode="dense")
    torch.manual_seed(1)
    patch_model(model, [1, 2], lambda l: make_normalizer("marsea", 8, head_block=head_block, relation_per_head=8, per_head=8,
                                                         trigger_tau=True, size_aware_tau_i=sized, cap_mode=cap_mode).double(), ctx)
    with torch.no_grad():
        for l in (1, 2):
            nm = model.model.layers[l].self_attn.normalizer
            nm.relation.b0.fill_(0.0)                                          # about half of the pairs: a relation that acts
            for p_ in list(nm.tauK_trig.parameters()) + list(nm.tauQ.parameters()): p_.add_(0.3 * torch.randn_like(p_))
    model.marsea_ctx = ctx; model.eval()
    return model, ctx


@pytest.mark.parametrize("hb", [None, 2])
def test_teacher_forced_logits_equal_cached_decoding_step_by_step(hb):
    """THE acceptance test: with ctx.triggered the teacher-forced pass gives, at every answer position, the logits that
    prefilling the prompt and then feeding the gold tokens one by one through the frozen-prefix decode path gives."""
    model, ctx = _tiny_model(hb)
    ids = torch.randint(0, 97, (1, 40), generator=torch.Generator().manual_seed(5)); n_p = 29
    with torch.no_grad():
        ctx.triggered = True; ctx.n_prefill = n_p; ctx.generation = False; ctx.collect = True
        tf = model(input_ids=ids, use_cache=False).logits                                  # [1, 40, V]
        rho = float(ctx.diags[2].E[:, :, n_p:].float().mean()); ctx.collect = False
        ctx.triggered = False; ctx.n_prefill = None
        full = model(input_ids=ids, use_cache=False).logits                                # the full-sequence form
        ctx.generation = True; ctx.clear_generation(); ctx.generation = True
        out = model(input_ids=ids[:, :n_p], use_cache=True); past = out.past_key_values; dec = [out.logits[:, -1]]
        for t in range(n_p, ids.shape[1]):
            out = model(input_ids=ids[:, t:t + 1], past_key_values=past, use_cache=True); past = out.past_key_values
            dec.append(out.logits[:, -1])
        ctx.clear_generation()
    dec = torch.stack(dec, 1)                                                              # positions n_p-1 .. 39
    assert rho > 0.05, rho                                                                 # the answer rows have a live relation
    # 1e-7: the trigger-time tau reads a mean and a std of the member list, summed in a different order on the two paths
    # (1.1e-8 measured with deliberately large random TauKTrigger weights; the full-sequence form is > 1e-6 away, below)
    assert float((tf[:, n_p - 1:] - dec).abs().max()) < 1e-7, float((tf[:, n_p - 1:] - dec).abs().max())
    assert float((full[:, n_p:] - dec[:, 1:]).abs().max()) > 1e-6                          # the form trained until now is NOT what decodes
    assert float((full[:, :n_p - 1] - tf[:, :n_p - 1]).abs().max()) > 1e-9                  # nor are its prompt rows the prefill's


def test_the_chunked_path_refuses_rather_than_silently_running_the_leaky_form():
    model, ctx = _tiny_model(sized=False, cap_mode="row")
    ctx.mode = "chunked"; ctx.triggered = True; ctx.n_prefill = 20
    with pytest.raises(NotImplementedError), torch.no_grad():
        model(input_ids=torch.randint(0, 97, (1, 32)), use_cache=False)
    ctx.n_prefill = None                                                                   # a generation prefill is never "triggered"
    with torch.no_grad():
        model(input_ids=torch.randint(0, 97, (1, 32)), use_cache=False)


def test_n_prompt_tokens():
    from marsea.train import n_prompt_tokens
    assert n_prompt_tokens(torch.tensor([[-100, -100, -100, 5, 6, 2]])) == 3
    assert n_prompt_tokens(torch.tensor([[-100, -100]])) == 2


# ------------------------------------------------------------------------------------------ the trigger-time temperature
def test_trigger_stats_reads_the_true_size_and_the_triggers_standing():
    from marsea.heads import trigger_stats
    vals = torch.tensor([[3.0, 1.0, -1e30, -1e30], [2.0, -1e30, -1e30, -1e30]], dtype=DT); valid = vals > -1e29
    st = trigger_stats(vals, valid, torch.tensor([7, 1]), torch.tensor([1.0, 2.0], dtype=DT))
    assert st.shape == (2, 6)
    assert st[0].tolist()[:4] == [7.0, 3.0, 2.0, 2.0] and abs(float(st[0, 4]) - 1.0) < 1e-6 and float(st[0, 5]) == -2.0   # size, max, gap, mean, std, s - max
    assert st[1].tolist()[:4] == [1.0, 2.0, 0.0, 2.0] and float(st[1, 5]) == 0.0                                          # a one-member column


def test_the_trigger_time_tau_is_used_changes_the_answer_rows_only_and_gets_gradients():
    norm, Q, K_kv, S, vis, n_p = _setup(trigger_tau=True, relation_per_head=6, per_head=6)
    with torch.no_grad():
        A1, _ = norm.normalize(S, vis, K_kv, Q, State(), n_prefill=n_p)
        saved = norm.tauK_trig; norm.tauK_trig = None
        A0, _ = norm.normalize(S, vis, K_kv, Q, State(), n_prefill=n_p)              # the sealed tau_j
        norm.tauK_trig = saved
    assert float((A1[:, :, :n_p] - A0[:, :, :n_p]).abs().max()) == 0.0               # the prompt's one solve keeps TauK
    assert float((A1[:, :, n_p:] - A0[:, :, n_p:]).abs().max()) > 1e-6
    A, _ = norm.normalize(S, vis, K_kv, Q, State(), n_prefill=n_p)
    A[:, :, n_p:].pow(2).sum().backward()
    assert all(p_.grad is not None and float(p_.grad.abs().sum()) > 0 for n_, p_ in norm.tauK_trig.named_parameters() if n_ in ("w1", "w2", "b2"))


def test_phase_b_init_starts_the_trigger_head_at_tauKs_calibrated_bias():
    import inspect, marsea.train as tr
    assert "nm.tauK_trig.last_bias.data.copy_(nm.tauK.last_bias.data)" in inspect.getsource(tr.phase_b_init)


# ------------------------------------------------------------------------------------------ the size-aware tau_i
def test_row_relation_stats_reads_the_rows_relation_set_and_its_columns():
    from marsea.heads import row_relation_stats
    Atil = torch.tensor([[[[0.5, 0.1, 0.3, 0.0], [0.2, 0.0, 0.0, 0.0]]]], dtype=DT)          # [1,1,2,4]
    E = torch.tensor([[[[True, True, True, False], [False, False, False, False]]]])
    st = row_relation_stats(Atil, E, torch.tensor([[[10, 2, 4, 9]]]), torch.full((1, 1, 2, 4), 0.5, dtype=DT))
    size, mx, gap, mean, std, cmean, cmax, share = st[0, 0, 0].tolist()
    assert (size, mx, cmax, share) == (3.0, 0.5, 10.0, 0.5) and abs(gap - 0.2) < 1e-12 and abs(mean - 0.3) < 1e-12 and abs(cmean - 16 / 3) < 1e-12
    assert abs(std - (((0.2 ** 2 + 0.2 ** 2 + 0.0) / 3) ** 0.5)) < 1e-6
    assert st[0, 0, 1].tolist()[:4] == [0.0, 0.0, 0.0, 0.0]                                   # an empty relation set: zeros, no NaN
    assert bool(torch.isfinite(st).all())


def test_the_size_aware_tau_i_sees_twelve_scalars_and_the_chunked_path_refuses():
    from marsea.chunked import marsea_chunked_attention
    norm, Q, K_kv, S, vis, n_p = _setup(size_aware_tau_i=True)
    assert norm.tauQ.sized and norm.tauQ.fc1.in_features == D + 12 and MarSeaNormalizer(D, R).tauQ.fc1.in_features == D + 4
    V = torch.randn(1, K_kv.shape[1], S.shape[-1], D, dtype=DT)
    with pytest.raises(NotImplementedError), torch.no_grad():
        marsea_chunked_attention(norm, Q, K_kv, V, vis, State(), chunk=16)


# ------------------------------------------------------------------------------------------ the relation-only cap
def _capped(cap_mode, **kw):
    norm, Q, K_kv, S, vis, n_p = _setup(rho=0.4, cap_mode=cap_mode, **kw)
    with torch.no_grad():
        A, d = norm.normalize(S, vis, K_kv, Q, State())
    return norm, d, vis


def test_the_relation_only_cap_leaves_the_tail_alone_and_the_row_at_its_softmax_mass():
    norm, d, vis = _capped("relation")
    visb = vis.expand_as(d.E); off = visb & ~d.E; b = d.cap_binds
    assert int(b.sum()) > 10, "no binding rows: the case does not exercise the cap"
    assert torch.equal(d.a1[off], d.A_sm[off])                                             # the off-relation tail: bitwise softmax, everywhere
    rel_sm = (d.A_sm * d.E).sum(-1); rel_a1 = (d.a1 * d.E).sum(-1); rel_til = (d.Atil * d.E).sum(-1)
    assert float((rel_a1[b] - rel_sm[b]).abs().max()) < 1e-9                               # a binding row: the relation back on its own softmax mass
    assert bool((rel_til[b] > rel_sm[b]).all()) and float((d.a1.sum(-1)[b] - d.A_sm.sum(-1)[b]).abs().max()) < 1e-9
    assert torch.equal(d.a1[~b], d.Atil[~b])                                               # a non-binding row: untouched
    assert bool((d.a1 <= d.Atil + 1e-15).all()) and bool(((d.a1 == 0) & (d.Atil > 0) & d.E)[b].any())   # never adds mass; exact zeros on the relation
    _, d_row, _ = _capped("row")
    assert torch.equal(d_row.cap_binds, b)                                                 # the SAME decision as the whole-row cap
    assert float((d_row.a1[off] - d_row.A_sm[off]).abs().max()) > 1e-6                     # which does touch the tail


def test_edge_cases_budget_is_never_negative_and_the_tail_dominates_fallback_is_a_switch():
    from marsea.normalizer import unit_cap_relation, apply_unit_cap, row_masses, unit_cap
    norm, d, vis = _capped("relation")
    assert float(((d.A_sm * d.E).sum(-1)).min()) >= 0.0                                    # edge case 1: a sum of softmax entries
    # a relation with ZERO softmax mass but paid by the column programme: projected to nothing, the row is its softmax
    A_sm = torch.tensor([[[[0.0, 0.6, 0.4]]]], dtype=DT); E = torch.tensor([[[[True, False, False]]]]); Atil = torch.tensor([[[[0.5, 0.6, 0.4]]]], dtype=DT)
    ex, sm = row_masses(Atil, A_sm, E)
    a1, th, took = unit_cap_relation(Atil, A_sm, E, ex)
    assert a1.tolist() == [[[[0.0, 0.6, 0.4]]]] and bool(took.all())
    # edge case 2: off by default; when switched on, a tail-dominated row gets the whole-row cap, the others do not
    norm.cap_fallback_tail_dominates = True
    visb = vis.expand_as(d.E); ex, sm = row_masses(d.Atil, d.A_sm, d.E)
    a1_fb, _, _ = apply_unit_cap(norm, d.Atil, d.A_sm, d.E, visb, ex, sm)
    a1_row, _, _ = unit_cap(d.Atil, visb, ex, sm)
    rel_sm = (d.A_sm * d.E).sum(-1); fb = (d.A_sm.sum(-1) - rel_sm) > rel_sm
    assert bool(fb.any()) and bool((~fb).any())
    assert torch.equal(a1_fb[fb], a1_row[fb]) and torch.equal(a1_fb[~fb], d.a1[~fb])


def test_gradients_flow_through_the_relation_only_cap_and_the_chunked_path_refuses():
    from marsea.chunked import marsea_chunked_attention
    norm, Q, K_kv, S, vis, n_p = _setup(rho=0.4, cap_mode="relation", relation_per_head=6)
    A, d = norm.normalize(S, vis, K_kv, Q, State(), n_prefill=n_p)
    A.pow(2).sum().backward()
    assert all(float(dict(norm.named_parameters())[n_].grad.abs().sum()) > 0 for n_ in ("relation.U", "relation.b0", "tauK.fc2.weight", "tauQ.fc2.weight"))
    V = torch.randn(1, K_kv.shape[1], S.shape[-1], D, dtype=DT)
    with pytest.raises(NotImplementedError), torch.no_grad():
        marsea_chunked_attention(norm, Q, K_kv, V, vis, State(), chunk=16)
