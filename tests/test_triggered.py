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
        if norm.tail_share_head is not None:
            for p_ in norm.tail_share_head.parameters(): p_.add_(0.5 * torch.randn_like(p_))   # a tail share that really varies
    return norm, Q, K_kv, S, vis, n_p


def _by_decoding(norm, Q, K_kv, S, vis, n_p, K_ret):
    """the reference: prefill the prompt, then one decode_step per answer row."""
    T = S.shape[-1]; rows = []
    with torch.no_grad():
        A_p, d_p = norm.normalize(S[:, :, :n_p, :n_p], vis[:, :, :n_p, :n_p], K_kv[:, :, :n_p], Q[:, :, :n_p], State(),
                                  prompt_rows=(n_p if norm.relation_answer_rows_only else None))   # the prefill: every row is a prompt row
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
                                      (64, dict(cap_mode="relation")), (3, dict(cap_mode="relation")),
                                      (64, dict(cap_mode="relation", tail_share=0.3)),
                                      (64, dict(relation_topk=3)), (64, dict(relation_from_scores=True, relation_topk=3, relation_answer_rows_only=True, direction="column", tau_col=3.0, lam_col=0.2)), (3, dict(relation_topk=2, relation_answer_rows_only=True)),
                                      (64, dict(relation_heads=[1, 4], relation_topk=4, relation_answer_rows_only=True, cap_mode="relation",
                                                tail_share_learned=True, tail_share_min=0.2, tail_share_init=0.25, tau_i_floor=1.0,
                                                size_aware_tau_i=True, trigger_tau=True, relation_per_head=6, per_head=6, head_block=2)), (64, dict(cap_mode="relation", tail_share=0.3, tau_i_floor=1.0, size_aware_tau_i=True, trigger_tau=True)), (3, dict(cap_mode="relation", tail_share_learned=True)), (64, dict(cap_mode="relation", tail_share_learned=True, tail_share_min=0.2, tail_share_init=0.25, tau_i_floor=1.0)),
                                      (64, dict(cap_mode="relation", tail_share_learned=True, size_aware_tau_i=True, trigger_tau=True, relation_per_head=6, per_head=6, head_block=2)),
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


def test_edge_cases_budget_is_never_negative_and_a_zero_budget_relation_binds():
    from marsea.normalizer import unit_cap_relation, row_masses
    norm, d, vis = _capped("relation")
    assert float(((d.A_sm * d.E).sum(-1)).min()) >= 0.0                                    # edge case 1: a sum of softmax entries
    # a relation with ZERO softmax mass but paid by the column programme: projected to nothing, the row is its softmax
    A_sm = torch.tensor([[[[0.0, 0.6, 0.4]]]], dtype=DT); E = torch.tensor([[[[True, False, False]]]]); Atil = torch.tensor([[[[0.5, 0.6, 0.4]]]], dtype=DT)
    ex, sm = row_masses(Atil, A_sm, E)
    a1, th, took = unit_cap_relation(Atil, A_sm, E, ex)
    assert a1.tolist() == [[[[0.0, 0.6, 0.4]]]] and bool(took.all())


def test_gradients_flow_through_the_relation_only_cap_and_the_chunked_path_refuses():
    from marsea.chunked import marsea_chunked_attention
    norm, Q, K_kv, S, vis, n_p = _setup(rho=0.4, cap_mode="relation", relation_per_head=6)
    A, d = norm.normalize(S, vis, K_kv, Q, State(), n_prefill=n_p)
    A.pow(2).sum().backward()
    assert all(float(dict(norm.named_parameters())[n_].grad.abs().sum()) > 0 for n_ in ("relation.U", "relation.b0", "tauK.fc2.weight", "tauQ.fc2.weight"))
    V = torch.randn(1, K_kv.shape[1], S.shape[-1], D, dtype=DT)
    with pytest.raises(NotImplementedError), torch.no_grad():
        marsea_chunked_attention(norm, Q, K_kv, V, vis, State(), chunk=16)


def test_a_padded_slot_on_a_column_the_row_cannot_see_does_not_poison_the_gradient():
    """P7, step 751 (2026-09-21): one head's relation at an answer row had grown to the whole row, so the block's slot count
    reached past what the EARLIER answer rows can see; their padded slots sat on invisible columns (score -inf), the
    trigger-time tau head read s_trig - max = -inf and returned NaN there.  The forward masks those slots out; the backward
    of the head is 0 * NaN, and every parameter below went NaN (three identical retries, run stopped)."""
    norm, Q, K_kv, S, vis, n_p = _setup(trigger_tau=True)
    T = S.shape[-1]
    lg = torch.full_like(S, -1.0)
    lg[:, :, T - 1, :] = 1.0                                   # the last answer row relates to every key it sees
    lg[:, :, n_p, 3] = 1.0                                     # the first one to a single key: its padded slots run past column n_p
    Q = Q.clone().requires_grad_(True)
    A, d = norm.normalize(S, vis, K_kv, Q, State(), n_prefill=n_p, logits_override=lg)
    assert bool(torch.isfinite(A).all()) and int(d.E[:, :, T - 1].sum(-1).min()) > n_p
    assert bool(torch.isfinite(d.extra["tau_trig_rows"]).all())
    A.pow(2).sum().backward()
    bad = [n_ for n_, p_ in norm.named_parameters() if p_.grad is not None and not bool(torch.isfinite(p_.grad).all())]
    assert bad == [] and bool(torch.isfinite(Q.grad).all()), bad
    assert float(norm.tauK_trig.fc2.weight.grad.abs().sum()) > 0


def test_a_tail_share_lets_the_relation_keep_mass_and_scales_the_tail_without_a_zero():
    """owner's decision, 2026-09-21: budget = R + lambda T; the tail pays for what the relation kept by ONE factor per row."""
    lam = 0.3
    norm, d, vis = _capped("relation", tail_share=lam)
    _, d0, _ = _capped("relation")                                                        # lambda = 0: same relation, same Atil
    assert torch.equal(d.E, d0.E) and torch.equal(d.Atil, d0.Atil)
    visb = vis.expand_as(d.E); off = visb & ~d.E; Ef = d.E.to(DT)
    R = (d.A_sm * Ef).sum(-1); T_ = (d.A_sm * off).sum(-1); W = (d.Atil * Ef).sum(-1); rel = (d.a1 * Ef).sum(-1); ex = W - R
    over = ex > 1e-9
    assert int(d.cap_binds.sum()) > 5 and int((over & ~d.cap_binds).sum()) > 5, "needs rows in BOTH over-paid regimes"
    # the tail: one factor per row, in [1 - lambda, 1]; its support is untouched (nothing is thresholded)
    fac = (d.a1 * off).sum(-1) / T_.clamp_min(1e-300)
    assert float((d.a1 - d.A_sm * fac.unsqueeze(-1))[off].abs().max()) < 1e-12
    assert float(fac[T_ > 0].min()) >= 1 - lam - 1e-9 and float(fac[T_ > 0].max()) <= 1 + 1e-12
    assert torch.equal((d.a1 > 0) & off, (d.A_sm > 0) & off)
    # the relation: within its budget, above its softmax mass wherever it was over-paid -- membership can now GAIN mass
    assert bool((rel <= R + lam * T_ + 1e-9).all()) and bool((rel[over & (T_ > 1e-6)] > R[over & (T_ > 1e-6)]).all())
    b = d.cap_binds
    assert float((rel[b] - (R + lam * T_)[b]).abs().max()) < 1e-9                          # projected onto the budget ...
    assert bool(((d.a1 == 0) & (d.Atil > 0) & d.E)[b].any())                               # ... with exact zeros ON the relation
    mid = over & ~b
    assert torch.equal(d.a1[mid][d.E[mid]], d.Atil[mid][d.E[mid]])                         # in between: the relation keeps Atil whole
    assert float((d.a1.sum(-1) - d.A_sm.sum(-1))[over].abs().max()) < 1e-9                 # the row: its softmax mass
    assert torch.equal(d.a1[~over], d.Atil[~over])                                         # not over-paid: untouched
    assert bool((d.a1 <= d.Atil + 1e-15).all())                                            # a cap never adds mass
    assert int(b.sum()) < int(d0.cap_binds.sum())                                          # fewer rows are projected than at lambda = 0
    # continuity in lambda: a vanishing share is the relation-only cap
    _, d_eps, _ = _capped("relation", tail_share=1e-9)
    assert float((d_eps.a1 - d0.a1).abs().max()) < 1e-7


def test_the_learned_tail_share_is_bounded_starts_small_trains_and_needs_the_relation_cap():
    from marsea.heads import TailShare
    from marsea.normalizer import MarSeaNormalizer
    hf = TailShare(D, lam_max=0.5, lam_init=0.25, per_head=6, lam_min=0.2).to(DT)             # the floored share (owner, 2026-09-21)
    lf = hf(torch.randn(1, 6, 9, D, dtype=DT), 50 * torch.randn(1, 6, 9, TailShare.N_SCALAR, dtype=DT))
    assert float(lf.min()) >= 0.2 and float(lf.max()) <= 0.5 and abs(float(hf(torch.zeros(1, 6, 1, D, dtype=DT), torch.zeros(1, 6, 1, TailShare.N_SCALAR, dtype=DT)).mean()) - 0.25) < 0.02
    h = TailShare(D, lam_max=0.5, lam_init=0.05, per_head=6).to(DT)
    lam = h(torch.randn(1, 6, 9, D, dtype=DT), torch.randn(1, 6, 9, TailShare.N_SCALAR, dtype=DT).abs())
    assert float(lam.min()) > 0 and float(lam.max()) < 0.5 and abs(float(lam.mean()) - 0.05) < 0.01
    with pytest.raises(AssertionError):
        MarSeaNormalizer(D, R, tail_share=0.2)                                             # cap_mode = "row"
    with pytest.raises(AssertionError):
        MarSeaNormalizer(D, R, cap_mode="relation", tail_share=0.2, tail_share_learned=True)
    norm, Q, K_kv, S, vis, n_p = _setup(rho=0.4, cap_mode="relation", tail_share_learned=True, relation_per_head=6, per_head=6)
    A, d = norm.normalize(S, vis, K_kv, Q, State(), n_prefill=n_p)
    A.pow(2).sum().backward()
    g = dict(norm.named_parameters())
    assert all(float(g[n_].grad.abs().sum()) > 0 for n_ in ("tail_share_head.w1", "tail_share_head.w2", "tail_share_head.b2", "relation.U"))
    w, o = norm.new_module_params()
    assert any(p_ is g["tail_share_head.w1"] for p_ in w) and any(p_ is g["tail_share_head.b2"] for p_ in o)


def test_a_floored_tau_i_makes_step_2_mass_preserving():
    """tau_i < 1 is a mass discount on the relation (final = tau_i x Atil_E, the rest leaves the row); floored at 1, step 2
    only sharpens and the relation ends on its quota exactly."""
    norm, d, vis = _capped("relation", tail_share=0.3, tau_i_floor=1.0)
    Ef = d.E.to(DT); has = d.E.any(-1)
    assert float(d.tau_i.min()) > 1.0 and abs(float(d.tau_i.mean()) - 1.1) < 0.2
    assert float(((d.A * Ef).sum(-1) - (d.a1 * Ef).sum(-1))[has].abs().max()) < 1e-9
    norm0, d0, _ = _capped("relation", tail_share=0.3)                                     # the spec's floor (0.05), tau_i pushed below 1
    with torch.no_grad():
        norm0.tauQ.set_init_tau(0.6)
        A0, d0 = norm0.normalize(*_capped_inputs())
    Ef0 = d0.E.to(DT)
    assert float(((d0.a1 * Ef0).sum(-1) - (d0.A * Ef0).sum(-1)).max()) > 0.05              # mass that simply vanished
    from marsea.heads import TauQ
    q = TauQ(D, tau_min=1.0); q.set_init_tau(1.0)                                          # train.py's phase_b_init call must not kill it
    assert abs(float(torch.nn.functional.softplus(q.last_bias).mean()) - 0.1) < 1e-6


def _capped_inputs():
    norm, Q, K_kv, S, vis, n_p = _setup(rho=0.4, cap_mode="relation", tail_share=0.3)
    return S, vis, K_kv, Q, State()


def test_partition_relation_options_shape_E_as_specified():
    """owner's decision (2026-09-22): a per-row budget, no relation on non-retrieval heads, prompt rows without a relation."""
    norm, Q, K_kv, S, vis, n_p = _setup(rho=0.4, relation_heads=[0, 2, 5], relation_topk=3, relation_answer_rows_only=True)
    with torch.no_grad():
        A, d = norm.normalize(S, vis, K_kv, Q, State(), n_prefill=n_p)
    T = S.shape[-1]; E = d.E[0]
    assert bool((E[[1, 3, 4]] == False).all())                                       # heads without a relation
    assert bool((E[:, :n_p] == False).all())                                          # prompt rows carry none
    sizes = E[[0, 2, 5], n_p:].sum(-1)
    assert bool((sizes == 3).all()), sizes                                             # exactly k per answer row on the relation heads
    assert torch.equal(A[0, [1, 3, 4]], d.A_sm[0, [1, 3, 4]])                         # a head without a relation IS softmax
    # the full-sequence form with prompt_rows agrees on E
    with torch.no_grad():
        A2, d2 = norm.normalize(S, vis, K_kv, Q, State(), prompt_rows=n_p)
    assert torch.equal(d2.E, d.E)
    # top-k alone: every visible row of every head has k members (or all of a shorter row)
    norm2, Q, K_kv, S, vis, n_p = _setup(rho=0.4, relation_topk=5)
    with torch.no_grad():
        _, d3 = norm2.normalize(S, vis, K_kv, Q, State())
    n_vis = vis.expand_as(d3.E).sum(-1)
    from marsea.relation import sink_pair_mask
    cand = (vis.expand_as(d3.E) & ~sink_pair_mask(S.shape[-2], S.shape[-1], S.device)).sum(-1)
    assert bool((d3.E.sum(-1) == torch.minimum(cand, torch.full_like(cand, 5))).all())
    # gradients still reach the relation through the top-k gate
    norm3, Q, K_kv, S, vis, n_p = _setup(rho=0.4, relation_topk=4, relation_answer_rows_only=True)
    A, d = norm3.normalize(S, vis, K_kv, Q, State(), n_prefill=n_p)
    A.pow(2).sum().backward()
    assert any(p_.grad is not None and float(p_.grad.abs().sum()) > 0 for n_, p_ in norm3.relation.named_parameters() if n_.startswith("U"))



# --------------------------------------------------------------------------- MarSea v3: the row-constrained one-end form
@pytest.mark.parametrize("lam,tau,k", [(0.0, 2.0, 4), (0.25, 4.0, 8), (0.5, 1.0, 3)])
def test_row_constrained_form_mass_rules_and_decode_equality(lam, tau, k):
    """owner's decision (2026-09-22): the relation lives on the many-end (answer rows); each row commits its relation mass
    (R_i + lam T_i) to ONE key by sparsemax(tau S); the tail is scaled by 1 - lam; no column programme.  Row-local, so a
    decode step equals the teacher-forced row exactly."""
    norm, Q, K_kv, S, vis, n_p = _setup(rho=0.4, direction="row", tau_row=tau, lam_row=lam, relation_topk=k,
                                        relation_answer_rows_only=True, relation_heads=[0, 2, 5])
    with torch.no_grad():
        A, d = norm.normalize(S, vis, K_kv, Q, State(), n_prefill=n_p)
    T = S.shape[-1]; E = d.E; Ef = E.to(DT); visb = vis.expand_as(E)
    assert d.extra["direction"] == "row" and bool((E[:, :, :n_p] == False).all()) and bool((E[:, [1, 3, 4]] == False).all())
    has = E.any(-1)
    # rows: one unit kept; relation mass = R + lam T; the tail scaled by (1 - lam); exact zeros on the relation
    R = (d.A_sm * Ef).sum(-1); Tm = (d.A_sm * (1 - Ef) * visb).sum(-1)
    assert float((A.sum(-1) - 1.0)[visb.any(-1)].abs().max()) < 1e-9
    assert float(((A * Ef).sum(-1) - (R + lam * Tm))[has].abs().max()) < 1e-9
    off = visb & ~E
    assert float((A - d.A_sm * (1 - lam))[off & has.unsqueeze(-1)].abs().max()) < 1e-12
    assert torch.equal(A[~has], d.A_sm[~has])                                            # a row without a relation is softmax
    assert bool(((A == 0) & E).any()) or k <= 1                                           # sparsemax zeros inside the relation
    top = (A * Ef).amax(-1) / (R + lam * Tm).clamp_min(1e-12)
    assert float(top[has].min()) > 0.5 or tau < 2                                         # commits most of it to one key at tau >= 2
    # the full-sequence form with prompt_rows and the decode step agree with it row by row
    with torch.no_grad():
        A2, _ = norm.normalize(S, vis, K_kv, Q, State(), prompt_rows=n_p)
        A_p, A_dec = _by_decoding(norm, Q, K_kv, S, vis, n_p, 64)
    assert float((A2 - A).abs().max()) < 1e-12
    assert float((A_dec - A[:, :, n_p:]).abs().max()) < 1e-9 and torch.equal(A_p, A[:, :, :n_p, :n_p])
    # gradients reach the relation
    norm.zero_grad(); A3, _ = norm.normalize(S, vis, K_kv, Q, State(), n_prefill=n_p); A3.pow(2).sum().backward()
    assert any(p_.grad is not None and float(p_.grad.abs().sum()) > 0 for n_, p_ in norm.relation.named_parameters() if n_.startswith("U"))


def test_row_constrained_whole_model_teacher_forced_equals_cached_decoding():
    from marsea.evaluate import teacher_forced_pass
    from marsea.backbone import patch_model, MarSeaContext
    from marsea.baselines import make_normalizer
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=8,
                      num_key_value_heads=2, max_position_embeddings=512, attn_implementation="eager")
    model = Qwen2ForCausalLM(cfg).double(); ctx = MarSeaContext(mode="dense")
    torch.manual_seed(1)
    patch_model(model, [1, 2], lambda l: make_normalizer("marsea", 8, relation_per_head=8, per_head=8, direction="row", tau_row=3.0, lam_row=0.25,
                                                         relation_topk=4, relation_answer_rows_only=True).double(), ctx)
    model.eval(); ctx.triggered = True
    g = torch.Generator().manual_seed(3); prompt = torch.randint(0, 97, (30,), generator=g).tolist(); gold = torch.randint(0, 97, (6,), generator=g).tolist()
    ids = torch.tensor([prompt + gold])
    ctx.n_prefill = len(prompt)
    with torch.no_grad():
        full = model(input_ids=ids, use_cache=False).logits[0, len(prompt) - 1:-1]
    ctx.n_prefill = None
    # cached decoding: prefill the prompt, then one token at a time
    from marsea.evaluate import generate_greedy
    ctx.set_generation = None
    import marsea.evaluate as ev
    outs = []
    with torch.no_grad():
        ctx.generation = True; ctx.decode_caches = {}
        pre = model(input_ids=ids[:, :len(prompt)], use_cache=True)
        past = pre.past_key_values; outs.append(pre.logits[0, -1])
        for t in range(len(gold) - 1):
            o = model(input_ids=ids[:, len(prompt) + t: len(prompt) + t + 1], past_key_values=past, use_cache=True)
            past = o.past_key_values; outs.append(o.logits[0, -1])
        ctx.generation = False; ctx.decode_caches = {}
    dec = torch.stack(outs)
    assert float((dec - full).abs().max()) < 1e-7


# --------------------------------------------------------------------------- MarSea v3: the column-constrained one-end form
@pytest.mark.parametrize("K_ret,lam,tau", [(64, 0.0, 2.0), (3, 0.25, 4.0), (64, 0.5, 1.0)])
def test_column_constrained_form_equals_decoding_and_obeys_its_mass_rules(K_ret, lam, tau):
    """owner's design (2026-09-22): QA -- answer rows are the one-end, supporting tokens the many-end.  Column j hands
    R_j + lam T_j (over the rows so far) to its members by sparsemax(tau S) at trigger time; a generated non-member row of
    a column with a relation is emitted at (1 - lam) A_sm; no cap after (the softmax is the unit normalisation).
    Teacher-forced == decode."""
    norm, Q, K_kv, S, vis, n_p = _setup(rho=0.4, direction="column", tau_col=tau, lam_col=lam, relation_topk=4,
                                        relation_answer_rows_only=True, relation_heads=[0, 2, 5])
    A_p, A_dec = _by_decoding(norm, Q, K_kv, S, vis, n_p, K_ret)
    with torch.no_grad():
        A, d = norm.normalize(S, vis, K_kv, Q, State(), n_prefill=n_p, decode_K_ret=K_ret)
    T = S.shape[-1]; E = d.E; visb = vis.expand_as(E)
    assert float((A[:, :, n_p:] - A_dec).abs().max()) < 1e-9 and torch.equal(A[:, :, :n_p, :n_p], A_p)
    assert bool((E[:, :, :n_p] == False).all()) and bool((E[:, [1, 3, 4]] == False).all())
    # every row within one unit; a head without a relation is softmax; prompt rows untouched
    assert torch.equal(A[:, [1, 3, 4]], d.A_sm[:, [1, 3, 4]]) and torch.equal(A[:, :, :n_p], d.A_sm[:, :, :n_p])
    assert not bool(d.cap_binds.any())                                                  # no cap after the column programme
    # the rules on every answer row: members take budget x share (a member row may carry MORE than one unit), non-members
    # of an ACTIVE column are scaled by (1 - lam), everything else is softmax
    Asm = d.A_sm
    for h in (0, 2, 5):
        for t in range(n_p, T):
            members_so_far = E[0, h, :t + 1]                                         # [t+1, n]: rows <= t
            active = members_so_far.any(0)
            for j in range(t + 1):
                if bool(E[0, h, t, j]):
                    R = float((Asm[0, h, :t + 1, j] * members_so_far[:, j]).sum()); Tm = float((Asm[0, h, :t + 1, j] * ~members_so_far[:, j]).sum())
                    assert A[0, h, t, j] <= (R + lam * Tm) + 1e-9
                elif bool(active[j]):
                    assert abs(float(A[0, h, t, j] - (1 - lam) * Asm[0, h, t, j])) < 1e-9
                else:
                    assert abs(float(A[0, h, t, j] - Asm[0, h, t, j])) < 1e-12
    # the ST gradient reaches the relation
    norm.zero_grad(); A3, _ = norm.normalize(S, vis, K_kv, Q, State(), n_prefill=n_p, decode_K_ret=K_ret); A3.pow(2).sum().backward()
    assert any(p_.grad is not None and float(p_.grad.abs().sum()) > 0 for n_, p_ in norm.relation.named_parameters() if n_.startswith("U"))


def test_auto_direction_switches_per_sequence():
    """direction "auto": the backbone copies ctx.relation_direction into the normaliser before each forward."""
    norm, Q, K_kv, S, vis, n_p = _setup(rho=0.4, direction="auto", tau_row=3.0, tau_col=3.0, relation_topk=4, relation_answer_rows_only=True)
    with torch.no_grad():
        norm.active_direction = "row"; A_r, d_r = norm.normalize(S, vis, K_kv, Q, State(), n_prefill=n_p)
        norm.active_direction = "column"; A_c, d_c = norm.normalize(S, vis, K_kv, Q, State(), n_prefill=n_p)
    assert d_r.extra.get("direction") == "row" and d_c.extra.get("triggered") is not None
    assert float((A_r - A_c).abs().max()) > 1e-3                                      # the two forms differ on the same input
    src = open("marsea/backbone.py").read(); tr = open("marsea/train.py").read()
    assert 'self.normalizer.active_direction = ctx.relation_direction or "row"' in src
    assert 'ctx.relation_direction = "column" if s.source in ("musique", "hotpot") else "row"' in tr


def test_relation_from_scores_is_the_rows_top_k_by_attention():
    norm, Q, K_kv, S, vis, n_p = _setup(rho=0.4, direction="row", tau_row=4.0, lam_row=0.25, relation_topk=3, relation_answer_rows_only=True,
                                        relation_from_scores=True)
    with torch.no_grad():
        A, d = norm.normalize(S, vis, K_kv, Q, State(), n_prefill=n_p)
        A_p, A_dec = _by_decoding(norm, Q, K_kv, S, vis, n_p, 64)
    T = S.shape[-1]; visb = vis.expand_as(d.E)
    Sm = S.masked_fill(~visb, float("-inf")).clone(); Sm[..., 0] = float("-inf")           # the sink column never
    top = Sm.topk(3, dim=-1).indices
    want = torch.zeros_like(d.E).scatter(-1, top, True); want[:, :, :n_p] = False
    assert torch.equal(d.E, want)                                                          # exactly the top-3 by attention, answer rows only
    assert float((A_dec - A[:, :, n_p:]).abs().max()) < 1e-9
    src = (A * d.E.to(DT)).amax(-1)[:, :, n_p:]                                           # the winner takes most of R + lam T
    assert float(src.min()) > 0.2 and float(src.mean()) > 0.5



def test_auto_direction_whole_model_qa_column_form_trains_and_decodes_like_teacher_forcing():
    """the QA column-form pilot's path end to end on a tiny model (2026-09-23): direction "auto" + the anchored relation; a
    QA sequence (ctx.relation_direction = "column") takes a training forward with the aux capture, the block loss on its
    supporting-span blocks has a gradient into the relation head, and the same weights decode token by token (the column
    decode path) to the teacher-forced logits."""
    from types import SimpleNamespace
    from marsea.backbone import patch_model, MarSeaContext
    from marsea.baselines import make_normalizer
    from marsea.train import aux_block_loss, TrainConfig
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=8,
                      num_key_value_heads=2, max_position_embeddings=512, attn_implementation="eager")
    model = Qwen2ForCausalLM(cfg).double(); ctx = MarSeaContext(mode="dense")
    torch.manual_seed(1)
    patch_model(model, [1, 2], lambda l: make_normalizer("marsea", 8, relation_per_head=8, per_head=8, direction="auto", tau_row=1.0, lam_row=0.0,
                                                         tau_col=1.0, lam_col=0.0, relation_topk=4, relation_answer_rows_only=True,
                                                         relation_anchor_scores=True).double(), ctx)
    model.marsea_ctx = ctx
    g = torch.Generator().manual_seed(3); prompt = torch.randint(0, 97, (30,), generator=g).tolist(); gold = torch.randint(0, 97, (5,), generator=g).tolist()
    ids = torch.tensor([prompt + gold]); n_p = len(prompt)
    rows = [n_p - 1 + t for t in range(len(gold))]
    blocks = [dict(name="answer", rows=rows, spans=[(4, 9), (17, 21)]), dict(name="passage0", rows=[], spans=[(10, 16)])]
    seq = SimpleNamespace(input_ids=ids, labels=torch.tensor([[-100] * n_p + gold]), source="hotpot", index=0, n_label_tokens=len(gold), blocks=blocks)
    # -- the training forward of a QA sequence: column form, aux capture, block loss with a gradient into the relation head
    model.train(); ctx.triggered = True; ctx.n_prefill = n_p; ctx.relation_direction = "column"
    ctx.aux_capture = True; ctx.aux_qk = {}
    out = model(input_ids=ids, use_cache=False)
    assert all(model.model.layers[l].self_attn.normalizer.active_direction == "column" for l in (1, 2))
    loss, w = aux_block_loss(model, ctx, seq, TrainConfig(aux_block_weight=1.0, aux_anneal_steps=100, phase_a_steps=0), step=0)
    assert loss is not None and w == 1.0 and torch.isfinite(loss)
    (out.logits[0, n_p - 1:-1].log_softmax(-1).gather(-1, torch.tensor(gold).view(-1, 1)).mean().neg() + loss).backward()
    nm = model.model.layers[1].self_attn.normalizer
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in nm.relation.parameters())
    assert sum(float(p.grad.abs().sum()) for n_, p in nm.relation.named_parameters() if n_.startswith(("U", "V"))) > 0
    model.zero_grad(); ctx.aux_capture = False; ctx.aux_qk = None
    # -- teacher forcing == cached decoding through the column decode path, on the same weights
    model.eval()
    with torch.no_grad():
        full = model(input_ids=ids, use_cache=False).logits[0, n_p - 1:-1]
    ctx.n_prefill = None
    outs = []
    with torch.no_grad():
        ctx.generation = True; ctx.decode_caches = {}
        pre = model(input_ids=ids[:, :n_p], use_cache=True); past = pre.past_key_values; outs.append(pre.logits[0, -1])
        for t in range(len(gold) - 1):
            o = model(input_ids=ids[:, n_p + t: n_p + t + 1], past_key_values=past, use_cache=True)
            past = o.past_key_values; outs.append(o.logits[0, -1])
        ctx.generation = False; ctx.decode_caches = {}
    assert float((torch.stack(outs) - full).abs().max()) < 1e-7
    # -- and the same model, told the next sequence is RULER, runs the row form
    ctx.relation_direction = "row"; ctx.n_prefill = n_p
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    assert all(model.model.layers[l].self_attn.normalizer.active_direction == "row" for l in (1, 2))


# --------------------------------------------------------------------------- MarSea on a softmax-one base (owner, 2026-09-23)
def test_softmax_one_base_row_form_masses_and_decoding():
    """base="softmax_one": step 1 is exp(s) / (1 + sum exp).  Off E and with E empty the row IS softmax-one (B1); on E the
    relation carries R_i (+ lam T_i) of THAT base; rows sum to R_i + T_i < 1; the row form stays row-local (TF == decoding)."""
    from marsea.baselines import SoftmaxOneNorm
    norm, Q, K_kv, S, vis, n_p = _setup(rho=0.4, direction="row", tau_row=1.0, lam_row=0.25, relation_topk=4, relation_answer_rows_only=True,
                                        base="softmax_one")
    with torch.no_grad():
        A, d = norm.normalize(S, vis, K_kv, Q, State(), n_prefill=n_p)
        A_p, A_dec = _by_decoding(norm, Q, K_kv, S, vis, n_p, 64)
        A_b1, _ = SoftmaxOneNorm().normalize(S, vis)
    visb = vis.expand_as(d.E)
    assert torch.equal(d.A_sm, A_b1.to(d.A_sm.dtype)) or float((d.A_sm - A_b1).abs().max()) < 1e-12   # step 1 IS B1
    assert float(A.sum(-1).max()) < 1.0                                                     # every row attends partly to nothing
    has = d.E.any(-1)
    assert float((A.sum(-1) - d.A_sm.sum(-1))[has].abs().max()) < 1e-9                       # ...and keeps softmax-one's total: R + T
    assert torch.equal(A[:, :, :n_p], A_b1[:, :, :n_p].to(A.dtype))                          # prompt rows: no relation -> B1 exactly
    off = ~d.E & visb & has.unsqueeze(-1)
    assert float((A[off] - (1 - 0.25) * d.A_sm[off]).abs().max()) < 1e-12                    # tail scaled by 1 - lam
    R = (d.A_sm * d.E.to(DT)).sum(-1); T = (d.A_sm * (~d.E & visb).to(DT)).sum(-1)
    assert float(((A * d.E.to(DT)).sum(-1) - (R + 0.25 * T))[has].abs().max()) < 1e-9        # the relation holds R + lam T of the base
    assert float((A_dec - A[:, :, n_p:]).abs().max()) < 1e-9                                 # row-local: decoding == teacher forcing


def test_softmax_one_base_whole_model_teacher_forced_equals_cached_decoding():
    from marsea.backbone import patch_model, MarSeaContext
    from marsea.baselines import make_normalizer
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=8,
                      num_key_value_heads=2, max_position_embeddings=512, attn_implementation="eager")
    model = Qwen2ForCausalLM(cfg).double(); ctx = MarSeaContext(mode="dense")
    torch.manual_seed(1)
    patch_model(model, [1, 2], lambda l: make_normalizer("marsea", 8, relation_per_head=8, per_head=8, direction="row", tau_row=1.0, lam_row=0.0,
                                                         relation_topk=8, relation_answer_rows_only=True, relation_anchor_scores=True,
                                                         base="softmax_one").double(), ctx)
    model.eval(); ctx.triggered = True
    g = torch.Generator().manual_seed(3); prompt = torch.randint(0, 97, (30,), generator=g).tolist(); gold = torch.randint(0, 97, (6,), generator=g).tolist()
    ids = torch.tensor([prompt + gold]); ctx.n_prefill = len(prompt)
    with torch.no_grad():
        full = model(input_ids=ids, use_cache=False).logits[0, len(prompt) - 1:-1]
    ctx.n_prefill = None; outs = []
    with torch.no_grad():
        ctx.generation = True; ctx.decode_caches = {}
        pre = model(input_ids=ids[:, :len(prompt)], use_cache=True); past = pre.past_key_values; outs.append(pre.logits[0, -1])
        for t in range(len(gold) - 1):
            o = model(input_ids=ids[:, len(prompt) + t: len(prompt) + t + 1], past_key_values=past, use_cache=True)
            past = o.past_key_values; outs.append(o.logits[0, -1])
        ctx.generation = False; ctx.decode_caches = {}
    assert float((torch.stack(outs) - full).abs().max()) < 1e-7
    # with the relation forced empty the whole model IS the B1 arm
    from marsea.baselines import SoftmaxOneNorm
    ctx.force_empty = True; ctx.n_prefill = len(prompt)
    with torch.no_grad():
        off = model(input_ids=ids, use_cache=False).logits
    ctx.force_empty = False
    model_b1 = Qwen2ForCausalLM(cfg).double(); model_b1.load_state_dict({k: v for k, v in model.state_dict().items() if "normalizer" not in k}, strict=False)
    ctx_b1 = MarSeaContext(mode="dense"); patch_model(model_b1, [1, 2], lambda l: SoftmaxOneNorm().double(), ctx_b1); model_b1.eval()
    with torch.no_grad():
        b1 = model_b1(input_ids=ids, use_cache=False).logits
    assert float((off - b1).abs().max()) < 1e-9


# --------------------------------------------------------------------------- column form on row-normalised scores (owner, 2026-09-25)
@pytest.mark.parametrize("K_ret", [64, 3])
def test_col_rownorm_members_compete_on_row_normalised_scores_and_decode_equals_teacher_forcing(K_ret):
    """col_rownorm: the members of column j compete on z_ij = log A_sm[i, j] (row-normalised) instead of raw S_ij, so a row
    with a globally high logit offset no longer wins every column.  Same quota / tail rules; TF == decode."""
    kw = dict(rho=0.4, direction="column", tau_col=0.75, lam_col=0.1, relation_topk=4, relation_answer_rows_only=True,
              relation_heads=[0, 2, 5], base="softmax_one")
    norm_r, Q, K_kv, S, vis, n_p = _setup(col_rownorm=True, **kw)
    norm_s, _, _, _, _, _ = _setup(col_rownorm=False, **kw)
    norm_s.load_state_dict(norm_r.state_dict())
    # give one answer row a large positive offset on ALL its scores: raw-S competition hands it every column it joins
    S2 = S.clone(); S2[:, :, n_p + 1, :] += 6.0
    with torch.no_grad():
        A_r, d_r = norm_r.normalize(S2, vis, K_kv, Q, State(), n_prefill=n_p, decode_K_ret=K_ret)
        A_s, d_s = norm_s.normalize(S2, vis, K_kv, Q, State(), n_prefill=n_p, decode_K_ret=K_ret)
    A_p, A_dec = _by_decoding(norm_r, Q, K_kv, S2, vis, n_p, K_ret)
    assert float((A_r[:, :, n_p:] - A_dec).abs().max()) < 1e-9 and torch.equal(A_r[:, :, :n_p, :n_p], A_p)   # TF == decode
    assert torch.equal(d_r.E, d_s.E)                                                    # the relation is the same; only the competition differs
    assert float((A_r - A_s).abs().max()) > 1e-6                                        # ...and it changes the assignment
    T = S.shape[-1]; E = d_r.E; Asm = d_r.A_sm
    # the offset row: under raw S it takes (nearly) the whole quota of every column it is a member of; under z it takes
    # its row-normalised share, which is NOT larger than the other members' whenever its A_sm on that key is not larger
    r = n_p + 1
    for h in (0, 2, 5):
        for t in range(r, T):
            for j in range(t + 1):
                mem = E[0, h, :t + 1, j].clone(); mem[t + 1:] = False
                if bool(E[0, h, r, j]) and int(mem.sum()) >= 2:
                    others = [i for i in range(t + 1) if bool(mem[i]) and i != r]
                    zr = float(torch.log(Asm[0, h, r, j])); zo = max(float(torch.log(Asm[0, h, i, j])) for i in others)
                    if zo > zr + 1e-6:                                                  # another member is row-normalised-better
                        assert float(A_r[0, h, r, j]) <= max(float(A_r[0, h, i, j]) for i in others) + 1e-9
    # rows without a relation and prompt rows: the base normalisation exactly
    assert torch.equal(A_r[:, [1, 3, 4]], Asm[:, [1, 3, 4]]) and torch.equal(A_r[:, :, :n_p], Asm[:, :, :n_p])
