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
        for p_ in norm.tauQ.parameters(): p_.add_(0.05 * torch.randn_like(p_))
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
                                      (64, dict(tau_i_pinned=True))])
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
def _tiny_model(head_block=None):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from marsea.backbone import patch_model, MarSeaContext
    from marsea.baselines import make_normalizer
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=8,
                      num_key_value_heads=2, max_position_embeddings=512, attn_implementation="eager")
    model = Qwen2ForCausalLM(cfg).double()
    ctx = MarSeaContext(mode="dense")
    torch.manual_seed(1)
    patch_model(model, [1, 2], lambda l: make_normalizer("marsea", 8, head_block=head_block, relation_per_head=8).double(), ctx)
    with torch.no_grad():
        for l in (1, 2):
            model.model.layers[l].self_attn.normalizer.relation.b0.fill_(0.0)     # about half of the pairs: a relation that acts
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
    assert float((tf[:, n_p - 1:] - dec).abs().max()) < 1e-8, float((tf[:, n_p - 1:] - dec).abs().max())
    assert float((full[:, n_p:] - dec[:, 1:]).abs().max()) > 1e-6                          # the form trained until now is NOT what decodes
    assert float((full[:, :n_p - 1] - tf[:, :n_p - 1]).abs().max()) > 1e-9                  # nor are its prompt rows the prefill's


def test_the_chunked_path_refuses_rather_than_silently_running_the_leaky_form():
    model, ctx = _tiny_model()
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
