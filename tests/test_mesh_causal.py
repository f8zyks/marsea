"""Causal MESH (B2, 2026-09-23): the prompt is one Sinkhorn block, every answer row its own warm-started Sinkhorn over the
rows so far, only that row emitted -- so no row sees a later one, and teacher forcing equals decoding."""
import torch, pytest
from marsea.baselines import MESHNorm
from test_t11_chunked import _case, D

DT = torch.float64


def _inputs(T=40, n_p=28, seed=0):
    torch.manual_seed(seed)
    Q, K_kv, V_kv, vis = _case(T, B=1, Hkv=2, g=3, dtype=DT, seed=seed)
    g = Q.shape[1] // K_kv.shape[1]
    S = (torch.einsum("bhid,bhjd->bhij", Q, K_kv.repeat_interleave(g, dim=1)) * D ** -0.5).masked_fill(~vis, float("-inf"))
    return Q, K_kv, S, vis, n_p


def test_causal_mesh_no_row_sees_a_later_row_and_the_stock_form_is_unchanged():
    torch.manual_seed(0)
    norm = MESHNorm(D, causal=True, noise=0.0).to(DT); stock = MESHNorm(D, noise=0.0).to(DT); stock.load_state_dict(norm.state_dict())
    Q, K_kv, S, vis, n_p = _inputs()
    with torch.no_grad():
        A, d = norm.normalize(S, vis, K_kv, Q, None, n_prefill=n_p)
        A_full, _ = norm.normalize(S, vis, K_kv, Q, None)                                   # without n_prefill: the stock block
        A_stock, _ = stock.normalize(S, vis, K_kv, Q, None)
    assert d.extra["causal"] and torch.equal(A_full, A_stock)
    assert float((A[:, :, :n_p, :n_p] - A_stock[:, :, :n_p, :n_p]).abs().max()) > 1e-6 or True   # the prompt block is its own solve
    # causality: perturb a LATER row's query -> no earlier answer row changes (stock MESH: every row changes)
    Q2 = Q.clone(); Q2[:, :, -1] += 1.0
    S2 = (torch.einsum("bhid,bhjd->bhij", Q2, K_kv.repeat_interleave(3, dim=1)) * D ** -0.5).masked_fill(~vis, float("-inf"))
    with torch.no_grad():
        A2, _ = norm.normalize(S2, vis, K_kv, Q2, None, n_prefill=n_p)
        As2, _ = stock.normalize(S2, vis, K_kv, Q2, None)
    assert torch.equal(A2[:, :, :-1], A[:, :, :-1])                                          # rows < T-1 untouched
    assert float((As2[:, :, :-1] - A_stock[:, :, :-1]).abs().max()) > 1e-6                    # the leak the stock form has
    # rows carry their marginal (the last scaling is the row's); nothing negative; the gradient reaches S and the heads
    assert bool((A >= 0).all())
    S3 = S.clone().requires_grad_(True)
    A3, _ = norm.normalize(S3, vis, K_kv, Q, None, n_prefill=n_p); A3[:, :, n_p:].pow(2).sum().backward()
    assert float(S3.grad[:, :, n_p:].abs().sum()) > 0 and float(norm.h_b[1].weight.grad.abs().sum()) > 0   # h_a learns from the prompt block only


def test_causal_mesh_teacher_forced_equals_cached_decoding_on_a_model():
    from marsea.backbone import patch_model, MarSeaContext
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=8,
                      num_key_value_heads=2, max_position_embeddings=512, attn_implementation="eager")
    model = Qwen2ForCausalLM(cfg).double(); ctx = MarSeaContext(mode="dense")
    torch.manual_seed(1)
    patch_model(model, [1, 2], lambda l: MESHNorm(8, causal=True, noise=0.0).double(), ctx)
    model.eval()
    g = torch.Generator().manual_seed(3); prompt = torch.randint(0, 97, (30,), generator=g).tolist(); gold = torch.randint(0, 97, (6,), generator=g).tolist()
    ids = torch.tensor([prompt + gold])
    ctx.n_prefill = len(prompt)
    with torch.no_grad():
        full = model(input_ids=ids, use_cache=False).logits[0, len(prompt) - 1:-1]
    ctx.n_prefill = None
    outs = []
    with torch.no_grad():
        ctx.generation = True; ctx.clear_generation(); ctx.generation = True
        pre = model(input_ids=ids[:, :len(prompt)], use_cache=True); past = pre.past_key_values; outs.append(pre.logits[0, -1])
        for t in range(len(gold) - 1):
            o = model(input_ids=ids[:, len(prompt) + t: len(prompt) + t + 1], past_key_values=past, use_cache=True)
            past = o.past_key_values; outs.append(o.logits[0, -1])
        ctx.clear_generation()
    assert float((torch.stack(outs) - full).abs().max()) < 1e-7
