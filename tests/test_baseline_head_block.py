"""The dense baselines evaluated head by head (2026-09-20).

B4_s0_E3pad died at 138 GB on a card it had to itself: the baselines' dense path built every [1, 12, T, T] fp32
tensor for all heads at once, and only MarSea's normaliser was head-blocked.  B2's MESH step would have been next.
A tiny random Qwen2 on CPU, in float64: the blocked forward must give the unblocked one's logits and kept tensors."""
import os, pathlib, subprocess, sys
import pytest, torch

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _tiny(arm, hb):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from marsea.backbone import patch_model, MarSeaContext
    from marsea.baselines import make_normalizer
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=8,
                      num_key_value_heads=2, max_position_embeddings=512, attn_implementation="eager")
    model = Qwen2ForCausalLM(cfg).double().eval()
    ctx = MarSeaContext(mode="dense")
    torch.manual_seed(1)
    patch_model(model, [1, 2], lambda l: make_normalizer(arm, 8).double(), ctx)
    ctx.phase_a = (arm == "B0"); ctx.baseline_head_block = hb
    model.marsea_ctx = ctx
    model.eval()                                                # AFTER patching, as load_arm does: the new modules start in training mode
    return model, ctx


@pytest.mark.parametrize("arm", ["B0", "B1", "B4", "B2"])
def test_blocked_baseline_forward_equals_the_full_one(arm):
    from marsea.evaluate import teacher_forced_pass
    ids = torch.randint(0, 97, (1, 96), generator=torch.Generator().manual_seed(3)).tolist()[0]
    sites = [(2, 1), (2, 6)]                                      # two kept heads, in different blocks at hb = 2
    out = {}
    for hb in (None, 2, 3):
        model, ctx = _tiny(arm, hb)
        if arm == "B2":                                             # its 1e-6 cost noise is drawn per block: silence it
            for l in (1, 2):
                model.model.layers[l].self_attn.normalizer.noise = 0.0
        loss, _, dg = teacher_forced_pass(model, ctx, ids[:-4], ids[-4:], 2, device="cpu", h_star=1, sites=sites)
        out[hb] = (loss, dg)
    ref_logits, ref_dg = out[None]
    assert "baseline_head_block" not in ref_dg[2].extra
    for hb in (2, 3):
        logits, dg = out[hb]
        assert abs(logits - ref_logits) < 1e-9 * max(1.0, abs(ref_logits)), (arm, hb, logits, ref_logits)   # the gold loss: every layer's output
        d, r = dg[2], ref_dg[2]
        assert d.extra["baseline_head_block"] == hb and d.extra["head_sub"] == [1, 6]
        assert d.A.shape == r.A.shape and d.A.shape[1] == 2, (d.A.shape, r.A.shape)
        assert torch.allclose(d.A, r.A, atol=1e-12, rtol=0)
        if r.A_sm is not None:
            assert torch.allclose(d.A_sm, r.A_sm, atol=1e-12, rtol=0)
        for k, v in r.extra.items():                                # row_sum, col_resid, a, b, supp: every head, in order
            if torch.is_tensor(v) and k != "S":
                assert d.extra[k].shape == v.shape and torch.allclose(d.extra[k].double(), v.double(), atol=1e-10), (arm, hb, k)
        assert torch.equal(d.extra["S"], r.extra["S"])


def test_not_blocked_when_it_must_not_be():
    """training (grad on), a MarSea arm, and a pass that keeps EVERY head all take the old path."""
    from marsea.evaluate import teacher_forced_pass
    model, ctx = _tiny("B1", 2)
    seen = []
    att = model.model.layers[2].self_attn
    orig = att._baseline_by_head_block
    att._baseline_by_head_block = lambda *a, **k: (seen.append(1), orig(*a, **k))[1]
    ids = torch.randint(0, 97, (1, 40))
    with torch.enable_grad():
        model(input_ids=ids)
    assert not seen
    with torch.no_grad():
        model(input_ids=ids)
    assert seen == [1]
    ctx.collect = True; ctx.keep_dense = True; ctx.keep_dense_head = None      # every head kept: the full path
    with torch.no_grad():
        model(input_ids=ids)
    assert seen == [1] and ctx.diags[2].A.shape[1] == 8


def test_the_queue_blocks_B2_and_B4_only_and_sets_the_allocator():
    s = (ROOT / "scripts/run_evalsuite.sh").read_text()
    assert 'case $arm in B2|B4) base="$base --baseline_head_block $BASELINE_HEAD_BLOCK";; esac' in s
    assert "export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" in s
    assert s.index("export PYTORCH_CUDA_ALLOC_CONF") < s.index("run_eval.py")
