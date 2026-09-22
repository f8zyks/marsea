"""
Sec. 6: backbone integration.  All seven arms run on the same borrowed pretrained decoder with only the
normalisation step of a subset of attention layers replaced.  Our code enters at exactly one line.

  * MarSeaAttention WRAPS the original Qwen2Attention / LlamaAttention (same nn.Linear objects, same names,
    so peft's LoRA targets them unchanged -- round-2 C-22); its forward is the original forward with the
    attention call replaced by `marsea_eager_forward`, whose softmax line is `normalizer.normalize(...)`.
  * Unpatched layers keep SDPA.  We never set config._attn_implementation="eager" globally (Sec. 6.1).
  * `vis` is built INSIDE the patched layer from the 2-D pad mask + the causal KV offset (Sec. 6.1 [v4.1]):
        vis[b,0,i,j] = pad[b,j] & pad[b, i + n_k - n_q] & (j <= i + n_k - n_q)
    never from the 4-D additive mask (version-fragile, C-6).  The 2-D mask reaches the layer through the
    MarSeaContext, set by a forward pre-hook on the base model (C-23: never rely on **kwargs plumbing).
  * S in bf16 under autocast -> fp32 inside the normaliser -> A cast back; the value-mix is bf16 (Sec. 6.3).
  * nu is handed from patched layer to patched layer through the context (Sec. 5.3), not detached.
"""
from __future__ import annotations
import inspect
import math
from dataclasses import dataclass, field
from typing import Callable, Optional
import torch
import torch.nn as nn

from .normalizer import State, Diagnostics, broadcast_vis
from .relation import repeat_kv
from .baselines import SoftmaxNorm, SoftmaxOneNorm, MESHNorm, RowEntmaxNorm

SUPPORTED_FAMILIES = ("qwen2", "llama")


# --------------------------------------------------------------------------- import-time assertions (Sec. 1.3, D-30)
def assert_transformers_contract(min_version: str = "4.53"):
    import transformers
    from packaging.version import Version
    v = Version(transformers.__version__)
    assert v >= Version(min_version), f"transformers {transformers.__version__} < {min_version} (D-30)"
    from transformers.models.qwen2 import modeling_qwen2 as mq
    from transformers.models.llama import modeling_llama as ml
    for mod in (mq, ml):
        sig = list(inspect.signature(mod.eager_attention_forward).parameters)
        assert sig[:7] == ["module", "query", "key", "value", "attention_mask", "scaling", "dropout"], \
            f"eager_attention_forward signature changed: {sig}"
        fsig = list(inspect.signature(mod.__dict__[[n for n in dir(mod) if n.endswith('Attention') and not n.startswith('_')][0]].forward).parameters)
        assert "hidden_states" in fsig and "position_embeddings" in fsig and "attention_mask" in fsig, fsig
    return transformers.__version__


def family_of(model) -> str:
    t = model.config.model_type
    assert t in SUPPORTED_FAMILIES, f"unsupported backbone family {t}"
    return t


def _modeling_module(family: str):
    if family == "qwen2":
        from transformers.models.qwen2 import modeling_qwen2 as m
    else:
        from transformers.models.llama import modeling_llama as m
    return m


# --------------------------------------------------------------------------- vis
def build_vis(pad_mask: Optional[torch.Tensor], B: int, n_q: int, n_k: int, device) -> torch.Tensor:
    """[B,1,n_q,n_k] bool.  pad_mask [B, n_k] (1 = real token) or None (no padding).  Right padding: a pad
    QUERY row is fully invisible (A = 0 there, excluded from every statistic).  Asserts every real row sees
    >= 1 key (a left-padded real row would NaN the row-softmax)."""
    i = torch.arange(n_q, device=device)[:, None]
    j = torch.arange(n_k, device=device)[None, :]
    off = n_k - n_q
    causal = (j <= i + off)                                                     # [n_q, n_k]
    if pad_mask is None:
        return causal[None, None].expand(B, 1, n_q, n_k)
    pad = pad_mask.bool()
    if pad.shape[-1] != n_k:                                                    # e.g. generate() passes a longer mask
        pad = pad[:, -n_k:] if pad.shape[-1] > n_k else torch.nn.functional.pad(pad, (0, n_k - pad.shape[-1]), value=True)
    vis = causal[None, None] & pad[:, None, None, :]
    q_real = pad[:, -n_q:] if n_q <= n_k else torch.ones(B, n_q, dtype=torch.bool, device=device)
    vis = vis & q_real[:, None, :, None]
    ok = vis.any(-1) | ~q_real[:, None, :]
    if not bool(ok.all()):
        raise ValueError("a real query row has no visible key: left padding is not supported (spec Sec. 6.1)")
    return vis


# --------------------------------------------------------------------------- context
def head_index(h, device):
    """a slice/index tensor for `keep_dense_head`'s int-or-list form; the kept heads keep their given order."""
    import torch as _t
    return slice(h, h + 1) if isinstance(h, int) else _t.as_tensor(list(h), dtype=_t.long, device=device)


@dataclass
class MarSeaContext:
    """Per-forward state shared by the patched layers of one model."""
    pad_mask: Optional[torch.Tensor] = None
    collect: bool = False                          # keep detached Diagnostics per layer
    keep_dense: bool = False                       # keep A / Atil etc. in the diagnostics (memory!)
    mode: str = "dense"                            # "dense" | "chunked"
    chunk: int = 1024
    phase_a: bool = False                          # Phase A: SoftmaxNorm in every patched layer
    force_empty: bool = False                      # MarSea with E forced empty (Phase-B sanity 5(a), T0)
    generation: bool = False                       # frozen-prefix decode caches active
    K_ret: int = 64                                # the DECODE-time bound (Sec. 6.4); the training-time solve is
                                                   # exact unless the normaliser sets its own K_ret
    block_size: int = 512
    policy: str = "frozen_prefix"
    overrides: dict = field(default_factory=dict)  # tau_j_override etc. for tests
    nu: dict = field(default_factory=dict)         # layer_idx -> nu tensor (graph-connected)
    diags: dict = field(default_factory=dict)      # layer_idx -> Diagnostics (detached)
    decode_caches: dict = field(default_factory=dict)
    patched_layers: list = field(default_factory=list)
    n_forward: int = 0
    last_S: dict = field(default_factory=dict)     # layer -> (S32, vis, K_kv, Q) when capture_inputs
    keep_dense_layers: set = field(default_factory=set)   # layers whose dense A / Atil / E are kept (eval at l*)
    keep_dense_head: Optional[int] = None                # F-8: keep the dense tensors of THIS Q-head only ([1,1,T,T]
                                                         # instead of [1,H,T,T]): 2.4 GB rather than 29 GB at T = 8K.
                                                         # May also be a {layer: head} dict: D-9a scores the
                                                         # coreference column at the INDUCTION head and the value
                                                         # side at the detector's (l*, h*), so a pass can need two.
    keep_dense_fields: Optional[tuple] = None            # F-8: keep only these Diagnostics fields (B5 needs E, supp_rel)
    capture_inputs: bool = False
    triggered: bool = False                              # the TRIGGERED teacher-forced form (marsea/triggered.py): a full pass
    n_prefill: Optional[int] = None                      #   computes what prefill (rows < n_prefill) + decoding computes
    baseline_head_block: Optional[int] = None            # EVALUATION of B0/B1/B2/B4: that many Q-heads at a time on the
                                                         # dense path (see _baseline_by_head_block)

    def reset_forward(self, pad_mask):
        self.pad_mask = pad_mask
        self.nu = {}
        self.diags = {}
        self.last_S = {}
        self.n_forward += 1

    def dense_head_for(self, layer_idx: int):
        """None (every head), an int (one head), or a LIST of heads.

        A list is needed when two measurement sites fall in the SAME layer -- D-9a's coreference head and the
        detector's (l*, h*) are independent, and nothing stops the sweep from licensing a head of l*'s own layer.
        Keyed by layer alone, the second site silently overwrote the first and the column was measured at h* while
        being labelled h' (review 30372ae B-1)."""
        h = self.keep_dense_head
        return h.get(layer_idx) if isinstance(h, dict) else h

    def nu_prev_for(self, layer_idx: int):
        prev = [l for l in self.patched_layers if l < layer_idx]
        if not prev:
            return None
        return self.nu.get(prev[-1])

    def clear_generation(self):
        self.decode_caches = {}
        self.generation = False


# --------------------------------------------------------------------------- the patched attention module
class MarSeaAttention(nn.Module):
    def __init__(self, orig: nn.Module, layer_idx: int, normalizer: nn.Module, ctx: MarSeaContext, family: str):
        super().__init__()
        self.orig_class = type(orig)
        self.family = family
        self.config = orig.config
        self.layer_idx = layer_idx
        self.head_dim = orig.head_dim
        self.num_key_value_groups = orig.num_key_value_groups
        self.scaling = orig.scaling
        self.attention_dropout = orig.attention_dropout
        self.is_causal = True
        self.layer_type = getattr(orig, "layer_type", None)
        self.sliding_window = getattr(orig, "sliding_window", None)
        assert self.sliding_window is None, "patched layers must be full-attention layers"
        # the SAME nn.Linear objects (LoRA targets by unchanged names)
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = orig.q_proj, orig.k_proj, orig.v_proj, orig.o_proj
        self.normalizer = normalizer
        self.softmax_norm = SoftmaxNorm()
        self.ctx = ctx
        self._rope = _modeling_module(family).apply_rotary_pos_emb

    # ---- Qwen2Attention.forward with the attention call replaced
    def forward(self, hidden_states: torch.Tensor, position_embeddings, attention_mask=None,
                past_key_values=None, cache_position=None, **kwargs):
        if past_key_values is None and kwargs.get("past_key_value") is not None:
            past_key_values = kwargs.pop("past_key_value")                 # transformers 4.53-4.55 spelling (review C)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = self._rope(q, k, cos, sin)                                        # keys are POST-RoPE from here
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        attn_output = self.marsea_eager_forward(q, k, v)                         # [B, n_q, H*d]
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), None

    @torch.no_grad()
    def _baseline_by_head_block(self, norm, q, k_rep, v_rep, vis, hb: int, kept: bool):
        """The dense baselines (B0, B1, B2, B4), `hb` Q-heads at a time -- evaluation only (no grad).

        Their normalisers act on each head alone (row-wise for B0/B1/B4; B2's marginals, Sinkhorn and MESH step are
        per head), so the blocks give the numbers the full call gives -- except B2's 1e-6 cost noise, which is drawn
        per block.  The full call holds several [B, H, T, T] fp32 tensors at once: at 16K one is 12 GB, B4's stable
        sort alone adds 12 + 24 GB, and B4_s0_E3pad died at 138 GB on a card it had to itself (2026-09-20); B2's MESH
        step, which differentiates through five Sinkhorn iterations, is far beyond that.  Nothing of size [B, H, T, T]
        is built here: each block's A is mixed into the output at once, and only the heads the evaluator asked for
        (ctx.dense_head_for) are kept, in the order head_index would give them."""
        ctx = self.ctx
        H = q.shape[1]
        h = ctx.dense_head_for(self.layer_idx) if kept else None
        want = [] if h is None else ([int(h)] if isinstance(h, int) else [int(x) for x in h])
        outs, got, extras = [], {"A": {}, "A_sm": {}}, {}
        for h0 in range(0, H, hb):
            h1 = min(H, h0 + hb)
            v_blk = vis if vis.shape[1] == 1 else vis[:, h0:h1]
            S = (torch.matmul(q[:, h0:h1], k_rep[:, h0:h1].transpose(2, 3)) * self.scaling).masked_fill(~v_blk, float("-inf"))
            A, d = norm.normalize(S, v_blk, k_rep[:, h0:h1], q[:, h0:h1], State(nu_prev=None))
            outs.append(torch.matmul(A.to(q.dtype), v_rep[:, h0:h1]))
            if ctx.collect:
                for hg in want:
                    if h0 <= hg < h1:
                        for name in ("A", "A_sm"):
                            t = getattr(d, name, None)
                            if t is not None:
                                same = name == "A_sm" and d.A_sm is d.A and hg in got["A"]
                                got[name][hg] = got["A"][hg] if same else t[:, hg - h0:hg - h0 + 1].detach().clone()
                for kx, tx in d.extra.items():
                    extras.setdefault(kx, []).append(tx.detach() if torch.is_tensor(tx) else tx)
            del S, A, d
        diag = Diagnostics(extra={kx: (torch.cat(v, 1) if torch.is_tensor(v[0]) else v[0]) for kx, v in extras.items()})
        for name in ("A", "A_sm"):
            if got[name]:
                setattr(diag, name, torch.cat([got[name][hg] for hg in want], 1))
        diag.extra["baseline_head_block"] = hb
        return torch.cat(outs, 1), diag

    @torch.no_grad()
    def _fp32_scores(self, q, k, vis):
        """fp32 scores for the fidelity statistics.  With ctx.keep_dense_head set, only that Q-head is built (F-8)."""
        h = self.ctx.dense_head_for(self.layer_idx)
        k_rep = repeat_kv(k, self.num_key_value_groups)
        if h is None:
            S = torch.matmul(q.float(), k_rep.float().transpose(2, 3)) * self.scaling
        else:
            hi = head_index(h, q.device)
            S = torch.matmul(q[:, hi].float(), k_rep[:, hi].float().transpose(2, 3)) * self.scaling
        return S.masked_fill(~vis, float("-inf"))

    def marsea_eager_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """eager_attention_forward with THE LINE replaced.  Returns [B, n_q, H, d] (pre-reshape)."""
        ctx = self.ctx
        B, H, n_q, d = q.shape
        n_k = k.shape[-2]
        g = self.num_key_value_groups
        vis = build_vis(ctx.pad_mask, B, n_q, n_k, q.device)                     # [B,1,n_q,n_k]
        # ---- generation: frozen-prefix decode step on cached keys (Sec. 6.4)
        if ctx.generation and n_q == 1 and self.layer_idx in ctx.decode_caches and not ctx.phase_a and not ctx.force_empty \
                and hasattr(self.normalizer, "tauK"):
            from .causal import decode_step
            k_rep = repeat_kv(k, g)
            S = (torch.matmul(q, k_rep.transpose(2, 3)) * self.scaling)
            with torch.autocast(device_type=q.device.type, enabled=False):
                nu_prev = ctx.nu_prev_for(self.layer_idx)
                nu_new = nu_prev[..., -1:] if nu_prev is not None else None
                A_row, info = decode_step(self.normalizer, ctx.decode_caches[self.layer_idx], S, vis, k, q, nu_new)
                ctx.nu[self.layer_idx] = ctx.decode_caches[self.layer_idx].nu
            if ctx.collect:
                ctx.diags[self.layer_idx] = Diagnostics(E=info["E"], cbar_i=info["cbar_i"], tau_i=info["tau_i"], theta=info["theta"],
                                                        Rtil=info["Rtil"], supp_rel=info["supp_rel"], cap_binds=info["cap_binds"],
                                                        extra={"eviction_events": info["eviction_events"],
                                                               "near_truncation_events": info["near_truncation_events"]})
            out = torch.matmul(A_row.to(q.dtype), repeat_kv(v, g))
            return out.transpose(1, 2)
        # ---- chunked path (Sec. 6.6)
        trig = bool(ctx.triggered and ctx.n_prefill and not ctx.generation and n_q == n_k and 0 < int(ctx.n_prefill) < n_q
                    and not ctx.phase_a and not ctx.force_empty and hasattr(self.normalizer, "tauK"))
        if trig and ctx.mode == "chunked":
            raise NotImplementedError("the triggered form runs on the dense path only so far: --mode dense (4K fits an H200)")
        if ctx.mode == "chunked" and not ctx.phase_a and hasattr(self.normalizer, "tauK"):
            from .chunked import marsea_chunked_attention
            state = State(nu_prev=ctx.nu_prev_for(self.layer_idx))
            keep = ctx.keep_dense or self.layer_idx in ctx.keep_dense_layers
            O, diag = marsea_chunked_attention(self.normalizer, q, k, v, vis, state, chunk=ctx.chunk, scaling=self.scaling,
                                               force_empty=ctx.force_empty, dense_outputs=keep,
                                               dense_head=ctx.dense_head_for(self.layer_idx),
                                               want_e8=bool(ctx.collect and not ctx.force_empty))
            ctx.nu[self.layer_idx] = state.nu_next
            if ctx.generation and n_q > 1 and not ctx.force_empty and "sparse" in diag.extra:
                # PREFILL: seal the decode state here too.  Only the dense path used to do this, and the chunked
                # branch returns before reaching it -- so under --mode chunked (what run_evalsuite passes for E3)
                # no cache was ever seeded and every decode step recomputed tau_j and cbar_j live, which is not
                # D-28's mechanism (review d5bd980 B-3).
                from .causal import FrozenPrefixCache
                cache = FrozenPrefixCache(K_ret=ctx.K_ret, block_size=ctx.block_size, policy=ctx.policy)
                cache.init_from_sparse(diag, B, H, n_q, n_k, q.device)
                if ctx.policy == "seal_at_boundary":
                    cache.Q_hist = q.detach().float().clone()
                ctx.decode_caches[self.layer_idx] = cache
            if ctx.collect:
                dd = diag.detach()
                if keep:                                                   # the fidelity statistics need fp32 SCORES at l* (A4/B12)
                    dd.extra["S"] = self._fp32_scores(q, k, vis)             # one head when ctx.keep_dense_head is set
                if dd.extra.get("head_sub") is not None and "sparse" in dd.extra:
                    # the decode cache (above) was seeded from the FULL store; the collected copy follows the kept
                    # heads, or several GB per patched layer at 16K would be retained for nothing (review e982f83 C-4)
                    from .chunked import narrow_sparse
                    dd.extra["sparse"] = narrow_sparse(dd.extra["sparse"], dd.extra["head_sub"], H)
                    dd.extra.pop("decode_seed", None)
                # the summary is built INSIDE the chunked call, on the full-H sparse store and before any dense
                # rebuild narrows the per-head tensors (review 30372ae E)
                if "e8" not in dd.extra:
                    dd.extra["e8_missing"] = f"tau_j={dd.tau_j is not None} sparse={'sparse' in dd.extra}"
                ctx.diags[self.layer_idx] = dd
            return O.to(q.dtype).transpose(1, 2)
        # ---- dense path
        k_rep = repeat_kv(k, g)
        v_rep = repeat_kv(v, g)
        base_norm = self.softmax_norm if (ctx.phase_a or not hasattr(self.normalizer, "normalize")) else self.normalizer
        kept = bool(ctx.collect and (ctx.keep_dense or self.layer_idx in ctx.keep_dense_layers))
        blocked = bool(ctx.baseline_head_block and H > ctx.baseline_head_block and n_q > 1
                       and isinstance(base_norm, (SoftmaxNorm, SoftmaxOneNorm, MESHNorm, RowEntmaxNorm))
                       and not torch.is_grad_enabled() and not self.training and not ctx.capture_inputs
                       and not (kept and ctx.dense_head_for(self.layer_idx) is None))   # every head kept: nothing to save
        S = None
        if blocked:
            out, diag = self._baseline_by_head_block(base_norm, q, k_rep, v_rep, vis, int(ctx.baseline_head_block), kept)
        else:
            S = torch.matmul(q, k_rep.transpose(2, 3)) * self.scaling           # bf16 under autocast
            S = S.masked_fill(~vis, float("-inf"))                               # replaces the additive mask (C-26)
        if ctx.capture_inputs:
            ctx.last_S[self.layer_idx] = (S.detach().float(), vis, k.detach(), q.detach())
        if blocked:
            pass
        elif ctx.phase_a or not hasattr(self.normalizer, "normalize"):
            A, diag = self.softmax_norm.normalize(S, vis)
        else:
            state = State(nu_prev=ctx.nu_prev_for(self.layer_idx))
            kw = dict(ctx.overrides)
            if ctx.force_empty:
                kw["logits_override"] = torch.full((1, 1, 1, 1), -1e4, device=S.device)
            if isinstance(self.normalizer, SoftmaxNorm) or not hasattr(self.normalizer, "tauK"):
                A, diag = self.normalizer.normalize(S, vis, k, q, state)
            else:
                # with head blocking, merging the dense diagnostics costs a second full-size copy of every program
                # tensor; only ask for them when something will read them
                if hasattr(self.normalizer, "want_dense_diag"):
                    # NOT ctx.collect: a logging step keeps only the e8 summary, and the block below nulls the dense
                    # fields anyway -- concatenating seven full-size [1,H,T,T] copies to throw them away every 50th
                    # step is pure waste (review 2026-09-12, fix 4).  E is the exception: the e8 summary IS computed
                    # from it, so it follows ctx.collect (review d5bd980 B-1).
                    self.normalizer.want_dense_diag = bool(ctx.keep_dense or self.layer_idx in ctx.keep_dense_layers)
                    # ... and the GENERATION prefill: FrozenPrefixCache.init_from_prefill reads diag.E to seed the decode
                    # cache.  With head blocking the merged diagnostics carry E only on request, generation does not set
                    # ctx.collect, and every dense-path generation job with --head_block (the whole 8K eval queue) died
                    # on `~E` with E = None (eval smoke test, 2026-09-19).  Without head blocking E is always there,
                    # which is why no test saw it.
                    self.normalizer.want_E = bool(ctx.collect or self.normalizer.want_dense_diag or ctx.generation)
                if trig:                                                          # prompt rows as the prefill, answer rows as decode steps
                    kw["n_prefill"] = int(ctx.n_prefill); kw["decode_K_ret"] = int(ctx.K_ret)
                elif getattr(self.normalizer, "relation_answer_rows_only", False) and n_q == n_k:
                    # the full-sequence form under relation_answer_rows_only: which leading rows are the prompt -- the
                    # generation prefill is all prompt; a teacher-forced pass says through ctx.n_prefill
                    if ctx.generation:
                        kw["prompt_rows"] = n_q
                    elif ctx.n_prefill:
                        kw["prompt_rows"] = int(ctx.n_prefill)
                A, diag = self.normalizer.normalize(S, vis, k, q, state, **kw)
                ctx.nu[self.layer_idx] = state.nu_next
                if ctx.generation:                                                # prefill: seed the decode cache
                    from .causal import FrozenPrefixCache
                    cache = FrozenPrefixCache(K_ret=ctx.K_ret, block_size=ctx.block_size, policy=ctx.policy)
                    cache.init_from_prefill(S.float().masked_fill(~vis, float("-inf")), diag, q if ctx.policy == "seal_at_boundary" else None)
                    ctx.decode_caches[self.layer_idx] = cache
        if ctx.collect:
            dd = diag.detach()
            if dd.E is not None and dd.tau_j is not None:
                from .fidelity import e8_summary_from_diag
                dd.extra["e8"] = e8_summary_from_diag(dd, vis, getattr(self.normalizer, "K_ret", None))
            elif dd.tau_j is not None and "sparse" in dd.extra:
                from .fidelity import e8_summary_from_sparse
                dd.extra["e8"] = e8_summary_from_sparse(dd, vis, getattr(self.normalizer, "K_ret", None))
            elif hasattr(self.normalizer, "tauK") and not ctx.phase_a:
                # E8 is a pre-registered reading (spec Sec. 12.4): if it cannot be computed, say so in the log rather
                # than writing nothing -- silence is what made B-1 survive a whole review round.
                dd.extra["e8_missing"] = f"E={dd.E is not None} tau_j={dd.tau_j is not None} sparse={'sparse' in dd.extra}"
            # F-8: drop what the consumer will not read, THEN narrow what survives to h*.  These two used to be
            # if / elif, so passing keep_dense_fields skipped the head slice entirely and left E as [1,H,T,T]; every
            # reader that indexes [0, 0] -- run_eval's paired_E, i.e. the dense arms' headline A7 precision -- then
            # scored head 0's relation against h*'s attention (review d5bd980 B-5).
            PER_HEAD_FIELDS = ("p", "a1", "Atil", "A_sm", "u", "A", "logits", "E", "theta", "tau_i", "tau_j", "Rtil",
                               "cbar_i", "cbar_j", "kstar", "nu", "psi_j", "c", "cap_binds", "supp_rel")
            if ctx.keep_dense_fields is not None:
                for name in PER_HEAD_FIELDS:
                    if name not in ctx.keep_dense_fields:
                        setattr(dd, name, None)
            h = ctx.dense_head_for(self.layer_idx)
            if h is not None and (ctx.keep_dense or self.layer_idx in ctx.keep_dense_layers):
                                                                       # F-8: the kept heads, re-indexed from 0
                hi = head_index(h, q.device)
                for name in PER_HEAD_FIELDS:
                    t = getattr(dd, name, None)                          # the head-blocked baseline path kept only these heads
                    if torch.is_tensor(t) and t.dim() >= 2 and t.shape[1] == H and not blocked:
                        setattr(dd, name, t[:, hi].contiguous())
                dd.extra["head_sub"] = h
            if not (ctx.keep_dense or self.layer_idx in ctx.keep_dense_layers):
                # E is deliberately NOT dropped: T0, the detector and the S0 sweep all read the relation of layers
                # they never asked dense tensors for.  It costs 0.81 GB per patched layer at 8K, but only on a
                # collecting (~2 %) step, and that is the behaviour the unblocked dense path always had.
                for name in ("p", "a1", "Atil", "A_sm", "u", "A", "logits"):
                    setattr(dd, name, None)
            if ctx.capture_inputs:
                dd.extra["S"] = S.detach().float()
            if self.layer_idx in ctx.keep_dense_layers:
                # interval statistics read SCORES: recompute them in fp32 from q, k -- the bf16 matmul's spacing (0.0625 at
                # |s| in [8, 16)) makes spurious ties, W_j = 0 and hi = inf more often than the fp32 column (review B12)
                dd.extra["S"] = self._fp32_scores(q, k, vis)
            ctx.diags[self.layer_idx] = dd
        if blocked:
            return out.transpose(1, 2)
        A = nn.functional.dropout(A, p=self.attention_dropout if self.training else 0.0, training=self.training)
        out = torch.matmul(A.to(q.dtype), v_rep)                                  # value-mix in the backbone's dtype
        return out.transpose(1, 2)


# --------------------------------------------------------------------------- swap
def _base_model(model):
    return model.get_decoder() if hasattr(model, "get_decoder") else model.model


def patch_model(model, patched_layers, make_normalizer: Callable[[int], nn.Module], ctx: Optional[MarSeaContext] = None,
                checkpoint_patched: bool = False) -> MarSeaContext:
    """Swap self_attn of `patched_layers` for MarSeaAttention; register the pre-hook that feeds the context."""
    fam = family_of(model)
    ctx = ctx or MarSeaContext()
    ctx.patched_layers = sorted(int(l) for l in patched_layers)
    base = _base_model(model)
    for l in ctx.patched_layers:
        layer = base.layers[l]
        old = layer.self_attn
        if isinstance(old, MarSeaAttention):
            old = old  # already patched: replace the normaliser only
            layer.self_attn = MarSeaAttention(old, l, make_normalizer(l), ctx, fam)
        else:
            layer.self_attn = MarSeaAttention(old, l, make_normalizer(l), ctx, fam)
        if checkpoint_patched:
            _checkpoint_layer(layer)

    def pre_hook(module, args, kwargs):
        am = kwargs.get("attention_mask", None)
        if am is None and len(args) > 1:
            am = args[1]
        pad = am if (am is not None and am.dim() == 2) else None
        ctx.reset_forward(pad)
        return None
    if getattr(base, "_marsea_hook", None) is not None:
        base._marsea_hook.remove()
    base._marsea_hook = base.register_forward_pre_hook(pre_hook, with_kwargs=True)
    model.marsea_ctx = ctx
    return ctx


def _checkpoint_layer(layer: nn.Module):
    """Gradient checkpointing of ONE decoder layer (Sec. 6.5 / training Sec. 4), non-reentrant so tensors
    created inside (nu) may be consumed by later layers."""
    import torch.utils.checkpoint as cp
    if getattr(layer, "_marsea_ckpt", False):
        return
    fwd = layer.forward

    def ckpt_forward(*args, **kwargs):
        if torch.is_grad_enabled() and layer.training:
            return cp.checkpoint(fwd, *args, use_reentrant=False, **kwargs)
        return fwd(*args, **kwargs)
    layer.forward = ckpt_forward
    layer._marsea_ckpt = True


def patched_modules(model):
    base = _base_model(model)
    return {l: base.layers[l].self_attn for l in model.marsea_ctx.patched_layers}


def normalizers(model) -> dict:
    return {l: m.normalizer for l, m in patched_modules(model).items()}


def load_backbone(name: str = "Qwen/Qwen2.5-1.5B", dtype=torch.bfloat16, device="cuda", attn_implementation="sdpa"):
    """Sec. 6.1: bf16, SDPA everywhere (unpatched layers keep it).  Re-verifies the Sec. 1.3 facts."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    assert_transformers_contract()
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation=attn_implementation)
    except TypeError:                                                    # transformers < 4.56 spelling (review C)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, attn_implementation=attn_implementation)
    model.to(device)
    tok = AutoTokenizer.from_pretrained(name)
    cfg = model.config
    facts = dict(num_attention_heads=cfg.num_attention_heads, num_key_value_heads=cfg.num_key_value_heads,
                 head_dim=getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads),
                 max_position_embeddings=cfg.max_position_embeddings, num_layers=cfg.num_hidden_layers,
                 dtype=str(dtype), model_type=cfg.model_type)
    if "Qwen2.5-1.5B" in name:
        assert facts["num_attention_heads"] == 12 and facts["num_key_value_heads"] == 2 and facts["head_dim"] == 128, facts
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok, facts
