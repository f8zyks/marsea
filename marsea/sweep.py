"""
S0 as a whole-stack sweep (Sec. 12.3, revised after the first probe).

The retrieval-head detector (Sec. 6.7) selects heads whose queries are answer positions and whose keys are the needle's
VALUE tokens -- construction C-a.  Coreference (a later mention attending to an earlier mention of the same string) is the
INDUCTION pattern, which lives in other, usually earlier, layers, and induction heads attend with a one-position offset
(query = the last token of the later mention, key = the token AFTER the antecedent).  So the routing check must
  * sweep every (layer, head), not the patched layers only;
  * match on SPANS: key span = the first mention plus one following token; query = any token of the later mention;
    the ratio is the mass on the key span against the row maximum;
  * exclude the attention sink (position 0) from the row maximum -- it is not a candidate key.
One forward pass per example with the attention captured layer by layer (eager attention, hooks), like the detector.

Output per construction and (layer, head): fraction of columns whose ratio > 0.5, mean ratio.  A construction is usable at a
head iff the fraction >= 0.8.  The coreference "later mention -> first mention" fraction is the head's INDUCTION score.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np
import torch

USABLE_FRAC = 0.8
RATIO_THRESHOLD = 0.5


def constructions(ex) -> dict:
    """{name: [(key_span (a, b), [query positions])]} for one RULER example (token indices in prompt + gold).
    Constructions of the SPLIT T_j: 'coref_mentions' (the key phrase's coreference column) and 'C-a' (the value columns);
    'C-b' (question mention) and 'coref_answers_diag' are diagnostics."""
    out = {}
    spans = ex.key_mention_spans
    if spans:
        a0, b0 = spans[0]
        key_span = (a0, b0 + 1)                                              # + one following token (induction offset)
        # the coreference column (T_j kind 'coref'): the LATER MENTIONS only (m_j = m + 1 on MV-NIAH); this is what E7 scores
        out["coref_mentions"] = [(key_span, list(range(a, b))) for (a, b) in spans[1:]]
        # diagnostic only (not a construction of the split): do answer positions route to the key phrase?
        out["coref_answers_diag"] = [(key_span, rows) for rows in ex.answer_value_rows]
    # the value columns (T_j kind 'value'): key = value span, target = the emitted value's rows
    out["C-a"] = [((a, b), rows) for (a, b), rows in zip(ex.value_spans, ex.answer_value_rows)]
    if ex.question_mention is not None and spans:
        # the key phrase's mention inside the question (the last mention before the answer prefix mentions)
        q_spans = [sp for sp in spans if sp[0] == ex.question_mention]
        if q_spans:
            out["C-b"] = [(q_spans[0], rows) for rows in ex.answer_value_rows]
    return out


@torch.no_grad()
def routing_ratios(A: torch.Tensor, key_span, queries, exclude_sink: bool = True) -> Optional[float]:
    """A [H, T, T] one layer's attention.  For each query t: mass on the key span / max over candidate keys (position 0
    excluded, keys <= t).  Returns the best query's ratio per head as a [H] array (span-tolerant: any query token)."""
    a, b = key_span
    T = A.shape[-1]
    best = None
    for t in queries:
        if t >= T or b > t + 1:
            continue
        row = A[:, t, : t + 1]                                               # visible keys
        cand = row[:, 1:] if (exclude_sink and t >= 1) else row
        mx = cand.amax(-1).clamp_min(1e-30)
        mass = row[:, a:min(b, t + 1)].sum(-1)
        r = mass / mx
        best = r if best is None else torch.maximum(best, r)
    return best


@dataclass
class SweepResult:
    n_examples: int
    L_layers: int
    H_heads: int
    frac: dict            # construction -> [L][H] fraction of EXAMPLES whose mean column ratio > 0.5 (spec: >= 80 % of examples)
    mean: dict            # construction -> [L][H] mean ratio over columns
    n_columns: dict       # construction -> count
    best: dict            # construction -> top-5 [(frac, mean, layer, head)]
    usable_heads: dict    # construction -> [(layer, head)] with frac >= USABLE_FRAC
    replication_pair: dict          # construction -> {a: fraction of columns with M(a) > m_j} at (l*, h*), plus m_j sizes
    notes: dict = field(default_factory=dict)
    frac_columns: dict = field(default_factory=dict)   # the per-column fraction (the earlier statistic, kept for comparison)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=1)

    def induction_scores(self):
        """the coreference later-mention routing fraction per (layer, head): the induction score."""
        return self.frac.get("coref_mentions")


@torch.no_grad()
def replication_M(S_col: torch.Tensor, j: int, a_levels=(0.5, 1.0)) -> dict:
    """paper tab:compute S0 / Assumption richness: M(a) = number of CANDIDATE queries whose score is within `a` of the
    column's maximum.  S_col [T] fp32 scores of key j over queries (entries for i < j are -inf).

    F-5: query 0 is the attention sink of a BOS-less decoder and is not a candidate (D-31 excludes it from every
    relation), so it is dropped here too -- unless the key IS position 0, when the column has no other content.  Counting
    it inflated M(a) in the PERMISSIVE direction, and M > m_j is what licenses Cor. capacity on real data."""
    vis = torch.isfinite(S_col).clone()
    if j != 0 and vis.numel() > 1:
        vis[0] = False
    if not vis.any():
        return {a: 0 for a in a_levels}
    mx = S_col[vis].max()
    return {a: int(((S_col >= mx - a) & vis).sum()) for a in a_levels}


def _scores_capture(module, l_star: int, h_star: int, scaling: float, store: dict):
    """forward pre-hook on layer l*'s self_attn: recompute the fp32 post-RoPE scores of head h* (queries x keys)."""
    from .relation import repeat_kv
    def pre(mod, args, kwargs):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        pe = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
        if hs is None or pe is None:
            return
        B, T, _ = hs.shape
        q = mod.q_proj(hs).view(B, T, -1, mod.head_dim).transpose(1, 2)
        k = mod.k_proj(hs).view(B, T, -1, mod.head_dim).transpose(1, 2)
        cos, sin = pe
        import importlib
        rope = importlib.import_module(type(mod).__module__).apply_rotary_pos_emb
        q, k = rope(q, k, cos, sin)
        g = q.shape[1] // k.shape[1]
        kh = repeat_kv(k, g)[0, h_star].float(); qh = q[0, h_star].float()
        S = (qh @ kh.T) * scaling                                          # [T, T] fp32
        i = torch.arange(T, device=S.device)[:, None]; jj = torch.arange(T, device=S.device)[None, :]
        store["S"] = S.masked_fill(jj > i, float("-inf"))
    return pre


@torch.no_grad()
def head_sweep(model, examples: list, device="cuda", exclude_sink: bool = True, max_examples: Optional[int] = None,
               l_star: Optional[int] = None, h_star: Optional[int] = None) -> SweepResult:
    """Sweep every (layer, head) of an eager-attention decoder (unpatched or LoRA-adapted) on all constructions.
    With (l_star, h_star) given, also the replication statistic M(a) at that head (A9)."""
    model.eval()
    base = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    if hasattr(base, "get_decoder"):
        base = base.get_decoder()
    layers = base.layers
    L = len(layers); H = model.config.num_attention_heads
    store = {}
    score_store = {}
    pre_handle = None
    if l_star is not None and h_star is not None:
        attn = layers[l_star].self_attn
        pre_handle = attn.register_forward_pre_hook(_scores_capture(attn, l_star, h_star, attn.scaling, score_store), with_kwargs=True)

    def hook(li):
        def f(mod, args, out):
            w = out[1]
            if w is None:
                return
            W = w[0].float()                                                  # [H, T, T]
            for name, pairs in f.cons.items():
                for k, (key_span, queries) in enumerate(pairs):
                    r = routing_ratios(W, key_span, queries, exclude_sink)
                    if r is not None:
                        store.setdefault(name, {}).setdefault(k, {})[li] = r.cpu()
        return f
    handles = []
    for li, layer in enumerate(layers):
        h = hook(li); h.cons = {}
        handles.append((h, layer.self_attn.register_forward_hook(h)))
    acc_frac = {}; acc_mean = {}; n_cols = {}; ex_frac = {}; n_ex = {}
    repl = {}                                                  # construction -> {a: [M(a) > m_j per column]}, m_j sizes
    n_used = 0
    try:
        for ex in examples[:max_examples]:
            cons = constructions(ex)
            for h, _ in handles:
                h.cons = cons
            store.clear(); score_store.clear()
            ids = torch.tensor([ex.prompt_ids + ex.gold_ids], device=device)
            model(input_ids=ids, use_cache=False)
            n_used += 1
            for name, cols in store.items():
                Rs = []
                for k, per_layer in cols.items():
                    R = torch.stack([per_layer[li] for li in range(L)])       # [L, H]
                    Rs.append(R)
                    acc_frac[name] = acc_frac.get(name, 0) + (R > RATIO_THRESHOLD).double()
                    acc_mean[name] = acc_mean.get(name, 0) + R.double()
                    n_cols[name] = n_cols.get(name, 0) + 1
                if "S" in score_store:
                    # replication pair at (l*, h*): per KEY, m_j = number of target queries (pairs sharing the key span,
                    # one per later mention / emitted value -- not the token count of a span); M(a) = queries within a of
                    # the column max; the statistic is the fraction of keys with M(a) > m_j
                    keys = {}
                    for key_span, queries in cons[name]:
                        keys[key_span] = keys.get(key_span, 0) + 1
                    r = repl.setdefault(name, dict(gt=dict(), m=[]))
                    for key_span, m_j in keys.items():
                        M = replication_M(score_store["S"][:, key_span[0]], key_span[0])
                        r["m"].append(m_j)
                        for a, Ma in M.items():
                            r["gt"].setdefault(a, []).append(Ma > m_j)
                # per EXAMPLE (spec Sec. 12.3: ">= 80 % of examples"): the example's mean column ratio
                Rex = torch.stack(Rs).mean(0)
                ex_frac[name] = ex_frac.get(name, 0) + (Rex > RATIO_THRESHOLD).double()
                n_ex[name] = n_ex.get(name, 0) + 1
    finally:
        for _, hd in handles:
            hd.remove()
        if pre_handle is not None:
            pre_handle.remove()
    frac_cols = {n: (acc_frac[n] / n_cols[n]).tolist() for n in acc_frac}
    frac = {n: (ex_frac[n] / n_ex[n]).tolist() for n in ex_frac}
    mean = {n: (acc_mean[n] / n_cols[n]).tolist() for n in acc_mean}
    best = {}; usable = {}
    for n in frac:
        F = np.array(frac[n]); M = np.array(mean[n])
        order = sorted(((F[l, h], M[l, h], l, h) for l in range(L) for h in range(H)), reverse=True)
        best[n] = [(float(a), float(b), int(l), int(h)) for a, b, l, h in order[:5]]
        usable[n] = [(int(l), int(h)) for l in range(L) for h in range(H) if F[l, h] >= USABLE_FRAC]
    pair = {}
    for n, r in repl.items():
        pair[n] = {f"frac_columns_M_gt_m_at_a={a}": float(np.mean(v)) for a, v in r["gt"].items()}
        pair[n]["m_sizes"] = sorted(set(r["m"])); pair[n]["l_star_h_star"] = [l_star, h_star]
    return SweepResult(n_examples=n_used, L_layers=L, H_heads=H, frac=frac, mean=mean, n_columns=n_cols, best=best,
                       usable_heads=usable, replication_pair=pair, frac_columns=frac_cols,
                       notes=dict(exclude_sink=exclude_sink, ratio_threshold=RATIO_THRESHOLD, usable_frac=USABLE_FRAC,
                                  key_span="first mention + 1 following token (coref) / value span (C-a) / question mention (C-b)",
                                  query="any token of the later mention / of the emitted value"))


def print_summary(res: SweepResult):
    print(f"[S0 sweep] {res.n_examples} examples, {res.L_layers} layers x {res.H_heads} heads; sink excluded={res.notes['exclude_sink']}; "
          f"frac = fraction of EXAMPLES with mean ratio > 0.5")
    for n in res.frac:
        print(f"  {n:16s} columns={res.n_columns[n]:4d}  usable heads (frac>=0.8): {res.usable_heads[n][:8]}{'...' if len(res.usable_heads[n]) > 8 else ''}"
              + (f"  replication at (l*,h*): {res.replication_pair[n]}" if n in res.replication_pair else ""))
        for f, m, l, h in res.best[n]:
            print(f"      layer {l:2d} head {h:2d}  frac>0.5 = {f:.2f}  mean ratio = {m:.2f}")
