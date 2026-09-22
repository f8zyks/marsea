"""
Block gold (owner's decision, 2026-09-22): the gold of a retrieval task is a PARTITION -- every key has exactly one owner,
and an owner may hold many keys.  This module measures attention against that partition, for every Q-head of every
patched layer, on one teacher-forced pass per example.

A BLOCK is dict(rows=[answer rows that own it], spans=[(a, b) token spans of the keys it owns], name, emitted_at):
  RULER multi-value NIAH   block v = the rows emitting value v  x  the tokens of value v's needle  (Q > 1: several keys' values)
  RULER variable tracking  block v = the rows emitting variable v  x  v's definition mention; distractor chains are blocks
                           with NO rows (their mentions are keys somebody else owns -- nobody in the answer)
  QA (HotpotQA / MuSiQue)  one block: every answer row  x  the supporting sentences (Hotpot) or paragraphs (MuSiQue);
                           each distractor passage is a row-less block
Row-less blocks carry keys that must NOT be attended; `rest` is every other visible key (the haystack, the question).

Per (layer, head), summed over the answer rows r that belong to a block b (means are taken by the caller):
  rel_own / rel_other / rel_rest      |E_r. ∩ own span| / ∩ other blocks' spans / ∩ the rest      (block precision = own / total)
  rel_recall                          |E_r. ∩ own| / |own|                                        (block recall)
  mass_{own,other,rest}_{sm,final}    softmax and final attention mass on the three regions
  mass_stale_{sm,final}               mass on OTHER blocks whose rows were all emitted before r  (RULER: values already listed)
  mass_remaining_{sm,final}           mass on other blocks not yet emitted (any of them is a valid continuation at a value start)
  col_pure / col_members              for the keys j of own span: members of column j among the GENERATED rows that belong to
                                      b, and all such members  (column purity = pure / members)
  n_rows                              answer rows counted
The generation-side counterpart is list_errors(): repeated / omitted / extra items of the produced list.
"""
from __future__ import annotations
from typing import Optional
import math
import torch

from .evaluate import teacher_forced_pass

BLOCK_KEYS = ("n_rows", "rel_total", "rel_own", "rel_other", "rel_rest", "rel_recall", "own_size",
              "mass_own_sm", "mass_own_final", "mass_other_sm", "mass_other_final", "mass_rest_sm", "mass_rest_final",
              "mass_stale_sm", "mass_stale_final", "mass_remaining_sm", "mass_remaining_final", "col_pure", "col_members")


# --------------------------------------------------------------------------- blocks from the examples
def ruler_blocks(ex) -> list:
    """RULER (multi-value NIAH and variable tracking): one block per gold value / variable; VT distractor chains as row-less
    blocks (their variables' definition mentions)."""
    blocks = []
    for v, (span, rows) in enumerate(zip(ex.value_spans, ex.answer_value_rows)):
        blocks.append(dict(name=f"v{v}", rows=list(rows), spans=[tuple(span)], key=ex.value_owner[v] if v < len(ex.value_owner) else 0))
    meta = getattr(ex, "meta", {}) or {}
    if meta.get("task") == "variable_tracking":
        vm = meta.get("var_mentions", {})
        for c in meta.get("chains", []):
            if not c.get("queried"):
                spans = [tuple(vm[x][0]) for x in c["vars"] if x in vm]              # each distractor variable's definition mention
                if spans:
                    blocks.append(dict(name=f"chain{c['value']}", rows=[], spans=spans, key=-1))
    return blocks


def qa_blocks(ex) -> list:
    """QA: the answer rows own the supporting sentences (HotpotQA) or paragraphs (MuSiQue); each distractor passage is a
    row-less block."""
    gold = set(ex.gold_slots)
    if ex.sentence_spans:
        spans = [tuple(sp) for (_, _, sp) in ex.sentence_spans]
    else:
        spans = [tuple(ex.passage_spans[s]) for s in ex.gold_slots if s < len(ex.passage_spans)]
    blocks = [dict(name="gold", rows=list(ex.answer_rows), spans=spans, key=0)]
    for s, sp in enumerate(ex.passage_spans):
        if s not in gold:
            blocks.append(dict(name=f"passage{s}", rows=[], spans=[tuple(sp)], key=-1))
    return blocks


def blocks_of(ex) -> list:
    return qa_blocks(ex) if hasattr(ex, "passage_spans") else ruler_blocks(ex)


def _mask(spans, n, device):
    m = torch.zeros(n, dtype=torch.bool, device=device)
    for a, b in spans:
        m[max(0, a):min(n, b)] = True
    return m


# --------------------------------------------------------------------------- the pass
@torch.no_grad()
def block_metrics_example(model, ctx, ex, layers, device="cuda", blocks: Optional[list] = None) -> dict:
    """-> {layer: {key: tensor [H] of SUMS}} over the example's owned answer rows (pooled exactly across examples)."""
    blocks = blocks_of(ex) if blocks is None else blocks
    owned = [(r, i) for i, b in enumerate(blocks) for r in b["rows"]]
    if not owned:
        return {}
    H = model.config.num_attention_heads
    sites = [(l, h) for l in layers for h in range(H)]
    _, _, dg = teacher_forced_pass(model, ctx, ex.prompt_ids, ex.gold_ids, layers[0], device, sites=sites, fields=("E", "A_sm", "A"))
    n_p = len(ex.prompt_ids)
    out = {}
    for l in layers:
        d = dg.get(l)
        if d is None or d.A is None:
            continue
        A = d.A[0]; Asm = d.A_sm[0] if d.A_sm is not None else A
        E = d.E[0] if d.E is not None else None                                  # None: a baseline without a relation (B0)
        n = A.shape[-1]; dev = A.device; dt = torch.float64
        rows = torch.tensor([r for r, _ in owned if r < n], device=dev)
        bidx = [i for r, i in owned if r < n]
        if rows.numel() == 0:
            continue
        span_masks = [_mask(b["spans"], n, dev) for b in blocks]
        own = torch.stack([span_masks[i] for i in bidx])                          # [R, n]
        all_spans = torch.stack(span_masks).any(0)                                # every block's keys
        other = all_spans.unsqueeze(0) & ~own
        vis = torch.arange(n, device=dev).unsqueeze(0) <= rows.unsqueeze(1)
        rest = vis & ~all_spans.unsqueeze(0)
        # stale / remaining: OTHER blocks with rows, split by whether all their rows precede r
        last_row = torch.tensor([max(b["rows"]) if b["rows"] else -1 for b in blocks], device=dev)
        has_rows = last_row >= 0
        stale = torch.zeros_like(own); remaining = torch.zeros_like(own)
        for k, (r, i) in enumerate([(int(r), i) for r, i in zip(rows.tolist(), bidx)]):
            for j, b in enumerate(blocks):
                if j == i or not bool(has_rows[j]): continue
                (stale if int(last_row[j]) < r else remaining)[k] |= span_masks[j]
        Ar = A[:, rows].to(dt); Sr = Asm[:, rows].to(dt)                          # [H, R, n]
        f = lambda M: M.to(dt).unsqueeze(0)
        res = dict(n_rows=torch.full((H,), float(rows.numel()), dtype=dt, device=dev),
                   own_size=own.sum(-1).to(dt).sum().expand(H).clone(),
                   mass_own_sm=(Sr * f(own)).sum((-1, -2)), mass_own_final=(Ar * f(own)).sum((-1, -2)),
                   mass_other_sm=(Sr * f(other)).sum((-1, -2)), mass_other_final=(Ar * f(other)).sum((-1, -2)),
                   mass_rest_sm=(Sr * f(rest)).sum((-1, -2)), mass_rest_final=(Ar * f(rest)).sum((-1, -2)),
                   mass_stale_sm=(Sr * f(stale)).sum((-1, -2)), mass_stale_final=(Ar * f(stale)).sum((-1, -2)),
                   mass_remaining_sm=(Sr * f(remaining)).sum((-1, -2)), mass_remaining_final=(Ar * f(remaining)).sum((-1, -2)))
        nan = torch.full((H,), float("nan"), dtype=dt, device=dev)
        if E is not None:
            Er = E[:, rows] & vis.unsqueeze(0)                                     # [H, R, n]
            res.update(rel_total=Er.sum((-1, -2)).to(dt), rel_own=(Er & own.unsqueeze(0)).sum((-1, -2)).to(dt),
                       rel_other=(Er & other.unsqueeze(0)).sum((-1, -2)).to(dt), rel_rest=(Er & rest.unsqueeze(0)).sum((-1, -2)).to(dt))
            res["rel_recall"] = ((Er & own.unsqueeze(0)).sum(-1).to(dt) / own.sum(-1).clamp_min(1).to(dt).unsqueeze(0)).sum(-1)
            # column purity: for the keys of each owned block, the GENERATED rows in its column vs those of the owner
            gen = torch.arange(n, device=dev) >= n_p
            pure = torch.zeros(H, dtype=dt, device=dev); members = torch.zeros(H, dtype=dt, device=dev)
            for i, b in enumerate(blocks):
                if not b["rows"]: continue
                keys = span_masks[i]
                if not bool(keys.any()): continue
                col = E[:, :, keys] & gen.view(1, -1, 1)                           # [H, n_rows_all, |keys|]
                mine = torch.zeros(n, dtype=torch.bool, device=dev); mine[[r for r in b["rows"] if r < n]] = True
                members += col.sum((-1, -2)).to(dt); pure += (col & mine.view(1, -1, 1)).sum((-1, -2)).to(dt)
            res.update(col_pure=pure, col_members=members)
        else:
            res.update(rel_total=nan, rel_own=nan, rel_other=nan, rel_rest=nan, rel_recall=nan, col_pure=nan, col_members=nan)
        out[l] = {k: v.cpu() for k, v in res.items()}
    return out


def block_metrics(model, ctx, examples, layers, device="cuda") -> dict:
    """pooled over examples -> dict(n_rows, layers={layer: {metric: [H]}}) with the RATIOS the docstring names."""
    tot = {}; n_rows = 0
    for ex in examples:
        o = block_metrics_example(model, ctx, ex, layers, device)
        for l, dd in o.items():
            for k, v in dd.items():                                              # NaN (no relation: B0) stays NaN -> None below
                tot.setdefault(l, {}); tot[l][k] = tot[l].get(k, 0) + v
        if o:
            n_rows += int(next(iter(o.values()))["n_rows"][0])
    res = dict(n_rows=n_rows, layers={})
    for l, dd in tot.items():
        n = dd["n_rows"].clamp_min(1)
        r = {}
        def ratio(a, b):
            out = []
            for x, y in zip(a.tolist(), b.tolist()):
                out.append(None if (y == 0 or math.isnan(y) or math.isnan(x)) else x / y)
            return out
        rel_total = dd["rel_total"]
        r["block_precision_rel"] = ratio(dd["rel_own"], rel_total)
        r["rel_other_frac"] = ratio(dd["rel_other"], rel_total); r["rel_rest_frac"] = ratio(dd["rel_rest"], rel_total)
        r["block_recall"] = ratio(dd["rel_recall"], n)
        r["relation_size"] = ratio(rel_total, n)
        r["column_purity"] = ratio(dd["col_pure"], dd["col_members"])
        for reg in ("own", "other", "rest", "stale", "remaining"):
            for st in ("sm", "final"):
                r[f"mass_{reg}_{st}"] = ratio(dd[f"mass_{reg}_{st}"], n)
        fin = dd["mass_own_final"] + dd["mass_other_final"] + dd["mass_rest_final"]
        r["block_precision_mass"] = ratio(dd["mass_own_final"], fin)
        res["layers"][str(l)] = r
    return res


def format_blocks(step, res: dict, top: int = 8) -> str:
    """the heads that put the most softmax mass on their own block, one line each."""
    if not res or not res.get("layers"):
        return f"[blocks] step {step}: no rows"
    f = lambda v, nd=2: "-" if v is None else f"{v:.{nd}f}"
    items = [(l, h, r) for l, r in res["layers"].items() for h in range(len(r["mass_own_sm"]))]
    items.sort(key=lambda t: -(t[2]["mass_own_sm"][t[1]] or 0.0))
    lines = [f"[blocks] step {step}  the {top} heads with most softmax mass on their own block (of {len(items)}); rows {res['n_rows']}"]
    for l, h, r in items[:top]:
        g = lambda k: r[k][h]
        lines.append(f"   L{l} h{h}: mass own {f(g('mass_own_sm'))}>{f(g('mass_own_final'))} other {f(g('mass_other_sm'))}>{f(g('mass_other_final'))} "
                     f"stale {f(g('mass_stale_sm'))}>{f(g('mass_stale_final'))} remaining {f(g('mass_remaining_sm'))}>{f(g('mass_remaining_final'))} rest {f(g('mass_rest_sm'))}>{f(g('mass_rest_final'))} "
                     f"| relation: precision {f(g('block_precision_rel'))} recall {f(g('block_recall'))} other {f(g('rel_other_frac'))} |E_i| {f(g('relation_size'), 0)} col purity {f(g('column_purity'))}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- generation: the produced list against the gold set
def list_errors(generated: str, outputs: list) -> dict:
    """the produced list (first line, split on ', ') against the gold set: repeated (an item listed more than once),
    omitted (gold never listed), extra (listed, not gold), n_listed, and the order-free exact flag."""
    import re
    text = generated.strip().split("\n")[0]
    text = re.split(r"[.]|\bmentioned\b", text)[0]
    items = [p.strip().strip(".,") for p in text.split(",") if p.strip()]
    gold = [str(o) for o in outputs]; gs = set(gold)
    seen = set(); repeated = 0
    for it in items:
        if it in seen: repeated += 1
        seen.add(it)
    omitted = sum(1 for g in gold if g not in seen)
    extra = sum(1 for it in seen if it not in gs)
    return dict(n_listed=len(items), n_gold=len(gold), repeated=repeated, omitted=omitted, extra=extra,
                exact=(seen == gs and repeated == 0), recall=1.0 - omitted / max(1, len(gold)))
