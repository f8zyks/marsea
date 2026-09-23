"""
Sec. 12 evaluation.  Two modes, never mixed:
  teacher-forced  prompt + gold in one forward; read A and the diagnostics at (l*, h*): every attention-level quantity
  generation      greedy decode (frozen-prefix, Sec. 6.4); parse; score: every task-level quantity
Also: the S0 routing check (Sec. 12.3), B5's two-pass evaluation (D-29), the E-suite drivers (Sec. 13), and the
250-step quick evaluation used by the trainer (training Sec. 8.2).  Pre-committed decision rules are printed
beside the numbers they govern (Sec. 13).
"""
from __future__ import annotations
import json, math, pathlib, time
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable
import numpy as np
import torch
import torch.nn.functional as F

from .backbone import MarSeaContext, normalizers
from .normalizer import Diagnostics
from .fidelity import (column_fidelity, dense_column_precision, row_fidelity, passage_support, set_prf,
                       coverage_residual, e8_summary_from_diag)
from .data.ruler import RULERExample, exact_set_accuracy, ruler_recall_score
from .data.qa import QAExample, f1_score, em_score, support_prf, hotpot_joint, parse_answer

DECISION_RULES = {
    "E2": "falsification: interval-hit rate FALLING with m.  The comparator's precision gap SHRINKS with m by arithmetic (1 - m/n); do NOT read that as failure.",
    "E3": "LOAD-BEARING.  Flat precision on delta_j > 0 columns beside a falling fraction of such columns = the theory's prediction; flat precision on ALL columns = the relation is doing the selection, not the program.  Both reportable.",
    "E4": "m = 1: prediction gap ~ 0; a LARGE gap is the interesting negative.",
    "E5": "corroborates E2/E3, never replaces them (m confounded with difficulty).",
    "E6": "accuracy vs conflicting-tier count as a SLOPE; a flat gap across depth withdraws the ranked framing.",
    "E7": "hit rate at chance with unbiased miss signs => the head is not using the column.  Recall twice; precision not bracketed.",
    "S0": "a T_j construction is usable iff the routing fraction > 0.5 on >= 80 % of examples; replication pair M > m_j must exist.",
    "E8": "reported, never acted on: coverage < 0.5 % on every patched layer = 'became B0'; |E_.j| = 1 mass > 80 % = 'trained out of Thm. 1'; across-key var(tau_j) -> 0 = 'constant temperature'; k* near K_ret = hierarchy truncating.",
    "E9": "each ablation is a separate arm under the same recipe; the field-argument ablations (B3, no-nu, key-alone relation) are contribution (i)'s load-bearing test; a match on E7's hit rate says the field argument is decoration.",
}


# --------------------------------------------------------------------------- teacher-forced
@torch.no_grad()
def teacher_forced_pass(model, ctx: MarSeaContext, prompt_ids: list, gold_ids: list, l_star: int, device="cuda",
                        h_star: Optional[int] = None, fields: Optional[tuple] = None,
                        sites=None):
    """one forward on prompt + gold with the dense diagnostics of layer l* kept.  Returns (loss_sum, n_tok, diags).

    h_star: keep only that Q-head's dense tensors (F-8: 2.4 GB rather than 29 GB per example at T = 8K); the
    attention_level_* readers pick the kept head up through `extra["head_sub"]`.
    fields: keep only these Diagnostics fields (B5's pass 1 reads E and supp_rel and nothing else).
    sites: an iterable of (layer, head) PAIRS when the pass needs more than one -- D-9a scores the COREFERENCE column
    at the induction head the S0 sweep found and the value side at the detector's (l*, h*), and one (layer, head)
    cannot carry both.  Pairs, not a {layer: head} map: the two sites can share a LAYER, and keyed by layer the second
    silently overwrote the first, so the column was measured at h* and labelled h' (review 30372ae B-1)."""
    ids = torch.tensor([prompt_ids + gold_ids], device=device)
    labels = torch.tensor([[-100] * len(prompt_ids) + gold_ids], device=device)
    smap = site_map(sites if sites is not None else [(l_star, h_star)])
    ctx.collect = True; ctx.keep_dense_layers = set(smap); ctx.generation = False
    ctx.keep_dense_head = {l: (hs[0] if len(hs) == 1 else hs) for l, hs in smap.items()}
    ctx.keep_dense_fields = fields
    ctx.n_prefill = len(prompt_ids)                                         # the triggered form: the prompt is the prefill
    keep = torch.nonzero(labels[0, 1:] != -100).flatten()
    with torch.autocast(device.split(":")[0] if isinstance(device, str) else "cuda", dtype=torch.bfloat16):
        out = model(input_ids=ids, use_cache=False, logits_to_keep=keep)      # only the answer positions' logits (F-8)
    loss = F.cross_entropy(out.logits[0].float(), labels[0, keep + 1], reduction="sum")
    ctx.collect = False; ctx.keep_dense_layers = set(); ctx.keep_dense_head = None; ctx.keep_dense_fields = None
    ctx.n_prefill = None
    return float(loss), len(gold_ids), dict(ctx.diags)


def _deployed_K_ret(model) -> Optional[int]:
    """the K_ret the patched layers actually solve with (None = exact flat solve): the truncation flag keys on it."""
    try:
        from .backbone import normalizers
        for nm in normalizers(model).values():
            k = getattr(nm, "K_ret", None)
            if k is not None:
                return k
    except Exception:
        pass
    return None


def site_map(pairs) -> dict:
    """{layer: [heads]} from (layer, head) pairs (or a {layer: head} map), order preserved, duplicates dropped."""
    items = pairs.items() if isinstance(pairs, dict) else pairs
    out = {}
    for l, h in items:
        hs = out.setdefault(int(l), [])
        if h not in hs:
            hs.append(h)
    return out


def _kept_head(d, h_star: int) -> int:
    """the index of h_star inside the kept diagnostics.  head_sub is the head (or list of heads) the layer kept: 0
    when it kept one, that head's position when it kept several, and h_star itself when it kept them all."""
    sub = d.extra.get("head_sub") if d is not None else None
    if sub is None:
        return h_star
    kept = list(sub) if isinstance(sub, (list, tuple)) else [int(sub)]
    if h_star not in kept:
        # this used to return 0: head 7's data labelled head 5, a silent wrong answer in precisely B-1's class, in the
        # function written to prevent it.  Asking for a head the layer did not keep is a bug upstream (review e982f83 C-3).
        raise KeyError(f"head {h_star} was not kept at this layer (kept {kept})")
    return kept.index(h_star)


class PairedRelation:
    """A7: a dense arm's headline precision is on MarSea's relation E_.j from the SAME example, at every measurement
    site.  Built LAZILY, one example at a time, as the scorer asks for it.

    It used to be a dict built up front and keyed by LAYER: with the two D-9a sites in one layer the second head's
    relation overwrote the first, so value columns were scored against the coreference head's E_.j (review e982f83
    C-2); and at 16K the eager dict was 268 MB of host RAM per example -- ~268 GB for E3's n = 1000 (B-3).  Keyed by
    (layer, head) here; nothing outlives the example it belongs to."""

    def __init__(self, model, ctx, examples, l_star: int, h_star: int, sites, device="cuda"):
        self.model, self.ctx, self.examples = model, ctx, examples
        self.l_star, self.h_star, self.device = l_star, h_star, device
        self.sites = [(int(l), int(h)) for l, h in sites]

    def get(self, k, default=None):
        if k is None or not (0 <= k < len(self.examples)):
            return default
        ex = self.examples[k]
        _, _, dg = teacher_forced_pass(self.model, self.ctx, ex.prompt_ids, ex.gold_ids, self.l_star, self.device,
                                       h_star=self.h_star, fields=("E",), sites=self.sites)
        out = {(l, h): dg[l].E[0, _kept_head(dg[l], h)] for l, h in self.sites if l in dg and dg[l].E is not None}
        missing = [s for s in self.sites if s not in out]
        if missing:
            raise RuntimeError(f"the paired MarSea pass produced no relation at {missing}")
        return out


def _paired_at(E_paired, layer: int, head: int):
    """the paired relation at one (layer, head): a {(layer, head): E} map, or one tensor for a single-site pass."""
    if isinstance(E_paired, dict):
        if (layer, head) not in E_paired:
            raise KeyError(f"no paired relation at (layer, head) = {(layer, head)}; have {sorted(E_paired)}")
        return E_paired[(layer, head)]
    return E_paired


def _scores_at(diag: Diagnostics, h: int):
    """the fp32 score column source for the interval statistics.  Never fall back to attention values (review A4)."""
    S = diag.extra.get("S")
    if S is None:
        raise RuntimeError("fp32 scores of l* were not captured (keep_dense_layers must include l*; the chunked path stores them too)")
    return S[0, h]


@torch.no_grad()
def kind_site_map(coref_site=None, value_site=None, kind_sites=None) -> dict:
    """{T_j kind: (layer, head)} for the kinds whose licence names a site other than (l*, h*).

    Both kinds have a licence and a site: the `value` half had no way through, so a kind licensed only by
    `usable_anywhere` was still scored at (l*, h*) and reported as licensed -- the gate was half-fixed
    (review 30372ae D)."""
    out = dict(kind_sites or {})
    if coref_site is not None:
        out["coref"] = tuple(coref_site)
    if value_site is not None:
        out["value"] = tuple(value_site)
    return {k: tuple(v) for k, v in out.items() if v is not None}


def _column_at_site(diags, d, hh, l_star, h_star, kind, sites):
    """(diagnostics, kept-head index, layer, GLOBAL head) at which a column of this kind is measured."""
    site = sites.get(kind)
    if site is not None and site[0] in diags:
        dj = diags[site[0]]
        return dj, _kept_head(dj, site[1]), int(site[0]), int(site[1])
    return d, hh, int(l_star), int(h_star)


def attention_level_ruler(diags: dict, l_star: int, h_star: int, ex: RULERExample, arm_is_dense: bool,
                          E_paired=None, K_ret: Optional[int] = None,
                          coref_site: Optional[tuple] = None, value_site: Optional[tuple] = None,
                          kind_sites: Optional[dict] = None) -> dict:
    """all Sec. 12.4 quantities at (l*, h*) for one RULER example.  For dense arms, precision on both domains.

    coref_site: (layer, head) at which the COREFERENCE columns are scored.  D-9a splits T_j into a coreference
    column (later mentions of the key phrase) and value columns (the answer rows), and the two live at different
    heads -- coreference is the induction pattern the S0 sweep looks for, retrieval is the detector's (l*, h*).
    Without this the kind was only a TAG: both kinds were measured at (l*, h*) while aggregate_ruler split its
    reporting by kind, so the tables looked split when they were not (review d5bd980 E).  E_paired is a
    {(layer, head): E} map (PairedRelation.get) when a dense arm is scored against MarSea's relation, one entry per
    measurement site -- keyed by the PAIR, because the two sites can share a layer (review e982f83 C-2)."""
    d = diags[l_star]
    hh = _kept_head(d, h_star)                                         # F-8: 0 when only h* was kept
    A = d.A[0, hh] if d.A is not None else d.A_sm[0, hh]
    n = A.shape[-1]
    S = _scores_at(d, hh)
    vis_row = lambda i: torch.arange(n, device=A.device) <= i
    has_stages = d.Atil is not None                                       # MarSea / B3: Stage-1 and Stage-2 readings exist
    if not arm_is_dense and not has_stages:
        # B4 (query-local sparsifier, no relation) and B5 (trace relation, no Stage 1): P/R from the final A; the
        # relation domain is B5's trace E or the paired MarSea E; before == after (review A3)
        E_dom = d.E[0, hh] if d.E is not None else (_paired_at(E_paired, l_star, h_star) if E_paired is not None else None)
        return _stageless_attention_level(A, S, ex, E_dom, vis_row, n)
    rows = []
    # ---- row quantities at the answer rows; K_i = gold value token spans
    K_mask = torch.zeros(n, dtype=torch.bool, device=A.device)
    for (a, b) in ex.value_spans: K_mask[a:b] = True
    K_sent = torch.zeros(n, dtype=torch.bool, device=A.device)
    for (a, b) in ex.sentence_spans: K_sent[a:b] = True
    if not arm_is_dense:
        E = d.E[0, hh]; Atil = d.Atil[0, hh]
        for i in ex.answer_rows:
            if i >= n: continue
            r = row_fidelity(A[i], Atil[i], E[i] & vis_row(i), K_mask & vis_row(i), float(d.theta[0, hh, i]), float(d.tau_i[0, hh, i]),
                             float(d.Rtil[0, hh, i]), float(d.cbar_i[0, hh, i]), bool(d.cap_binds[0, hh, i]))
            r["row"] = i; r["K_level"] = "value"
            rs = row_fidelity(A[i], Atil[i], E[i] & vis_row(i), K_sent & vis_row(i), float(d.theta[0, hh, i]), float(d.tau_i[0, hh, i]),
                              float(d.Rtil[0, hh, i]), float(d.cbar_i[0, hh, i]), bool(d.cap_binds[0, hh, i]))
            r["P_after_sentence"] = rs["P_after"]; r["R_after_sentence"] = rs["R_after"]
            rows.append(r)
    else:
        for i in ex.answer_rows:
            if i >= n: continue
            supp = (A[i] > 0) & vis_row(i)
            inter = int((supp & K_mask).sum()); ns = int(supp.sum())
            rows.append(dict(row=i, P_after=inter / ns if ns else 1.0, R_after=inter / max(1, int((K_mask & vis_row(i)).sum())), K_level="value"))
    # ---- column quantities at T_j (coreference construction, D-9)
    cols = []
    sites = kind_site_map(coref_site, value_site, kind_sites)
    for j, targets in ex.T_j.items():
        if j >= n: continue
        kind = ex.T_j_kind.get(j, "coref")
        dj, hj, lj, h_global = _column_at_site(diags, d, hh, l_star, h_star, kind, sites)
        same = (dj is d) and (hj == hh)                 # same LAYER is not enough: the two sites can share one
        Aj = A if same else (dj.A[0, hj] if dj.A is not None else dj.A_sm[0, hj])
        Sj = S if same else _scores_at(dj, hj)
        Tm = torch.zeros(n, dtype=torch.bool, device=A.device)
        for t in targets:
            if t < n: Tm[t] = True
        vis_col = torch.arange(n, device=A.device) >= j
        if not arm_is_dense:
            E = dj.E[0, hj]
            c = column_fidelity(Sj[:, j] if Sj is not None else Aj[:, j], E[:, j] & vis_col, dj.p[0, hj][:, j], Aj[:, j],
                                float(dj.tau_j[0, hj, j]), Tm, int(dj.kstar[0, hj, j]), float(dj.nu[0, hj, j]),
                                float(dj.psi_j[0, hj, j]) if dj.psi_j is not None else None, K_ret)
        else:
            ep = _paired_at(E_paired, lj, h_global) if E_paired is not None else None
            c = dense_column_precision(Aj[:, j], vis_col, Tm, ep[:, j] if ep is not None else None)
            c["m_total"] = int(Tm.sum())
        # the site is the GLOBAL (layer, head), not the index inside the kept slice: with keep_dense_head set the
        # kept index is always 0, so recording hj here would report head 0 for every arm
        c["col"] = j; c["m_planted"] = len(targets); c["kind"] = kind; c["site"] = [lj, h_global]
        cols.append(c)
    # the residual is per KIND: pooling a coreference column (m_j = m + 1) with m value columns (m_j = 1 each) into
    # one n_q - sum(m_j) gives a number no decision rule can read (review d5bd980 E)
    resid = {k: coverage_residual(n, [len(v) for jj_, v in ex.T_j.items() if ex.T_j_kind.get(jj_, "coref") == k])
             for k in sorted(set(ex.T_j_kind.get(jj_, "coref") for jj_ in ex.T_j))}
    return dict(rows=rows, cols=cols, coverage_residual=resid,
                coverage_residual_pooled=coverage_residual(n, [len(v) for v in ex.T_j.values()]),
                n_q=n, m_j_zero_columns=sum(1 for c in cols if c.get("m", 1) == 0))


@torch.no_grad()
def _stageless_attention_level(A, S, ex, E_dom, vis_row, n) -> dict:
    """rows/cols for an arm with a final A but no Stage-1 tensors (B4, B5).  Domain = E_dom (trace / paired relation) when
    given, else the full visible row/column; before == after."""
    rows = []
    K_mask = torch.zeros(n, dtype=torch.bool, device=A.device)
    for (a, b) in ex.value_spans: K_mask[a:b] = True
    for i in ex.answer_rows:
        if i >= n: continue
        dom = (E_dom[i] if E_dom is not None else torch.ones(n, dtype=torch.bool, device=A.device)) & vis_row(i)
        supp = (A[i] > 0) & dom
        inter = int((supp & K_mask).sum()); ns = int(supp.sum()); nK = int((K_mask & dom).sum())
        P = inter / ns if ns else 1.0; R = inter / nK if nK else float("nan")
        rows.append(dict(row=i, P_before=P, R_before=R, P_after=P, R_after=R, supp_rel=ns, E_size=int(dom.sum()),
                         rel_recall=(int((K_mask & dom).sum()) / max(1, int(K_mask.sum()))) if E_dom is not None else 1.0, K_level="value"))
    cols = []
    for j, targets in ex.T_j.items():
        if j >= n: continue
        Tm = torch.zeros(n, dtype=torch.bool, device=A.device)
        for t in targets:
            if t < n: Tm[t] = True
        vis_col = torch.arange(n, device=A.device) >= j
        c = dense_column_precision(A[:, j], vis_col, Tm, E_dom[:, j] if E_dom is not None else None)
        c["m_total"] = int(Tm.sum()); c["col"] = j; c["m_planted"] = len(targets); c["kind"] = ex.T_j_kind.get(j, "coref")
        cols.append(c)
    resid = {k: coverage_residual(n, [len(v) for jj_, v in ex.T_j.items() if ex.T_j_kind.get(jj_, "coref") == k])
             for k in sorted(set(ex.T_j_kind.get(jj_, "coref") for jj_ in ex.T_j))}
    return dict(rows=rows, cols=cols, coverage_residual=resid,
                coverage_residual_pooled=coverage_residual(n, [len(v) for v in ex.T_j.values()]), n_q=n,
                m_j_zero_columns=0, stageless=True)


@torch.no_grad()
def attention_level_qa(diags: dict, l_star: int, h_star: int, ex: QAExample, arm_is_dense: bool, E_paired=None, K_ret=None,
                       coref_site=None, value_site=None, kind_sites=None) -> dict:
    """D-9a applies to E5 too: the QA T_j carries the same two kinds (a bridge string's coreference column and the
    answer's value column), and without the sites the per-kind E5 table had no provenance and was a different
    measurement from the RULER one without saying so (review 30372ae D)."""
    d = diags[l_star]
    hh = _kept_head(d, h_star)                                         # F-8: 0 when only h* was kept
    A = d.A[0, hh] if d.A is not None else d.A_sm[0, hh]
    n = A.shape[-1]
    S = _scores_at(d, hh)
    vis_row = lambda i: torch.arange(n, device=A.device) <= i
    if not arm_is_dense and d.Atil is None:
        arm_is_dense = True                                               # B4/B5 on QA: score like a dense arm on the paired domain (A3)
        E_paired = ({(int(l_star), int(h_star)): d.E[0, hh]} if d.E is not None else E_paired)
    K_mask = torch.zeros(n, dtype=torch.bool, device=A.device)
    for slot in ex.gold_slots:
        a, b = ex.passage_spans[slot]; K_mask[a:b] = True
    rows = []; sel_rel = []; sel_dense = []
    for i in ex.answer_rows:
        if i >= n: continue
        if not arm_is_dense:
            E = d.E[0, hh]; Atil = d.Atil[0, hh]
            r = row_fidelity(A[i], Atil[i], E[i] & vis_row(i), K_mask & vis_row(i), float(d.theta[0, hh, i]), float(d.tau_i[0, hh, i]),
                             float(d.Rtil[0, hh, i]), float(d.cbar_i[0, hh, i]), bool(d.cap_binds[0, hh, i]))
            sel_rel.append(passage_support(A[i], E[i], ex.passage_spans, on_relation=True))
        else:
            supp = (A[i] > 0) & vis_row(i); inter = int((supp & K_mask).sum()); ns = int(supp.sum())
            r = dict(P_after=inter / ns if ns else 1.0, R_after=inter / max(1, int((K_mask & vis_row(i)).sum())))
        sel_dense.append(passage_support(A[i], torch.ones(n, dtype=torch.bool, device=A.device), ex.passage_spans, on_relation=False))
        r["row"] = i; rows.append(r)
    # passage-level support = union over answer rows of the selected passages
    def union(sel):
        if not sel: return set()
        m = np.array(sel).any(0); return {k for k in range(len(m)) if m[k]}
    gold = set(ex.gold_slots)
    out = dict(rows=rows, support_dense=set_prf(union(sel_dense), gold), selected_slots_dense=union(sel_dense))
    if not arm_is_dense:
        out["support_rel"] = set_prf(union(sel_rel), gold); out["selected_slots_rel"] = union(sel_rel)
    cols = []
    sites = kind_site_map(coref_site, value_site, kind_sites)
    for j, targets in ex.T_j.items():
        if j >= n: continue
        kind = ex.T_j_kind.get(j, "coref")
        dj, hj, lj, h_global = _column_at_site(diags, d, hh, l_star, h_star, kind, sites)
        same = (dj is d) and (hj == hh)
        Aj = A if same else (dj.A[0, hj] if dj.A is not None else dj.A_sm[0, hj])
        Sj = S if same else _scores_at(dj, hj)
        Tm = torch.zeros(n, dtype=torch.bool, device=A.device)
        for t in targets:
            if t < n: Tm[t] = True
        vis_col = torch.arange(n, device=A.device) >= j
        if not arm_is_dense:
            E = dj.E[0, hj]
            c = column_fidelity(Sj[:, j] if Sj is not None else Aj[:, j], E[:, j] & vis_col, dj.p[0, hj][:, j], Aj[:, j],
                                float(dj.tau_j[0, hj, j]), Tm, int(dj.kstar[0, hj, j]), float(dj.nu[0, hj, j]),
                                float(dj.psi_j[0, hj, j]) if dj.psi_j is not None else None, K_ret)
        else:
            ep = _paired_at(E_paired, lj, h_global) if E_paired is not None else None
            c = dense_column_precision(Aj[:, j], vis_col, Tm, ep[:, j] if ep is not None else None)
        c["col"] = j; c["m_planted"] = len(targets); c["kind"] = kind; c["site"] = [lj, h_global]
        cols.append(c)
    out["cols"] = cols
    out["coverage_residual"] = {k: coverage_residual(n, [len(v) for jj_, v in ex.T_j.items()
                                                         if ex.T_j_kind.get(jj_, "coref") == k])
                                for k in sorted(set(ex.T_j_kind.get(jj_, "coref") for jj_ in ex.T_j))}
    return out


# --------------------------------------------------------------------------- generation
@torch.no_grad()
def generate_greedy(model, tok, ctx: MarSeaContext, prompt_ids: list, max_new_tokens: int, device="cuda", frozen_prefix=True) -> str:
    ids = torch.tensor([prompt_ids], device=device)
    ctx.n_prefill = None
    # a causal MESH (B2) decodes through its own cache like MarSea does: the prefill is one block, each new row one solve
    from .baselines import MESHNorm
    from .backbone import normalizers as _nms
    if any(isinstance(nm, MESHNorm) and nm.causal for nm in _nms(model).values()):
        frozen_prefix = True
    ctx.generation = frozen_prefix; ctx.clear_generation(); ctx.generation = frozen_prefix
    t0 = time.time()
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    ctx.clear_generation()
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return text, time.time() - t0


# --------------------------------------------------------------------------- S0 routing check (Sec. 12.3)
# The single-head/single-token probe that used to live here is RETIRED: marsea/sweep.py runs the whole-stack sweep
# (all layers x heads, span-tolerant, sink excluded, per-example fractions), which is what spec Sec. 12.3 asks for.
# It survived here as dead code whose docstring still documented the abolished merged column (m_j = 2m + 1), which
# is exactly the sort of thing that gets copied into a paper (review d5bd980 E).


# --------------------------------------------------------------------------- B5 (two-pass, teacher-forced only)
@torch.no_grad()
def b5_two_pass(model_marsea, ctx_m: MarSeaContext, model_b0, ctx_0: MarSeaContext, ex, l_star: int, h_star: int,
                attention_fn: Callable, device="cuda") -> dict:
    """pass 1: the trained MarSea arm records E and n_i = |supp(A[i, E_i.])| per (layer, head, row); pass 2: the
    trained B0 weights with MatchedSparsityNorm fed that trace on the SAME example."""
    from .baselines import MatchedSparsityNorm
    # F-8: pass 1 reads the relation and the support size and nothing else -- keeping every dense tensor of all four
    # patched layers was ~4 x 29 GB at T = 8K.  keep_dense is still on (B5's trace is per LAYER, not just l*).
    ctx_m.keep_dense = True
    _, _, diags = teacher_forced_pass(model_marsea, ctx_m, ex.prompt_ids, ex.gold_ids, l_star, device,
                                      fields=("E", "supp_rel"))
    ctx_m.keep_dense = False
    traces = {l: (d.E, d.supp_rel) for l, d in diags.items() if d.E is not None}
    for l, nm in normalizers(model_b0).items():
        assert isinstance(nm, MatchedSparsityNorm), "B5 pass 2 needs MatchedSparsityNorm in every patched layer"
        if l in traces:
            nm.trace = traces[l]
    # pass 2 scores at (l*, h*) only; the normaliser still receives the FULL trace (it re-shapes every head)
    _, _, diags0 = teacher_forced_pass(model_b0, ctx_0, ex.prompt_ids, ex.gold_ids, l_star, device, h_star=h_star)
    # B5's outputs: attention-level only, on MarSea's relation (paired domain).  The diagnostic E must sit in the same
    # head space as pass 2's kept tensors (F-8): slice it to h* when the layer kept a single head.
    d0 = diags0[l_star]
    E_tr = traces[l_star][0]
    d0.E = E_tr[:, h_star:h_star + 1] if d0.extra.get("head_sub") is not None else E_tr
    return attention_fn(diags0, l_star, h_star, ex, arm_is_dense=False)


def _paired_on(paired_E, k: int, device):
    """example k's paired relation on the scoring device (a PairedRelation, or a plain {k: ...} map in the tests)."""
    if paired_E is None:
        return None
    pe = paired_E.get(k)
    dev = device.split(":")[0] if isinstance(device, str) else device.type
    if isinstance(pe, dict):
        return {key: (t.to(device) if t.device.type != dev else t) for key, t in pe.items()}
    if pe is not None and pe.device.type != dev:
        pe = pe.to(device)
    return pe


# --------------------------------------------------------------------------- RULER set evaluation (E2/E3/E4/E7/E8)
@torch.no_grad()
def evaluate_ruler(model, tok, ctx: MarSeaContext, examples: list[RULERExample], l_star: int, h_star: int, arm: str,
                   modes=("teacher", "generation"), device="cuda", max_new_tokens=32, paired_E: Optional[dict] = None,
                   column_measurable=True, coref_site: Optional[tuple] = None,
                   kind_sites: Optional[dict] = None) -> list[dict]:
    """column_measurable: True, False, or {'coref': bool, 'value': bool} (S0's licence per T_j kind, spec Sec. 12.3).  Columns
    of a kind that S0 did not license are dropped and the record says so, so the tables read 'not measurable on this
    backbone' for that kind instead of carrying numbers."""
    ctx.relation_direction = "row"                                               # NIAH / VT: value tokens are the one-end

    dense = arm.lower() in ("b0", "b1", "b2")
    rows = []
    for k, ex in enumerate(examples):
        rec = dict(example=k, index=ex.meta.get("index"), m=len(ex.outputs), n=ex.meta.get("num_needle_k"),
                   depth=ex.meta.get("gold_depth"), length=ex.meta.get("length"), column_measurable=column_measurable)
        if "teacher" in modes:
            # D-9a: the coreference column is scored where coreference LIVES (the induction head the S0 sweep found),
            # the value columns and the rows at the detector's (l*, h*).  One site cannot carry both.
            ks = kind_site_map(coref_site, None, kind_sites)
            sites = [(l_star, h_star)] + [(int(l), int(h)) for l, h in ks.values()]
            # the paired relation FIRST: its pass's diagnostics are dropped before the arm's own pass keeps its dense ones
            pe = _paired_on(paired_E, k, device)
            loss, ntok, diags = teacher_forced_pass(model, ctx, ex.prompt_ids, ex.gold_ids, l_star, device,
                                                    h_star=h_star, sites=sites)
            rec["tf_loss"] = loss / max(1, ntok)
            rec["attention"] = attention_level_ruler(diags, l_star, h_star, ex, dense, pe, _deployed_K_ret(model),
                                                     kind_sites=ks)
            rec["sites"] = [[int(l), int(h)] for l, h in sites]
            allowed = column_measurable if isinstance(column_measurable, dict) else {"coref": bool(column_measurable), "value": bool(column_measurable)}
            dropped = [c["kind"] for c in rec["attention"]["cols"] if not allowed.get(c.get("kind", "coref"), True)]
            rec["attention"]["cols"] = [c for c in rec["attention"]["cols"] if allowed.get(c.get("kind", "coref"), True)]
            if dropped:
                rec["attention"]["cols_not_measurable"] = sorted(set(dropped))
            rec["e8"] = {l: d.extra.get("e8") for l, d in diags.items()}
        if "generation" in modes and arm.lower() != "b5":
            # max_new_tokens = None: the LIST's own budget, 9 tokens a value + 16 -- a fixed 160 cut a 32-value answer at 17
            # values and read as 15 omissions (the B0 gate, 2026-09-22)
            budget = max_new_tokens if max_new_tokens is not None else 9 * len(ex.outputs) + 16
            text, dt = generate_greedy(model, tok, ctx, ex.prompt_ids, budget, device, frozen_prefix=(not dense))
            rec["generated"] = text; rec["exact_set"] = exact_set_accuracy(text, ex.outputs)
            rec["ruler_recall"] = ruler_recall_score(text, ex.outputs); rec["decode_seconds"] = dt
            from .blockgold import list_errors
            rec["list_errors"] = list_errors(text, ex.outputs)                       # repeated / omitted / extra items (2026-09-22)
            rec["task"] = (ex.meta or {}).get("task", "niah"); rec["Q"] = len(getattr(ex, "query_keys", []) or [1])
        rows.append(rec)
    return rows


@torch.no_grad()
def evaluate_qa(model, tok, ctx: MarSeaContext, examples: list[QAExample], l_star: int, h_star: int, arm: str,
                modes=("teacher", "generation"), device="cuda", max_new_tokens=16, hotpot: bool = False,
                kind_sites: Optional[dict] = None, paired_E=None) -> list[dict]:
    """paired_E: as for evaluate_ruler.  E5's dense arms were scored on the full visible column while RULER's were
    scored on the paired relation -- two measurements of the quantity D-9a's provenance work exists for, under one
    table (review e982f83 G)."""
    ctx.relation_direction = "column"                                            # QA: answer rows are the one-end

    dense = arm.lower() in ("b0", "b1", "b2")
    sites = kind_site_map(kind_sites=kind_sites)
    rows = []
    for k, ex in enumerate(examples):
        rec = dict(example=k, id=ex.meta.get("id"), m=ex.m, hops=ex.meta.get("hops"))
        if "teacher" in modes:
            pairs = [(l_star, h_star)] + [(int(l), int(h)) for l, h in sites.values()]
            pe = _paired_on(paired_E, k, device)
            loss, ntok, diags = teacher_forced_pass(model, ctx, ex.prompt_ids, ex.gold_ids, l_star, device,
                                                    h_star=h_star, sites=pairs)
            rec["tf_loss"] = loss / max(1, ntok)
            rec["attention"] = attention_level_qa(diags, l_star, h_star, ex, dense, pe, _deployed_K_ret(model),
                                                  kind_sites=sites)
            rec["sites"] = [[int(l), int(h)] for l, h in pairs]
        if "generation" in modes and arm.lower() != "b5":
            text, dt = generate_greedy(model, tok, ctx, ex.prompt_ids, max_new_tokens, device, frozen_prefix=(not dense))
            pred = parse_answer(text)
            rec["generated"] = pred; rec["ans_em"] = em_score(pred, ex.answer); rec["ans_f1"] = f1_score(pred, ex.answer)[0]
            if "attention" in rec:
                sup = rec["attention"].get("support_rel", rec["attention"]["support_dense"])
                rec["support_P"], rec["support_R"], rec["support_F1"], rec["support_EM"] = sup
                if hotpot:
                    # F-4: both sides must be in the SAME index space -- prompt-order passage slots.  The prediction is the
                    # union over the answer rows that attention_level_qa computed; the gold is the gold slots themselves.
                    # (The earlier form passed range(len(gold_slots)) as gold and collapsed the prediction to [] unless the
                    # support EM was already 1, so the joint cell read 0 for every arm.)
                    pred_slots = rec["attention"].get("selected_slots_rel", rec["attention"].get("selected_slots_dense", set()))
                    rec["joint"] = hotpot_joint(pred, ex.answer, pred_slots, set(ex.gold_slots))
            rec["decode_seconds"] = dt
        rows.append(rec)
    return rows


# --------------------------------------------------------------------------- aggregation
def aggregate_ruler(rows: list[dict], stratify: str = "m") -> dict:
    """E2/E3 tables: per stratum -- exact-set accuracy, RULER recall, MarSea precision (rows), interval-hit rate,
    fraction of active columns with delta_j > 0, precision on those vs all, relation recall, P/R before/after."""
    out = {}
    for r in rows:
        key = r.get(stratify)
        g = out.setdefault(key, dict(n=0, exact_set=[], recall=[], P_row=[], R_row=[], P_row_before=[], R_row_before=[], hit=[], delta_pos=[],
                                    P_col_all=[], P_col_dpos=[], R_col_before=[], R_col_after=[], P_col_before=[], rel_recall_col=[], rel_recall_row=[],
                                    excl_stage1=[], rej_row=[], surviving=[], P_full=[], P_rel=[], excluded={}, m0=0, n_cols=0,
                                    delta_vs_recall=[], residual=[], measurable=[]))
        g["n"] += 1
        if "exact_set" in r: g["exact_set"].append(float(r["exact_set"])); g["recall"].append(r["ruler_recall"])
        # E5's task metrics are written per row and used to be dropped here: aggregate_ruler read only exact_set /
        # ruler_recall plus the attention block, so the E5 table reported exact_set_acc = nan and nothing else
        # (review d5bd980 F).
        for k_ in ("ans_em", "ans_f1", "support_P", "support_R", "support_F1", "support_EM", "joint"):
            v_ = r.get(k_)
            if v_ is None:
                continue
            if isinstance(v_, dict):
                # HotpotQA's `joint` is hotpot_joint()'s DICT (joint_em / joint_f1 / joint_p / joint_r beside the answer
                # and support parts already aggregated above); float() of it killed every E5 hotpot table -- MuSiQue
                # rows carry no `joint`, which is why only hotpot failed (eval smoke test, 2026-09-19)
                for kk, vv in v_.items():
                    if kk.startswith("joint") and vv is not None:
                        g.setdefault("qa", {}).setdefault(kk, []).append(float(vv))
            else:
                g.setdefault("qa", {}).setdefault(k_, []).append(float(v_))
        att = r.get("attention")
        if att:
            g["measurable"].append(not att.get("cols_not_measurable"))
            g["m0"] += att.get("m_j_zero_columns", 0)
            rres = att.get("coverage_residual")
            if isinstance(rres, dict):          # per T_j kind (D-9a): pooling m + 1 with m ones is uninterpretable
                for kk, vv in rres.items():
                    g.setdefault("residual_by_kind", {}).setdefault(kk, []).append(vv)
                g["residual"].append(att.get("coverage_residual_pooled"))
            else:
                g["residual"].append(rres)
            for rw in att["rows"]:
                g["P_row"].append(rw.get("P_after")); g["R_row"].append(rw.get("R_after"))
                g["P_row_before"].append(rw.get("P_before")); g["R_row_before"].append(rw.get("R_before"))
                if "rel_recall" in rw: g["rel_recall_row"].append(rw["rel_recall"])
                for k_, dst in (("excluded_by_stage1", "excl_stage1"), ("rejected_by_row", "rej_row"), ("surviving", "surviving")):
                    if k_ in rw and rw[k_] == rw[k_]: g[dst].append(rw[k_])
            for c in att["cols"]:
                if "hit" in c:
                    g["n_cols"] += 1
                    if c.get("m", 0) == 0: continue                      # m_j = 0 columns excluded from column averages (counted)
                    g.setdefault("m_by_kind", {}).setdefault(c.get("kind", "coref"), []).append(c.get("m_total", c.get("m")))
                    if c.get("site") is not None:
                        g.setdefault("site_by_kind", {}).setdefault(c.get("kind", "coref"), set()).add(tuple(c["site"]))
                    if c["hit"] is not None:
                        g["hit"].append(float(c["hit"])); g.setdefault("hit_by_kind", {}).setdefault(c.get("kind", "coref"), []).append(float(c["hit"]))
                    elif c.get("hit_excluded"): g["excluded"][c["hit_excluded"]] = g["excluded"].get(c["hit_excluded"], 0) + 1
                    g["delta_pos"].append(float(bool(c["delta_pos"]))); g["P_col_all"].append(c["P_after"])
                    g["P_col_before"].append(c.get("P_before")); g["R_col_before"].append(c.get("R_before")); g["R_col_after"].append(c.get("R_after"))
                    if c["delta_pos"]: g["P_col_dpos"].append(c["P_after"])
                    g["rel_recall_col"].append(c["rel_recall"])
                    if c.get("delta") is not None and c["delta"] == c["delta"] and math.isfinite(c["delta"]):
                        g["delta_vs_recall"].append((c["delta"], c["rel_recall"]))   # membership decision vs realized separation (E8)
                if "P_full" in c:
                    g["P_full"].append(c["P_full"])
                    if "P_rel" in c: g["P_rel"].append(c["P_rel"])
    def mean(v):
        v = [x for x in v if x is not None and x == x]
        return float(np.mean(v)) if v else float("nan")
    def corr(pairs):
        if len(pairs) < 3: return float("nan")
        x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
        return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    table = {}
    for key, g in sorted(out.items(), key=lambda kv: (kv[0] is None, kv[0])):
        table[key] = dict(n=g["n"], column_measurable=all(g["measurable"]) if g["measurable"] else None,
                          exact_set_acc=mean(g["exact_set"]), ruler_recall=mean(g["recall"]), row_precision=mean(g["P_row"]),
                          row_recall=mean(g["R_row"]), row_precision_before=mean(g["P_row_before"]), row_recall_before=mean(g["R_row_before"]),
                          interval_hit_rate=mean(g["hit"]), n_hit_columns=len(g["hit"]), hit_excluded_counts=g["excluded"],
                          interval_hit_rate_by_kind={k: mean(v) for k, v in g.get("hit_by_kind", {}).items()},
                          m_j_by_kind={k: sorted(set(x for x in v if x is not None)) for k, v in g.get("m_by_kind", {}).items()},
                          n_columns=g["n_cols"], n_columns_m0=g["m0"], col_recall_before=mean(g["R_col_before"]), col_recall_after=mean(g["R_col_after"]),
                          col_precision_before=mean(g["P_col_before"]), coverage_residual_mean=mean(g["residual"]),
                          coverage_residual_by_kind={k: mean(v) for k, v in g.get("residual_by_kind", {}).items()},
                          site_by_kind={k: sorted(v) for k, v in g.get("site_by_kind", {}).items()},
                          corr_delta_relation_recall=corr(g["delta_vs_recall"]),
                          frac_columns_delta_pos=mean(g["delta_pos"]), col_precision_all=mean(g["P_col_all"]),
                          col_precision_delta_pos=mean(g["P_col_dpos"]), relation_recall_col=mean(g["rel_recall_col"]),
                          relation_recall_row=mean(g["rel_recall_row"]), excluded_by_stage1=mean(g["excl_stage1"]),
                          rejected_by_row=mean(g["rej_row"]), surviving=mean(g["surviving"]),
                          dense_P_full=mean(g["P_full"]), dense_P_rel=mean(g["P_rel"]),
                          **{k_: mean(v) for k_, v in g.get("qa", {}).items()})
    return table


def quick_eval_factory(examples_ruler: list, examples_qa: list, tok, l_star: int, h_star: int, arm: str, n_gen: int = 50, device="cuda"):
    """training Sec. 8.2: teacher-forced loss / coverage / hit rate / row P-R / distractor split on the held-out set;
    exact-set accuracy by greedy generation on n_gen RULER examples."""
    def quick_eval(model, ctx, step):
        rows = evaluate_ruler(model, tok, ctx, examples_ruler, l_star, h_star, arm, modes=("teacher",), device=device)
        # 160 new tokens, not the evaluation queue's 32: 32 hold three values, so exact-set was impossible for every m >= 4
        # example and read 0.50 in every pilot whatever the model did (2026-09-21).  A diagnostic; the queue keeps the spec's 32.
        # spread over the whole list: the files arrive in glob order, and the first 50 examples were the K32_V1 and K32_V4
        # sets only -- no m = 16 example was ever generated, the one length where the arms differ (2026-09-21)
        pick = sorted(set(np.linspace(0, len(examples_ruler) - 1, min(n_gen, len(examples_ruler))).round().astype(int).tolist()))
        gen = evaluate_ruler(model, tok, ctx, [examples_ruler[i] for i in pick], l_star, h_star, arm, modes=("generation",), device=device,
                             max_new_tokens=None)                              # each list's own budget (9 m + 16)
        agg = aggregate_ruler(rows, stratify=None)[None]
        res = dict(tf_loss=float(np.mean([r["tf_loss"] for r in rows])), exact_set_acc=float(np.mean([r["exact_set"] for r in gen])) if gen else None,
                   interval_hit_rate=agg["interval_hit_rate"], row_precision=agg["row_precision"], row_recall=agg["row_recall"],
                   excluded_by_stage1=agg["excluded_by_stage1"], rejected_by_row=agg["rejected_by_row"])
        if gen:
            res["ruler_recall"] = float(np.mean([r["ruler_recall"] for r in gen]))
            by_m = {}
            for r in gen: by_m.setdefault(int(r["m"]), []).append(r["ruler_recall"])
            res["ruler_recall_by_m"] = {str(k): float(np.mean(v)) for k, v in sorted(by_m.items())}
            # the list errors softmax makes on long lists -- the headroom exclusion could claim (owner's B0 gate, 2026-09-22)
            le = [r["list_errors"] for r in gen if "list_errors" in r]
            if le:
                res["list_repeated"] = float(np.mean([x["repeated"] for x in le])); res["list_omitted"] = float(np.mean([x["omitted"] for x in le]))
                res["list_extra"] = float(np.mean([x["extra"] for x in le])); res["list_exact"] = float(np.mean([x["exact"] for x in le]))
                res["list_precision"] = float(np.mean([x["precision"] for x in le]))
                by_task = {}
                for r in gen:
                    if "list_errors" in r: by_task.setdefault(f"{r['task']}_m{int(r['m'])}", []).append(r["list_errors"])
                res["list_errors_by"] = {k: dict(n=len(v), repeated=float(np.mean([x["repeated"] for x in v])), omitted=float(np.mean([x["omitted"] for x in v])),
                                                  extra=float(np.mean([x["extra"] for x in v])), exact=float(np.mean([x["exact"] for x in v])),
                                                  precision=float(np.mean([x["precision"] for x in v])), recall=float(np.mean([x["recall"] for x in v]))) for k, v in sorted(by_task.items())}
        # the relation at the answer rows of (l*, h*): does it CONTAIN the gold keys, how large is it, how much mass does it
        # carry.  The 2026-09 grid had recall 0.000, size ~2 and mass ~1e-9 from its first quick eval, and nothing read it
        # (row_precision read 0.9+ there: precision is 1 on an empty support).  train()'s relation-recall gate reads this.
        rr = [r_ for r in rows for r_ in ((r.get("attention") or {}).get("rows") or []) if "rel_recall" in r_]
        if rr:
            res["relation_recall_row"] = float(np.nanmean([x["rel_recall"] for x in rr]))
            res["relation_size_row"] = float(np.mean([x["E_size"] for x in rr]))
            res["relation_mass_row"] = float(np.mean([x["Rtil"] for x in rr]))
            res["relation_support_nonempty"] = float(np.mean([x["supp_rel"] > 0 for x in rr]))
        if examples_qa:
            q = evaluate_qa(model, tok, ctx, examples_qa, l_star, h_star, arm, modes=("teacher",), device=device)
            res["qa_tf_loss"] = float(np.mean([r["tf_loss"] for r in q]))
        e8 = rows[0].get("e8", {}) if rows else {}
        res["coverage"] = {l: (v["rho"] if v else None) for l, v in e8.items()}
        return res
    return quick_eval


def save_rows(rows: list[dict], path):
    """parquet if pandas/pyarrow are available, else jsonl (Sec. 12.4: one row per unit).  Written under a temporary name
    and renamed, so a killed or concurrent writer never leaves a truncated file behind (review e16a843: run_e6 wrote
    in place)."""
    import os
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
        final = path.with_suffix(".parquet"); tmp = final.with_name(f".{final.name}.tmp{os.getpid()}")
        pd.DataFrame([{k: (json.dumps(v, default=str) if isinstance(v, (dict, list, tuple, set)) else v) for k, v in r.items()} for r in rows]).to_parquet(tmp)
        os.replace(tmp, final)
        return final
    except Exception as e:                           # no pandas/pyarrow, or a column it cannot type: never lose the rows
        print(f"[save_rows] parquet unavailable ({type(e).__name__}: {e}); writing jsonl")
        final = path.with_suffix(".jsonl"); tmp = final.with_name(f".{final.name}.tmp{os.getpid()}")
        with open(tmp, "w") as f:
            for r in rows: f.write(json.dumps(r, default=str) + "\n")
        os.replace(tmp, final)
        return final
