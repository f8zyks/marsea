"""
The relation monitor (2026-09-21): direct readings of what the relation and the two programmes do at the answer rows of the
measurement head (l*, h*), on a fixed set of RULER examples -- cheap enough to log every 20-50 steps, and runnable on a saved
checkpoint (scripts/monitor_ckpt.py).

The quick eval's aggregates could not tell a good relation from a broken one: "row precision" is 1 on an empty support
(0.9+ for a dead relation), "recall of gold keys" divides by every token of all m gold values (an attention-shaped
relation tops out near 0.45) and rises when the relation simply swallows half the row (0.95 at 2,300 keys per row).

The unit of measurement here is ONE answer row and the ONE key it is about to copy: row r_k emits token k of gold value
v, and its COPY SOURCE is token k of v's needle in the prompt.  Per row:

  row side     source in the relation / in the final support; its mass under softmax, after the column programme (Atil),
               after the cap (a1) and final (A); its rank in the final row; |E_i.| and the share of it that survives;
               relation mass at each stage (the over-payment is Atil / softmax on the relation); whether the unit cap
               binds; the share of the OFF-relation softmax tail that survives (the cap's projection zeroes it); tau_i;
               max / gap to the second / mean / spread of Atil on the relation
  column side  (the copy source's column) its size -- the true relation set when the row triggers it -- and how many of
               its members are answer rows; the row's share p; the tau used for that solve and the sealed tau_j; the
               quota; max / gap / mean / spread of the members' scores and the row's standing among them (score - max:
               0 = the top member, very negative = starved); precision of the column (members that are this value's rows)

Everything is averaged over rows, overall and by m (long lists are where the method has been losing)."""
from __future__ import annotations
from typing import Optional
import math
import numpy as np
import torch

from .evaluate import teacher_forced_pass, _kept_head, _scores_at

ROW_KEYS = ("src_in_relation", "src_in_support", "src_mass_softmax", "src_mass_Atil", "src_mass_a1", "src_mass_final", "src_rank_final",
            "relation_size", "relation_kept_frac", "relation_mass_softmax", "relation_mass_Atil", "relation_mass_final", "overpay_ratio",
            "cap_binds", "tail_kept_frac", "tail_mass_softmax", "tail_mass_final", "support_size", "tau_i",
            "rel_Atil_max", "rel_Atil_gap", "rel_Atil_mean", "rel_Atil_std")
COL_KEYS = ("col_size", "col_answer_members", "col_precision", "row_share_p", "tau_used", "tau_sealed", "col_quota",
            "col_score_max", "col_score_gap", "col_score_mean", "col_score_std", "row_standing")


def copy_sources(ex) -> list:
    """[(answer row, its copy-source key, value index, token index)]: row answer_value_rows[v][k] copies prompt token
    value_spans[v][0] + k.  A value whose needle and answer tokenise to different lengths is skipped."""
    out = []
    for v, (span, rows) in enumerate(zip(ex.value_spans, ex.answer_value_rows)):
        a, b = span
        if b - a != len(rows):
            continue
        out += [(int(r), int(a + k), v, k) for k, r in enumerate(rows)]
    return out


def _stats(x: torch.Tensor):
    """max, gap to the second, mean, std of a 1-D tensor (zeros when empty)."""
    if x.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0
    top = torch.topk(x, min(2, x.numel())).values
    return float(top[0]), float(top[0] - top[-1]) if x.numel() > 1 else 0.0, float(x.mean()), float(x.std()) if x.numel() > 1 else 0.0


@torch.no_grad()
def monitor_example(model, ctx, ex, l_star: int, h_star: int, device="cuda") -> list:
    """one teacher-forced pass; one record per (answer row, copy source)."""
    _, _, dg = teacher_forced_pass(model, ctx, ex.prompt_ids, ex.gold_ids, l_star, device, h_star=h_star, sites=[(l_star, h_star)])
    d = dg[l_star]
    if d.E is None or d.Atil is None:
        return []
    hh = _kept_head(d, h_star); n = d.A.shape[-1]; n_p = len(ex.prompt_ids)
    A, Asm, Atil, a1, E, p = d.A[0, hh], d.A_sm[0, hh], d.Atil[0, hh], d.a1[0, hh], d.E[0, hh], d.p[0, hh]
    S = _scores_at(d, hh)
    tau_trig = d.extra.get("tau_trig_rows")                                   # [B, H or kept, m, n]: the tau of each triggered solve
    if tau_trig is not None:                                                    # an extra: never narrowed to the kept head
        tau_trig = tau_trig[0, h_star if tau_trig.shape[1] > h_star else 0]
    ar = torch.arange(n, device=A.device)
    value_rows = {v: set(int(r) for r in rows) for v, rows in enumerate(ex.answer_value_rows)}
    recs = []
    for r, j, v, k in copy_sources(ex):
        if r >= n or j >= n:
            continue
        vis = ar <= r; Er = E[r] & vis; off = vis & ~Er; nE = int(Er.sum())
        rel_sm, rel_til, rel_fin = float(Asm[r][Er].sum()), float(Atil[r][Er].sum()), float(A[r][Er].sum())
        mx, gap, mean, std = _stats(Atil[r][Er])
        rec = dict(m=len(ex.outputs), token=k,
                   src_in_relation=float(Er[j]), src_in_support=float(A[r, j] > 0), src_mass_softmax=float(Asm[r, j]),
                   src_mass_Atil=float(Atil[r, j]), src_mass_a1=float(a1[r, j]), src_mass_final=float(A[r, j]),
                   src_rank_final=float((A[r][vis] > A[r, j]).sum()),
                   relation_size=float(nE), relation_kept_frac=float(((A[r] > 0) & Er).sum() / max(1, nE)),
                   relation_mass_softmax=rel_sm, relation_mass_Atil=rel_til, relation_mass_final=rel_fin,
                   overpay_ratio=float("nan"),                               # a ratio of MEANS, filled in by summarise()
                   cap_binds=float(d.cap_binds[0, hh, r]), tail_kept_frac=float(((A[r] > 0) & off).sum() / max(1, int(off.sum()))),
                   tail_mass_softmax=float(Asm[r][off].sum()), tail_mass_final=float(A[r][off].sum()),
                   support_size=float(((A[r] > 0) & vis).sum()), tau_i=float(d.tau_i[0, hh, r]),
                   rel_Atil_max=mx, rel_Atil_gap=gap, rel_Atil_mean=mean, rel_Atil_std=std)
        # ---- the copy source's column, as it stood when row r triggered it: its members are the rows <= r that relate to j
        mem = E[:, j] & (ar >= j) & (ar <= r)                                    # the triggered solve's members (the full-sequence
                                                                                  # form also had the rows after r in its one solve)
        nm = int(mem.sum()); sc = S[:, j][mem]
        cmx, cgap, cmean, cstd = _stats(sc)
        mine = sorted(value_rows.get(v, ()))
        member = bool(Er[j]); nan = float("nan")                                 # the column readings that describe THIS row's solve
        rec.update(col_size=float(nm), col_answer_members=float((mem & (ar >= n_p)).sum()),                # exist only if it is a member
                   col_precision=(float(sum(1 for q in mine if q <= r and bool(mem[q]))) / nm) if nm else nan,
                   row_share_p=float(p[r, j]) if member else nan, tau_sealed=float(d.tau_j[0, hh, j]),
                   tau_used=(float(tau_trig[r - n_p, j]) if (tau_trig is not None and r >= n_p) else float(d.tau_j[0, hh, j])) if member else nan,
                   col_quota=float(d.cbar_j[0, hh, j]), col_score_max=cmx, col_score_gap=cgap, col_score_mean=cmean, col_score_std=cstd,
                   row_standing=(float(S[r, j]) - cmx) if member else nan)
        recs.append(rec)
    return recs


def summarise(recs: list) -> dict:
    """means over rows, overall and by m; NaNs ignored; n = rows."""
    def agg(rows):
        out = dict(n=len(rows))
        for key in ROW_KEYS + COL_KEYS:
            vals = [r[key] for r in rows if key in r and r[key] is not None and not (isinstance(r[key], float) and math.isnan(r[key]))]
            out[key] = float(np.mean(vals)) if vals else None
        if out.get("relation_mass_softmax"):                                     # per-row ratios explode where softmax put ~0 on the relation
            out["overpay_ratio"] = out["relation_mass_Atil"] / out["relation_mass_softmax"]
        return out
    if not recs:
        return dict(n=0)
    res = dict(all=agg(recs), by_m={})
    for m in sorted({r["m"] for r in recs}):
        res["by_m"][str(m)] = agg([r for r in recs if r["m"] == m])
    res["first_token"] = agg([r for r in recs if r["token"] == 0])               # the retrieval step proper: the value's FIRST token
    return res


def monitor_factory(examples: list, l_star: int, h_star: int, device="cuda"):
    """-> callable(model, ctx, step) -> dict for train()'s `monitor` hook."""
    def monitor(model, ctx, step):
        recs = []
        for ex in examples:
            recs += monitor_example(model, ctx, ex, l_star, h_star, device)
        return summarise(recs)
    return monitor


def format_line(step, res: dict) -> str:
    """one readable line per monitor call (the full dict goes to train_log.jsonl)."""
    if not res or not res.get("n", 1) or "all" not in res:
        return f"[monitor] step {step}: no rows"
    f = lambda v, nd=3: "-" if v is None else f"{v:.{nd}f}"
    parts = []
    for name, a in [("all", res["all"]), ("first tok", res["first_token"])] + [(f"m={m}", a) for m, a in res["by_m"].items()]:
        parts.append(f"{name}: src in E {f(a['src_in_relation'], 2)} in supp {f(a['src_in_support'], 2)} mass {f(a['src_mass_softmax'], 2)}>{f(a['src_mass_final'], 2)} "
                     f"| |E_i| {f(a['relation_size'], 0)} overpay x{f(a['overpay_ratio'], 2)} cap {f(a['cap_binds'], 2)} tail kept {f(a['tail_kept_frac'], 2)} "
                     f"| col size {f(a['col_size'], 0)} share {f(a['row_share_p'], 2)} tau {f(a['tau_used'], 2)} standing {f(a['row_standing'], 2)}")
    return f"[monitor] step {step}  " + "  ||  ".join(parts)
