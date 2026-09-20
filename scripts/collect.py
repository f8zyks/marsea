"""Build the paper's tables from the evaluation queue's outputs (review e982f83 G).

Nothing aggregated across seeds and nothing built the E7/E8 tables: rec["e8"] was written per row and read by no one, and
E3's relation-coverage rate |E_.j| / n -- which the paper says E3 "reports alongside" -- was computed per column and
dropped.  This reads runs/eval/<tag>_table.json (+ <tag>.parquet) and writes:

  collected.json   {job: {stratum: {metric: {mean, std, n_seeds, per_seed}}}}, the E8 block per (job, layer), the
                   E3 coverage rate per stratum, and every input's provenance
  collected.md     the same, as tables a reader can check against the draft

A job is the tag with its seed removed (marsea_s1_E3 -> marsea_E3).  Seeds are only pooled when their provenance agrees
on arm, experiment, git commit and detector.  A disagreement is recorded in provenance_conflicts, that job is emitted
per seed and NOT pooled, every other job is still written, and the script exits non-zero (review e16a843 G-2).

Every metric aggregate_ruler writes is carried (a blacklist, not a whitelist: the whitelist dropped E7's "recall twice"
readings, support_EM, the sample sizes and the D-9a per-kind splits -- G-1).  Per-kind dicts become "metric[kind]".

Two captions the tables need (G-4): the coverage row is |E_.j| / n_q -- the number of QUERIES, not the stratum label n --
and for a dense arm E_size is MarSea's paired relation, not the arm's own (it has none).  E8 carries a seed dimension
(mean and std over the per-seed values) like everything else.

  python scripts/collect.py --eval_dir runs/eval
"""
import argparse, glob, json, math, pathlib, re, sys
from collections import defaultdict
import numpy as np

SEED_RE = re.compile(r"_s(\d+)(?=_|$)")
NOT_POOLED = ("corr_delta_relation_recall",)      # a within-seed correlation; per_seed only, never averaged into a mean
CARRIED_RAW = ("site_by_kind", "m_j_by_kind", "column_measurable", "hit_excluded_counts")   # provenance-like, per seed
E8_SCALARS = ("rho", "frac_Ecol_singleton", "var_tau_j", "var_tau_i", "tau_j_median", "tau_i_median",
              "frac_rows_over_unit", "frac_rows_zero_mass", "frac_rows_zero_mass_ambiguous", "frac_cbar_i_zero",
              "row_trigger_rate", "col_induced_rate", "rho_col", "rho_row",
              "zero_mass_tol")                    # the tolerance that DEFINES frac_rows_zero_mass (review 9bacef9 E)
E8_PER_HEAD = ("rho_head",)                       # NEVER pooled across heads (fidelity.py): mean over examples per head
E8_HISTS = ("hist_Ecol", "hist_Erow", "hist_kstar", "kstar_by_Ecol")
E8_TAGS = ("source",)                             # which E8 implementation (dense / sparse) produced the block


def job_of(tag: str) -> str:
    return SEED_RE.sub("", tag, count=1)


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _stat(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return dict(mean=None, std=None, n_seeds=0)
    return dict(mean=float(np.mean(v)), std=(float(np.std(v, ddof=1)) if len(v) > 1 else None), n_seeds=len(v))


def _flat_mean(values):
    """mean over a scalar, a per-head list, or None"""
    out = []
    for v in values:
        if isinstance(v, list):
            out += [x for x in (_num(y) for y in v) if x is not None]
        elif _num(v) is not None:
            out.append(_num(v))
    return float(np.mean(out)) if out else None


def _hist_mean(values):
    """mean of {label: fraction} dicts, given per head (list of dicts) or pooled (dict)."""
    acc = defaultdict(list)
    for v in values:
        for d in (v if isinstance(v, list) else [v]):
            if isinstance(d, dict):
                for k, x in d.items():
                    if _num(x) is not None:
                        acc[k].append(_num(x))
    return {k: float(np.mean(x)) for k, x in acc.items()} or None


def read_rows(parquet: pathlib.Path):
    if parquet.exists():
        import pandas as pd
        df = pd.read_parquet(parquet)
        rows = df.to_dict("records")
    else:
        jl = parquet.with_suffix(".jsonl")
        rows = [json.loads(l) for l in open(jl)] if jl.exists() else []
    for r in rows:
        for k in ("attention", "e8", "sites"):
            if isinstance(r.get(k), str):
                try:
                    r[k] = json.loads(r[k])
                except ValueError:
                    pass
    return rows


def e8_block(rows):
    """per patched layer: E8's pre-registered readings pooled over examples and heads."""
    by_layer = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for layer, blk in (r.get("e8") or {}).items():
            if not isinstance(blk, dict):
                continue
            for k in E8_SCALARS + E8_HISTS + E8_PER_HEAD + E8_TAGS + ("truncation_flag", "row_trigger_violation",
                                                                       "regime_q10_50_90", "pass2_rebuilt_row_frac", "_pass2_rows"):
                if k in blk:
                    by_layer[str(layer)][k].append(blk[k])
    out = {}
    for layer, d in sorted(by_layer.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
        o = {k: _flat_mean(d[k]) for k in E8_SCALARS if k in d}
        o.update({k: _hist_mean(d[k]) for k in E8_HISTS if k in d})
        for k in E8_PER_HEAD:                       # per head, over examples: a list stays a list (review 7814665 F)
            vals = [v for v in d.get(k, []) if isinstance(v, list) and v]
            if vals:
                nh = min(len(v) for v in vals)
                o[k] = [float(np.mean([_num(v[h]) for v in vals if _num(v[h]) is not None])) for h in range(nh)]
        for k in E8_TAGS:
            o[k] = sorted({str(v) for v in d.get(k, []) if v is not None}) or None
        # the pass-2 rebuild fraction is COUNT-weighted by its rows, as the chunked path computes it
        w = [int(x or 0) for x in d.get("_pass2_rows", [])]; f = [_num(x) for x in d.get("pass2_rebuilt_row_frac", [])]
        if f and any(x is not None for x in f):
            o["pass2_rebuilt_row_frac"] = (float(sum(fi * wi for fi, wi in zip(f, w) if fi is not None) / sum(w))
                                           if w and sum(w) and len(w) == len(f) else _flat_mean(f))
            o["pass2_rows"] = int(sum(w))                                  # its weight, emitted (review 9bacef9 E)
        o["truncation_flag_rate"] = _flat_mean([float(bool(x)) for x in d.get("truncation_flag", []) if x is not None])
        o["row_trigger_violations"] = int(sum(bool(x) for x in d.get("row_trigger_violation", [])))
        # a block's value is per head (a list of [q10, q50, q90]) or pooled (one triple): the pooled form bound the
        # comprehension's `x` to a float and the row came back empty (review e16a843 G-3)
        q = []
        for v in d.get("regime_q10_50_90", []):
            for x in (v if (isinstance(v, list) and v and isinstance(v[0], list)) else [v]):
                if isinstance(x, list) and len(x) == 3 and all(_num(y) is not None for y in x):
                    q.append(x)
        o["regime_median_of_q10_50_90"] = [float(np.median([x[i] for x in q])) for i in range(3)] if q else None
        # E8's reading rules, pre-registered (evaluate.DECISION_RULES["E8"]); reported, never acted on
        o["reading_became_B0"] = (o.get("rho") is not None and o["rho"] < 0.005)
        o["reading_trained_out_of_thm1"] = (o.get("hist_Ecol") or {}).get("1", 0.0) > 0.8 if o.get("hist_Ecol") else None
        out[layer] = o
    return out


def e8_by_seed(rows):
    """E8 per layer with a SEED dimension: each seed's block, then mean/std of every scalar across seeds."""
    by_seed = defaultdict(list)
    for r in rows:
        by_seed[r.get("seed")].append(r)
    seeds = sorted(by_seed, key=lambda x: (x is None, str(x)))
    blocks = {sd: e8_block(by_seed[sd]) for sd in seeds}
    out = {}
    for layer in sorted({l for b in blocks.values() for l in b}, key=lambda x: int(x) if str(x).isdigit() else str(x)):
        # POSITIONAL over `seeds`: a seed whose rows carry no block for this layer holds None at its index rather than
        # shortening the list (the jobs' per_seed fix of 7814665 was not applied here -- review 9bacef9 E)
        per = [blocks[sd].get(layer) for sd in seeds]
        pooled = e8_block([r for rs in by_seed.values() for r in rs])[layer]
        o = dict(pooled)                                     # histograms and flags pooled over seeds
        for k, v in pooled.items():
            if k == "pass2_rows":
                continue                                         # the pooled COUNT, not a per-seed statistic
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                vals = [(_num(p.get(k)) if p else None) for p in per]
                o[k] = dict(**_stat(vals), per_seed=vals, seeds=seeds)
                if k == "pass2_rebuilt_row_frac":
                    # the pooled value is COUNT-weighted by pass-2 rows; an unweighted mean over seeds overwrote it
                    o[k]["mean"] = v
        out[layer] = o
    return out


def _flatten_metrics(metrics: dict) -> dict:
    """every scalar metric, and per-kind dicts as metric[kind]; raw fields kept aside."""
    flat, raw = {}, {}
    for k, v in metrics.items():
        if k in CARRIED_RAW:
            raw[k] = v
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if _num(vv) is not None or vv is None:
                    flat[f"{k}[{kk}]"] = _num(vv)
        elif isinstance(v, bool):
            raw[k] = v
        elif isinstance(v, (int, float)) or v is None:
            flat[k] = _num(v)
    return flat, raw


def coverage_by_stratum(rows, stratify, seeds=None):
    """E3's relation-coverage rate: |E_.j| / n_q per active column, per stratum (MarSea's own relation, or the paired
    one for a dense arm).  With a seed dimension like everything else: per-seed means (positional over `seeds`) and
    their mean/std (review 9bacef9 E); `mean` stays the column-pooled rate."""
    acc = defaultdict(list)
    by_seed = defaultdict(lambda: defaultdict(list))
    for r in rows:
        att = r.get("attention") or {}
        n_q = att.get("n_q")
        for c in att.get("cols", []):
            if c.get("E_size") is not None and n_q:
                acc[str(r.get(stratify))].append(c["E_size"] / n_q)
                by_seed[str(r.get(stratify))][r.get("seed")].append(c["E_size"] / n_q)
    seeds = list(seeds) if seeds is not None else sorted({sd for d in by_seed.values() for sd in d}, key=lambda x: (x is None, str(x)))
    out = {}
    for k, v in acc.items():
        per = [(float(np.mean(by_seed[k][sd])) if by_seed[k].get(sd) else None) for sd in seeds]
        out[k] = dict(mean=float(np.mean(v)), n_columns=len(v), per_seed=per, seeds=seeds,
                      seed_mean=_stat(per)["mean"], seed_std=_stat(per)["std"], n_seeds=_stat(per)["n_seeds"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", default="runs/eval")
    ap.add_argument("--out", default=None, help="default: <eval_dir>/collected.{json,md}")
    ap.add_argument("--equivalent_commits", default=None,
                    help="a JSON declaration {commits: [hash, ...], evidence: str, decided_by: str, date: str}: these commits "
                         "are ONE commit for the pooling rule.  For a campaign whose queue was fixed mid-run by a commit that "
                         "does not touch the pooled arms' path; the declaration and the jobs it pooled go into collected.json")
    args = ap.parse_args()
    equiv = None
    if args.equivalent_commits:
        equiv = json.loads(pathlib.Path(args.equivalent_commits).read_text())
        commits = [str(c) for c in equiv.get("commits") or []]
        if len(commits) < 2 or any(len(c) < 7 for c in commits) or not str(equiv.get("evidence") or "").strip():
            sys.exit("--equivalent_commits needs >= 2 commit hashes of >= 7 characters and a non-empty `evidence`")
        equiv = dict(equiv, commits=commits, pooled_jobs=[])

    def canon_git(g: str) -> str:
        g = (g or "").replace("+dirty", "")
        if equiv and g and any(g.startswith(c) or c.startswith(g) for c in equiv["commits"]):
            return "equivalent:" + "=".join(c[:7] for c in equiv["commits"])
        return g
    ed = pathlib.Path(args.eval_dir)
    tables = sorted(ed.glob("*_table.json"))
    if not tables:
        sys.exit(f"no *_table.json under {ed}")
    jobs = defaultdict(list)
    for t in tables:
        payload = json.loads(t.read_text())
        prov = payload.get("provenance") or {}
        tag = prov.get("tag") or t.name[: -len("_table.json")]
        jobs[job_of(tag)].append((tag, payload, prov))
    result = dict(jobs={}, e8={}, coverage={}, provenance={}, provenance_conflicts={},
                  captions=dict(coverage="|E_.j| / n_q per active column, per stratum (n_q = queries, not the stratum's n); "
                                         "for dense arms E_.j is MarSea's paired relation; `mean` pools columns over seeds, "
                                         "per_seed / seed_mean / seed_std give the seed dimension",
                                e8="per patched layer; scalars are {mean, std, n_seeds, per_seed} over seeds, histograms "
                                   "pooled over seeds"))
    for job, items in sorted(jobs.items()):
        keyset = {(p.get("arm"), p.get("experiment"), canon_git(p.get("git")), p.get("detector_sha256"))
                  for _, _, p in items}
        conflict = len(keyset) > 1
        if equiv and not conflict and len({(p.get("git") or "").replace("+dirty", "") for _, _, p in items}) > 1:
            equiv["pooled_jobs"].append(job)                          # pooled ONLY because of the declaration: say so
        if conflict:
            result["provenance_conflicts"][job] = sorted(map(str, keyset))
            print(f"!! {job}: seeds disagree on (arm, experiment, git, detector) -- emitted per seed, not pooled",
                  file=sys.stderr)
        per_stratum = defaultdict(lambda: defaultdict(list))
        raw_stratum = defaultdict(lambda: defaultdict(list))
        seeds = []
        rows_all = []
        for si, (tag, payload, prov) in enumerate(sorted(items, key=lambda x: str(x[2].get("seed")))):
            seeds.append(prov.get("seed"))
            table = payload.get("table") or {}
            # E6 writes its strict-match and contains-match tables beside `table` (run_e6.py: the slope is fitted on
            # both); they are read as strata of their own, "exact_match_strict:<tier>" (review 9bacef9 E)
            strata = list(table.items())
            for alt in ("exact_match_strict", "contains"):
                if isinstance(payload.get(alt), dict):
                    strata += [(f"{alt}:{k}", v) for k, v in payload[alt].items()]
            for stratum, metrics in strata:
                # an E6 table's strata are {tier: acc} dicts AND bare scalars (`slope`); a scalar crashed the whole run and
                # nothing was written for any job -- the G-2 failure mode through another door (review 7814665 F-1)
                if not isinstance(metrics, dict):
                    metrics = {"value": metrics}
                flat, raw = _flatten_metrics(metrics)
                # per_seed lists are POSITIONAL: pad to this seed's index BEFORE appending.  Padding after the append
                # put a key first scored at seed k at index 0 and then labelled it seed 0 (review 9bacef9 C-3).
                for k, v in flat.items():
                    lst = per_stratum[stratum][k]; lst += [None] * (si - len(lst)); lst.append(v)
                for k, v in raw.items():
                    lst = raw_stratum[stratum][k]; lst += [None] * (si - len(lst)); lst.append(v)
            rows = read_rows(ed / f"{tag}.parquet")
            for r in rows:
                r.setdefault("seed", prov.get("seed"))
            rows_all += rows
            result["provenance"][tag] = prov
        jr = {}
        for st, m in per_stratum.items():
            jr[st] = {}
            for k, v in m.items():
                v = (v + [None] * len(seeds))[:len(seeds)]                  # positional: one slot per seed, None if unscored
                if conflict or k in NOT_POOLED:
                    jr[st][k] = dict(mean=None, std=None, n_seeds=sum(x is not None for x in v), per_seed=v, seeds=seeds)
                else:
                    jr[st][k] = dict(**_stat(v), per_seed=v, seeds=seeds)
            jr[st]["_per_seed_raw"] = dict(raw_stratum[st])
        jr["_seeds"] = seeds
        result["jobs"][job] = jr
        result["e8"][job] = e8_by_seed(rows_all)
        strat = next((p.get("stratify") for _, _, p in items if p.get("stratify")), None) or \
            ("n" if "_E3" in job and "depth" not in job else "depth" if "depth" in job else "m")
        result["coverage"][job] = coverage_by_stratum(rows_all, strat, seeds)
    out = pathlib.Path(args.out) if args.out else ed / "collected"
    if equiv:
        result["commit_equivalence"] = equiv
    out.with_suffix(".json").write_text(json.dumps(result, indent=1, default=str))
    md = ["# Collected evaluation tables", ""]
    if equiv:
        md += [f"**Commits declared equivalent for pooling** ({', '.join(c[:7] for c in equiv['commits'])}; "
               f"{equiv.get('decided_by', '?')}, {equiv.get('date', '?')}): {equiv['evidence']}  ",
               f"Jobs pooled across them: {', '.join(equiv['pooled_jobs']) or 'none'}.", ""]
    for job, strata in result["jobs"].items():
        md += [f"## {job}  (seeds {strata['_seeds']})", "", "| stratum | metric | mean | std | n seeds |", "|---|---|---|---|---|"]
        for s, m in strata.items():
            if s == "_seeds":
                continue
            for k, v in m.items():
                if k.startswith("_") or v["mean"] is None:
                    continue
                std = "-" if v["std"] is None else f"{v['std']:.4f}"
                md.append(f"| {s} | {k} | {v['mean']:.4f} | {std} | {v['n_seeds']} |")
        cov = result["coverage"].get(job) or {}
        if cov:
            md += ["", "relation coverage \\|E_.j\\| / n_q (queries; dense arms: MarSea's paired relation): " + ", ".join(f"{k}: {v['mean']:.4f} ({v['n_columns']} cols)" for k, v in cov.items())]
        e8 = result["e8"].get(job) or {}
        if e8:
            md += ["", "| layer | rho | frac singleton E_.j | var tau_j | frac rows over unit | frac zero-mass | k* near K_ret |",
                   "|---|---|---|---|---|---|---|"]
            for layer, o in e8.items():
                def f(x):
                    if isinstance(x, dict):
                        return "-" if x.get("mean") is None else (f"{x['mean']:.4f}" + (f" ± {x['std']:.4f}" if x.get("std") is not None else ""))
                    return "-" if x is None else f"{x:.4f}"
                md.append(f"| {layer} | {f(o.get('rho'))} | {f(o.get('frac_Ecol_singleton'))} | {f(o.get('var_tau_j'))} | "
                          f"{f(o.get('frac_rows_over_unit'))} | {f(o.get('frac_rows_zero_mass'))} | {f(o.get('truncation_flag_rate'))} |")
        md.append("")
    out.with_suffix(".md").write_text("\n".join(md))
    print(f"{len(result['jobs'])} jobs from {len(tables)} tables -> {out.with_suffix('.json')}, {out.with_suffix('.md')}")
    if result["provenance_conflicts"]:
        sys.exit(f"provenance conflicts in {len(result['provenance_conflicts'])} job(s): {sorted(result['provenance_conflicts'])}"
                 " -- written unpooled; re-run the stale seeds")


if __name__ == "__main__":
    main()
