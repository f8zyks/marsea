"""Evaluate a trained arm on a RULER / MuSiQue / HotpotQA set in both modes; writes per-example rows (parquet) and the
aggregated table with the pre-committed decision rule printed beside it.  For B5 pass --b5_marsea_ckpt.

Every output is named by --tag (the queue's arm/seed/experiment tag) and carries its provenance -- arm, seed, set,
checkpoint, git, detector -- in every row and in the table: the stem used to be {arm}_{kind}_{experiment}, so three
seeds, E3 and its two controls, and all seven E9 arms each collapsed onto ONE file, and the rows could not say which
job wrote them (review e982f83 B-2)."""
import argparse, glob, hashlib, json, os, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch


def read_checkpoint_meta(ckpt) -> dict:
    """the checkpoint's own record (config, detector stamp, git, step) without keeping its tensors."""
    if not ckpt:
        return {}
    ck = torch.load(ckpt, map_location="cpu", weights_only=False) or {}
    return dict(config=ck.get("config") or {}, detector=ck.get("detector"), git=ck.get("git"), step=ck.get("step"),
                spec_version=ck.get("spec_version"))


def load_arm(ckpt, arm, backbone, layers, arm_kwargs, mode="dense", device="cuda", head_block=None, det=None):
    """The arm's kwargs come from the CHECKPOINT unless the caller overrides them.  An E9 arm evaluated without the
    kwargs it trained with is a different model wearing its weights -- and the eval queue cannot be expected to
    re-type every ablation's JSON (review d5bd980 F)."""
    from marsea.train import TrainConfig, build_model, load_checkpoint
    meta = read_checkpoint_meta(ckpt)
    cfg_ck = meta.get("config") or {}
    if ckpt:
        check_detector(ckpt, meta, layers, det)
    if ckpt and not arm_kwargs:
        arm_kwargs = cfg_ck.get("arm_kwargs") or {}
        if arm_kwargs:
            print(f"[eval] arm_kwargs from the checkpoint: {arm_kwargs}")
    if ckpt and cfg_ck.get("mode") == "dense" and mode != "dense":
        # the arms that CANNOT run chunked (uniform quota, K_ret) trained dense and must be evaluated dense
        print(f"[eval] checkpoint trained dense: forcing --mode dense (was {mode})"); mode = "dense"
    if head_block is None:
        head_block = cfg_ck.get("head_block")
    cfg = TrainConfig(arm=arm, backbone=backbone, patched_layers=layers, arm_kwargs=arm_kwargs, mode=mode,
                      head_block=head_block)
    model, tok, ctx, layers = build_model(cfg, phase_a=(arm.lower() == "b0"))
    if ckpt:
        load_checkpoint(ckpt, model)
    model.eval()
    return model, tok, ctx


def check_detector(ckpt, meta, layers, det):
    """Nothing asserted that the detector at eval time is the one the checkpoint trained against (review e982f83 I).
    Checkpoints written before the stamp existed carry the provisional default in config.patched_layers, so only the
    stamp is trusted; its absence is reported, not guessed at."""
    stamp = meta.get("detector")
    if not stamp:
        print(f"[eval] WARNING: {ckpt} carries no detector stamp (written before e982f83's fix); layers unchecked")
        return
    if [int(x) for x in stamp.get("patched_layers") or []] != [int(x) for x in layers]:
        raise SystemExit(f"{ckpt} trained with patched layers {stamp.get('patched_layers')}, but this evaluation patches "
                         f"{list(layers)} -- the detector changed between training and evaluation")
    if det is not None and (stamp.get("l_star"), stamp.get("h_star")) != (det.get("l_star"), det.get("h_star")):
        raise SystemExit(f"{ckpt} trained against (l*, h*) = {(stamp.get('l_star'), stamp.get('h_star'))}, the detector "
                         f"now names {(det.get('l_star'), det.get('h_star'))}")


def balanced_take(per_file: list, n) -> list:
    """the first n examples spread EVENLY over the matched configs.  `[: n]` over a sorted concatenation kept the first
    configs whole and dropped the rest (lexicographic order: K128 and K16 first), so capping E3 at n = 400 would have
    evaluated two of the five n-levels.  A config holding fewer than its quota is reported, not silently short
    (review e16a843: a ragged set returned 330 of 400)."""
    if n is None:
        return [x for xs in per_file for x in xs]
    k = len(per_file)
    quota = [n // k + (1 if i < n % k else 0) for i in range(k)]
    short = [(i, len(xs), q) for i, (xs, q) in enumerate(zip(per_file, quota)) if len(xs) < q]
    if short:
        print(f"[eval] WARNING: {len(short)} config(s) under-deliver their quota (index, have, want): {short}; "
              f"{sum(min(len(xs), q) for xs, q in zip(per_file, quota))} of {n} examples")
    return [x for xs, q in zip(per_file, quota) for x in xs[:q]]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def eval_length(exs, set_name: str) -> int:
    """the job's sequence length, for unit_cap_record: the longest tokenised prompt + gold (every kind carries
    prompt_ids / gold_ids), falling back to the `_L<n>` in a RULER set name.  0 only if neither is available, in which
    case the record carries no tolerance and says so by its None (review 7e05ad6 2)."""
    import re
    lens = [len(getattr(e, "prompt_ids", None) or []) + len(getattr(e, "gold_ids", None) or []) for e in (exs or [])]
    if lens and max(lens) > 0:
        return int(max(lens))
    m = re.search(r"_L(\d+)", set_name or "")
    return int(m.group(1)) if m else 0


def atomic_write_text(path: pathlib.Path, text: str):
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text); os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="marsea"); ap.add_argument("--ckpt", default=None)
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-1.5B"); ap.add_argument("--detector", default="runs/detector.json")
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--set", required=True, help="glob of RULER config dirs or a MuSiQue jsonl (a `_L<n>` in the name is its length)")
    ap.add_argument("--kind", choices=["ruler", "musique", "hotpot"], default="ruler")
    ap.add_argument("--stratify", default="m"); ap.add_argument("--modes", nargs="*", default=["teacher", "generation"])
    ap.add_argument("--n", type=int, default=None,
                    help="examples to evaluate; for a RULER glob they are spread evenly over the matched configs")
    ap.add_argument("--out", default="runs/eval")
    ap.add_argument("--tag", default=None, help="output stem (the queue passes its job tag); default arm_kind_experiment_sSEED")
    ap.add_argument("--seed", type=int, default=None, help="recorded in every row; default: the checkpoint's own seed")
    ap.add_argument("--mode", default="dense", choices=["dense", "chunked"]); ap.add_argument("--arm_kwargs", default="{}")
    ap.add_argument("--head_block", type=int, default=None,
                    help="Q-heads per block, as preflight profiles it; default: the checkpoint's training value")
    ap.add_argument("--baseline_head_block", type=int, default=None,
                    help="B0/B1/B2/B4: that many Q-heads at a time on the dense path (B4 and B2 do not fit a 16K job otherwise)")
    ap.add_argument("--b5_marsea_ckpt", default=None); ap.add_argument("--b5_marsea_kwargs", default="{}")
    ap.add_argument("--paired_marsea_ckpt", default=None, help="dense arms / B4: MarSea checkpoint whose relation E[l*,h*] defines the paired domain (m_j/|E_.j|)")
    ap.add_argument("--experiment", default="E2")
    ap.add_argument("--coref_site", type=int, nargs=2, default=None,
                    help="(layer head) at which D-9a's COREFERENCE columns are scored; default: the site S0 licensed")
    ap.add_argument("--value_site", type=int, nargs=2, default=None,
                    help="(layer head) at which the VALUE columns are scored; default: the site S0 licensed")
    ap.add_argument("--allow_no_s0", action="store_true",
                    help="evaluate although the detector carries no S0 licence (smoke tests only: every column is then "
                         "reported as NOT measurable, never as measurable by default)")
    args = ap.parse_args()
    from marsea import SPEC_VERSION
    from marsea.train import git_hash, unit_cap_record
    from marsea.evaluate import (evaluate_ruler, evaluate_qa, aggregate_ruler, save_rows, DECISION_RULES, b5_two_pass,
                                 attention_level_ruler, PairedRelation)
    from marsea.data.ruler import load_set, build_example
    from marsea.data.qa import load_musique, musique_example, tokenize_example, load_hotpot, hotpot_example
    det = json.load(open(args.detector)) if pathlib.Path(args.detector).exists() else None
    layers = args.layers or (det["patched_layers"] if det else [12, 14, 16, 18])
    l_star, h_star = (det["l_star"], det["h_star"]) if det else (layers[-1], 0)
    # S0 is the gate on E7's column half (spec Sec. 12.3).  column_measurable used to DEFAULT TO TRUE, so a grid
    # trained with S0 never run reported every column as measurable without a licence (review e982f83 I).
    if det is None or "s0_licence" not in det:
        if not args.allow_no_s0:
            raise SystemExit(f"{args.detector} carries no S0 licence: run scripts/run_s0.sh before evaluating "
                             "(or pass --allow_no_s0 for a smoke test, which reports no column as measurable)")
        measurable = False
    else:
        measurable = det.get("column_measurable", False)
    meta = read_checkpoint_meta(args.ckpt)
    seed = args.seed if args.seed is not None else (meta.get("config") or {}).get("seed")
    model, tok, ctx = load_arm(args.ckpt, args.arm, args.backbone, layers, json.loads(args.arm_kwargs), args.mode,
                               head_block=args.head_block, det=det)
    ctx.baseline_head_block = args.baseline_head_block or None          # the ARM's model only; inert on MarSea / B3 / B5
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.arm}_{args.kind}_{args.experiment}_s{seed}"
    # D-9a: the coreference column is measured where S0 licensed it.  If that is not (l*, h*), the pass keeps TWO
    # sites and attention_level_ruler scores each kind at its own; otherwise the "licensed anywhere" gate would
    # license a measurement that is then taken at (l*, h*) regardless (review d5bd980 E).
    # BOTH kinds have a licence and a site.  Only `coref` had a way through, so a `value` kind licensed by
    # usable_anywhere alone was still scored at (l*, h*) and reported as licensed (review 30372ae D).
    kind_sites = {}
    for kind, flag in (("coref", args.coref_site), ("value", args.value_site)):
        st = tuple(flag) if flag else (((det.get("s0_licence") or {}).get(kind) or {}).get("site") if det else None)
        if st and list(st) != [l_star, h_star]:
            kind_sites[kind] = tuple(int(x) for x in st)
    for kind, st in kind_sites.items():
        print(f"[D-9a] {kind} columns scored at (layer, head) = {st}; everything else at ({l_star}, {h_star})")
    sites = [(l_star, h_star)] + [(int(l), int(h)) for l, h in kind_sites.values()]
    paired = None
    if args.paired_marsea_ckpt and args.arm.lower() in ("b0", "b1", "b2", "b4"):
        # A7: the headline precision of a dense arm is on MarSea's paired relation E_.j from the SAME example, at
        # every site, built per example as it is scored (PairedRelation: keyed by (layer, head), nothing cached)
        mm, _, cm = load_arm(args.paired_marsea_ckpt, "marsea", args.backbone, layers, json.loads(args.b5_marsea_kwargs),
                             args.mode, head_block=args.head_block, det=det)
    files = []
    if args.kind == "ruler":
        files = [f for pat in args.set.split(",") for f in sorted(glob.glob(pat if pat.endswith(".jsonl") else pat + "/validation.jsonl"))]
        if not files:
            raise SystemExit(f"--set {args.set} matched no RULER files")
        per_file = [[(f, r) for r in load_set(f)] for f in files]
        picked = balanced_take(per_file, args.n)
        exs = [build_example(r, tok) for _, r in picked]
        src = [pathlib.Path(f).parent.name for f, _ in picked]
        if args.arm.lower() == "b5":
            mm, _, cm = load_arm(args.b5_marsea_ckpt or args.paired_marsea_ckpt, "marsea", args.backbone, layers,
                                 json.loads(args.b5_marsea_kwargs), args.mode, head_block=args.head_block, det=det)
            rows = []
            for k, ex in enumerate(exs):
                att = b5_two_pass(mm, cm, model, ctx, ex, l_star, h_star, attention_level_ruler)
                allowed = measurable if isinstance(measurable, dict) else {"coref": bool(measurable), "value": bool(measurable)}
                dropped = sorted({c["kind"] for c in att["cols"] if not allowed.get(c.get("kind", "coref"), True)})
                att["cols"] = [c for c in att["cols"] if allowed.get(c.get("kind", "coref"), True)]
                if dropped: att["cols_not_measurable"] = dropped
                rows.append(dict(example=k, m=len(ex.outputs), n=ex.meta.get("num_needle_k"), attention=att, task_level="n/a by design (D-29)",
                                 column_measurable=measurable))
        else:
            if args.paired_marsea_ckpt and args.arm.lower() in ("b0", "b1", "b2", "b4"):
                paired = PairedRelation(mm, cm, exs, l_star, h_star, sites)
            rows = evaluate_ruler(model, tok, ctx, exs, l_star, h_star, args.arm, modes=tuple(args.modes), paired_E=paired,
                                  column_measurable=measurable, kind_sites=kind_sites)
        table = aggregate_ruler(rows, stratify=args.stratify)
    else:
        if args.kind == "musique":
            files = [args.set]
            recs = load_musique(args.set)[: args.n]
            exs = [tokenize_example(musique_example(r, 0), tok) for r in recs]
        else:
            ds = load_hotpot("validation", n=args.n)
            exs = [tokenize_example(hotpot_example(r, 0), tok) for r in ds]
        src = [args.kind] * len(exs)
        if args.paired_marsea_ckpt and args.arm.lower() in ("b0", "b1", "b2", "b4"):
            paired = PairedRelation(mm, cm, exs, l_star, h_star, sites)
        rows = evaluate_qa(model, tok, ctx, exs, l_star, h_star, args.arm, modes=tuple(args.modes),
                           hotpot=(args.kind == "hotpot"), kind_sites=kind_sites, paired_E=paired)
        table = aggregate_ruler(rows, stratify="m")
    prov = dict(tag=tag, arm=args.arm, seed=seed, experiment=args.experiment, kind=args.kind, set=args.set,
                stratify=(args.stratify if args.kind == "ruler" else "m"),
                ckpt=args.ckpt, ckpt_git=meta.get("git"), ckpt_step=meta.get("step"),
                ckpt_arm_kwargs=(meta.get("config") or {}).get("arm_kwargs"),
                paired_marsea_ckpt=args.paired_marsea_ckpt if paired is not None else None,
                git=git_hash(), spec_version=SPEC_VERSION, mode=args.mode, head_block=args.head_block,
                # the path the arm RAN on: load_arm forces dense for a checkpoint that trained dense, and `mode` alone said
                # `chunked` for the three dense-trained E9 arms (K64, uniform_quota, its control) in the 2026-09 campaign
                mode_effective=ctx.mode,
                baseline_head_block=args.baseline_head_block, cuda_alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                detector=args.detector, detector_sha256=sha256_file(args.detector) if det else None,
                l_star=l_star, h_star=h_star, patched_layers=list(layers), sites=[list(s) for s in sites],
                s0_licensed=(det is not None and "s0_licence" in det), n_examples=len(rows),
                # at the job's LENGTH: unit_cap_record(L) validates the MARSEA_TOL_CAP override against L eps / 4 and
                # records the tolerance and ceiling in force -- with no length the eval tables carried None for both and
                # an override the ceiling exists to refuse passed the whole queue unremarked (review 7e05ad6 2)
                unit_cap=unit_cap_record(eval_length(exs, args.set)),
                files={pathlib.Path(f).parent.name if args.kind == "ruler" else f: sha256_file(f) for f in files},
                created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    for r, s in zip(rows, src):
        r.update(tag=tag, arm=args.arm, seed=seed, experiment=args.experiment, set=s, ckpt=args.ckpt, git=prov["git"])
    p = save_rows(rows, out / tag)                     # atomic: written under a temporary name and renamed
    print(f"rows -> {p}")
    print(f"=== {args.experiment} ({args.arm}, {tag}) ===\nDECISION RULE: {DECISION_RULES.get(args.experiment, '')}")
    print(json.dumps(table, indent=1, default=str))
    atomic_write_text(out / f"{tag}_table.json",
                      json.dumps(dict(rule=DECISION_RULES.get(args.experiment), provenance=prov, table=table), indent=1, default=str))


if __name__ == "__main__":
    main()
