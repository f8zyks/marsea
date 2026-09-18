"""E6 -- IHEval multi-turn rule-following (paper Sec. 5, App. H): accuracy against the number of conflicting instruction
tiers, reported as a SLOPE, never as a pooled mean.

  Prediction: at depth one MarSea and a well-tuned query-local baseline are close; as depth grows the query-local arms
  degrade markedly and MarSea gently.  Falsification: a flat gap across depth withdraws the ranked framing for the
  weaker binary-exclusivity claim.

The loader (marsea/data/iheval.py) reads a jsonl/json of records carrying a system prompt, the turns, a reference and a
tier count; base checkpoint, no chat template (spec Sec. 13.1 [v4.1]).  Scoring is exact-match on the reference after
normalisation, plus a contains-match, both reported -- IHEval's own checkers are task-specific and are not vendored.

  python scripts/run_e6.py --arm marsea --ckpt runs/marsea_seed0/final.pt --set data/iheval/multiturn.jsonl
"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="marsea"); ap.add_argument("--ckpt", default=None)
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-1.5B"); ap.add_argument("--detector", default="runs/detector.json")
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--set", required=True, help="IHEval multi-turn jsonl/json")
    ap.add_argument("--n", type=int, default=None); ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--out", default="runs/eval"); ap.add_argument("--mode", default="dense", choices=["dense", "chunked"])
    ap.add_argument("--arm_kwargs", default="{}")
    ap.add_argument("--head_block", type=int, default=None, help="as run_eval.py: default the checkpoint's training value")
    ap.add_argument("--tag", default=None, help="output stem; default {arm}_iheval_E6")
    args = ap.parse_args()
    from marsea.data.iheval import load_iheval, accuracy_by_tier
    from marsea.data.qa import normalize_answer
    from marsea.evaluate import generate_greedy, save_rows, DECISION_RULES
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from run_eval import load_arm, read_checkpoint_meta, sha256_file, atomic_write_text
    from marsea import SPEC_VERSION
    from marsea.train import git_hash
    det = json.load(open(args.detector)) if pathlib.Path(args.detector).exists() else None
    layers = args.layers or (det["patched_layers"] if det else [12, 14, 16, 18])
    model, tok, ctx = load_arm(args.ckpt, args.arm, args.backbone, layers, json.loads(args.arm_kwargs), args.mode,
                               head_block=args.head_block, det=det)
    meta = read_checkpoint_meta(args.ckpt)
    tag = args.tag or f"{args.arm}_iheval_E6"
    stats = {}
    exs = load_iheval(args.set, stats)[: args.n]
    assert exs, f"no IHEval records with a tier count in {args.set}"
    if stats.get("n_dropped_no_tier"):
        print(f"[E6] {stats['n_dropped_no_tier']} of {stats['n_records']} records dropped for want of a tier count")
    rows = []
    for k, ex in enumerate(exs):
        ids = tok(ex.prompt, add_special_tokens=False)["input_ids"]
        text, dt = generate_greedy(model, tok, ctx, ids, args.max_new_tokens, frozen_prefix=(args.arm.lower() not in ("b0", "b1", "b2", "b4")))
        pred = text.strip().split("\n\n")[0].strip()
        # normalize_answer is QA-SPAN normalisation (lowercase, strip punctuation and articles).  Applied to
        # instruction-following output it is a lenient proxy, so the strict match is reported beside it and the
        # slope is fitted on BOTH: IHEval's own per-task checkers are not vendored (review d5bd980 F).
        em_strict = float(" ".join(pred.split()) == " ".join(ex.reference.split()))
        em = float(normalize_answer(pred) == normalize_answer(ex.reference))
        contains = float(normalize_answer(ex.reference) in normalize_answer(pred)) if ex.reference else float("nan")
        rows.append(dict(example=k, tiers=ex.tiers, generated=pred, reference=ex.reference, correct=em,
                         em_strict=em_strict, contains=contains, decode_seconds=dt, arm=args.arm, tag=tag,
                         seed=(meta.get("config") or {}).get("seed"), ckpt=args.ckpt))
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    p = save_rows(rows, out / tag)
    prov = dict(tag=tag, arm=args.arm, seed=(meta.get("config") or {}).get("seed"), ckpt=args.ckpt, ckpt_git=meta.get("git"),
                ckpt_step=meta.get("step"), git=git_hash(), spec_version=SPEC_VERSION, set=args.set,
                set_sha256=sha256_file(args.set), detector=args.detector, mode=args.mode, head_block=args.head_block)
    table = accuracy_by_tier(rows)
    table_contains = accuracy_by_tier([dict(tiers=r["tiers"], correct=r["contains"]) for r in rows])
    table_strict = accuracy_by_tier([dict(tiers=r["tiers"], correct=r["em_strict"]) for r in rows])
    print(f"rows -> {p}\n=== E6 ({args.arm}) ===\nDECISION RULE: {DECISION_RULES['E6']}")
    print(json.dumps(dict(exact_match_normalised=table, exact_match_strict=table_strict, contains=table_contains,
                          loader=stats, scoring="proxy: IHEval's per-task checkers are not vendored"), indent=1))
    atomic_write_text(out / f"{tag}_table.json", json.dumps(
        dict(rule=DECISION_RULES["E6"], provenance=prov, table=table, exact_match_normalised=table, exact_match_strict=table_strict,
             contains=table_contains, loader=stats,
             scoring="proxy: IHEval's per-task checkers are not vendored (marsea/data/iheval.py)"), indent=1))


if __name__ == "__main__":
    main()
