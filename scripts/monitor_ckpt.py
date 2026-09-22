"""The relation monitor on saved checkpoints:  python scripts/monitor_ckpt.py --ckpt runs/x/step900.pt [more ...] --out m.json
Reads marsea/monitor.py's per-row records at the detector's (l*, h*) on the first --n examples of each --set file."""
import argparse, glob, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1])); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True); ap.add_argument("--arm", default="marsea")
    ap.add_argument("--set", default="data/ruler/QUICK_L4096_*/validation.jsonl"); ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--detector", default="runs/detector.json"); ap.add_argument("--backbone", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from run_eval import load_arm
    from marsea.data.ruler import load_set, build_example
    from marsea.monitor import monitor_example, summarise, format_line, gold_by_head
    det = json.load(open(args.detector)); layers = det["patched_layers"]
    res = {}
    for ck in args.ckpt:
        model, tok, ctx = load_arm(ck, args.arm, args.backbone, layers, {}, "dense", head_block=2, det=det)
        recs = []; exs = [build_example(r, tok) for f in sorted(glob.glob(args.set)) for r in load_set(f)[:args.n]]
        for ex in exs:
            recs += monitor_example(model, ctx, ex, det["l_star"], det["h_star"])
        res[ck] = summarise(recs)
        res[ck]["by_head"] = gold_by_head(model, ctx, exs, layers)             # where the gold tokens are, for every head
        from marsea.blockgold import block_metrics
        res[ck]["blocks"] = block_metrics(model, ctx, exs, layers)             # the partition view: own / other / stale / rest
        print(ck); print(format_line("-", res[ck]), flush=True)
        del model; torch.cuda.empty_cache()
        if args.out:
            json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
