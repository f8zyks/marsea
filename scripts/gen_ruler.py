"""Generate the RULER grids of spec Sec. 13.1 (E2 / E3 / E3 depth / E4 / training pool / detector) with the backbone tokenizer."""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from marsea.data import ruler as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--grid", choices=["E2", "E3", "E3depth", "E4", "train", "detector", "eval_quick"], required=True)
    ap.add_argument("--out", default="data/ruler")
    ap.add_argument("--n", type=int, default=None, help="samples per config (default: grid default)")
    ap.add_argument("--L", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()
    kw = {}
    if args.n: kw["n_samples"] = args.n
    if args.L: kw["L"] = args.L
    if args.seeds is not None: kw["seeds"] = tuple(args.seeds)
    grids = dict(E2=R.grid_E2, E3=R.grid_E3, E3depth=R.grid_E3_depth, E4=R.grid_E4, train=R.grid_training, detector=R.grid_detector)
    if args.grid == "eval_quick":       # training Sec. 8.2: 200 examples spanning the E2/E3 grid at 8K
        Lq = args.L or 8192
        sq = args.seeds[0] if args.seeds else R.QUICK_SEED       # NOT an eval seed: seed 0 is E2's own test set
        cfgs = [c for c in (R.NIAHConfig("QUICK", Lq, K, V, 1, seed=sq, num_samples=args.n or 20)
                            for K in (8, 32) for V in (1, 4, 16) if K * V <= 128) if R.feasible(c)]
    else:
        fn = grids[args.grid]
        cfgs = fn(**{k: v for k, v in kw.items() if k in fn.__code__.co_varnames})
    # per-CONFIG, not per-grid: gen_data.sh's continue-on-failure was grid-level, so one bad cell of `--grid train`
    # still lost the other 68 (review 30372ae E).  The script still exits non-zero.
    failed = []
    for cfg in cfgs:
        try:
            p = R.generate(cfg, args.backbone, pathlib.Path(args.out))
            print("ok", p)
        except Exception as e:
            failed.append((cfg.save_name(), repr(e)))
            print(f"FAILED {cfg.save_name()}: {e}", file=sys.stderr)
    if failed:
        print(f"=== {len(failed)} of {len(cfgs)} configs failed in grid {args.grid}", file=sys.stderr)
        for name, err in failed:
            print(f"  {name}: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
