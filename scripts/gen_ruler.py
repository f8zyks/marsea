"""Generate the RULER grids of spec Sec. 13.1 (E2 / E3 / E3 depth / E4 / training pool / detector) with the backbone tokenizer."""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from marsea.data import ruler as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--grid", choices=["E2", "E3", "E3depth", "E4", "train", "detector", "eval_quick", "train_mq", "quick_mq", "train_vt", "quick_vt"], required=True)
    ap.add_argument("--out", default="data/ruler")
    ap.add_argument("--n", type=int, default=None, help="samples per config (default: grid default)")
    ap.add_argument("--L", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--jobs", type=int, default=1, help="configs generated at once (each is one single-threaded niah.py process; CPU-bound)")
    args = ap.parse_args()
    kw = {}
    if args.n: kw["n_samples"] = args.n
    if args.L: kw["L"] = args.L
    if args.seeds is not None: kw["seeds"] = tuple(args.seeds)
    grids = dict(E2=R.grid_E2, E3=R.grid_E3, E3depth=R.grid_E3_depth, E4=R.grid_E4, train=R.grid_training, detector=R.grid_detector,
                 train_mq=R.grid_training_mq)
    if args.grid == "eval_quick":       # training Sec. 8.2: 200 examples spanning the E2/E3 grid at 8K
        Lq = args.L or 8192
        sq = args.seeds[0] if args.seeds else R.QUICK_SEED       # NOT an eval seed: seed 0 is E2's own test set
        cfgs = [c for c in (R.NIAHConfig("QUICK", Lq, K, V, 1, seed=sq, num_samples=args.n or 20)
                            for K in (8, 32) for V in (1, 4, 16) if K * V <= 128) if R.feasible(c)]
    elif args.grid == "train_vt":
        cfgs = R.grid_training_vt(L=args.L or 4096, seeds=tuple(args.seeds) if args.seeds else R.TRAIN_SEEDS, n_samples=args.n or 400)
    elif args.grid == "quick_vt":
        cfgs = R.grid_quick_vt(L=args.L or 4096, seed=(args.seeds[0] if args.seeds else R.QUICK_SEED), n_samples=args.n or 25)
    elif args.grid == "quick_mq":
        cfgs = R.grid_quick_mq(L=args.L or 4096, seed=(args.seeds[0] if args.seeds else R.QUICK_SEED), n_samples=args.n or 25)
    else:
        fn = grids[args.grid]
        cfgs = fn(**{k: v for k, v in kw.items() if k in fn.__code__.co_varnames})
    # per-CONFIG, not per-grid: gen_data.sh's continue-on-failure was grid-level, so one bad cell of `--grid train`
    # still lost the other 68 (review 30372ae E).  The script still exits non-zero.
    # --jobs N: N niah.py processes at once.  generate() is a subprocess call plus a length check, so threads are enough;
    # ensure_patched() is done once up front rather than raced by the workers.
    failed = []
    if args.jobs > 1 and any(not isinstance(c, R.VTConfig) and not (pathlib.Path(args.out) / c.save_name() / "validation.jsonl").exists() for c in cfgs):
        R.ensure_patched()
    if any(isinstance(c, R.VTConfig) for c in cfgs):
        R.ensure_vt_patched()
    from concurrent.futures import ThreadPoolExecutor
    def one(cfg):
        try:
            gen = R.generate_vt if isinstance(cfg, R.VTConfig) else R.generate
            return cfg, gen(cfg, args.backbone, pathlib.Path(args.out)), None
        except Exception as e:
            return cfg, None, repr(e)
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for cfg, p, err in ex.map(one, cfgs):
            if err is None:
                print("ok", p, flush=True)
            else:
                failed.append((cfg.save_name(), err)); print(f"FAILED {cfg.save_name()}: {err}", file=sys.stderr, flush=True)
    if failed:
        print(f"=== {len(failed)} of {len(cfgs)} configs failed in grid {args.grid}", file=sys.stderr)
        for name, err in failed:
            print(f"  {name}: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
