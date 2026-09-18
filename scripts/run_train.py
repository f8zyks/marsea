"""Train one arm x seed (training procedure).  Example:
  python scripts/run_train.py --arm marsea --seed 0 --L 8192 --total_steps 2500 --detector runs/detector.json \
      --ruler_train data/ruler/TRAIN_* --musique data/musique/musique_ans_v1.0_train.jsonl --hotpot_n 20000
"""
import argparse, glob, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="marsea"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--L", type=int, default=8192); ap.add_argument("--total_steps", type=int, default=3500)
    ap.add_argument("--phase_a_steps", type=int, default=500); ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--detector", default="runs/detector.json"); ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--out", default="runs"); ap.add_argument("--mode", default="dense", choices=["dense", "chunked"])
    ap.add_argument("--ruler_train", nargs="*", default=[]); ap.add_argument("--musique", default=None)
    ap.add_argument("--hotpot_n", type=int, default=0); ap.add_argument("--dry_run_steps", type=int, default=None)
    ap.add_argument("--arm_kwargs", default="{}", help="JSON: E9 knobs, e.g. '{\"tau_i_pinned\": true}'")
    ap.add_argument("--eval_ruler", nargs="*", default=[], help="held-out RULER jsonl files for the 250-step quick eval")
    ap.add_argument("--checkpoint_layers", default="patched", choices=["patched", "all"])
    ap.add_argument("--phase_a_ckpt", default=None, help="fork from this Phase-A checkpoint (default runs/phaseA_seed{seed}.pt under --out)")
    ap.add_argument("--phase_a_only", action="store_true", help="train/verify the Phase-A checkpoint and stop")
    ap.add_argument("--allow_phase_a_layer_change", action="store_true",
                    help="fork a Phase-A checkpoint whose patched layers differ from the detector's (S0 changed them); recorded")
    ap.add_argument("--head_block", type=int, default=None, help="Q-heads processed at a time in the patched layers (memory; identical numbers); 0 = all")
    ap.add_argument("--chunk", type=int, default=1024, help="the chunked path's key-chunk width (MarSeaContext.chunk)")
    ap.add_argument("--log_every", type=int, default=50, help="steps between E8 blocks / log lines")
    ap.add_argument("--eval_every", type=int, default=250); ap.add_argument("--ckpt_every", type=int, default=250)
    ap.add_argument("--phase_a_unpatched", action="store_true",
                    help="run Phase A on the unpatched SDPA model (cheaper; a different kernel from the one Phase B forks "
                         "into -- the delta is measured and written to the run README).  Default: patched SoftmaxNorm.")
    args = ap.parse_args()
    from marsea.train import TrainConfig, train
    from marsea.data.mix import MixedDataset, build_training_sources
    from marsea.data.ruler import load_jsonl, build_example
    from marsea.evaluate import quick_eval_factory
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.backbone)
    ruler_files = [f for pat in args.ruler_train for f in sorted(glob.glob(pat if pat.endswith(".jsonl") else pat + "/validation.jsonl"))]
    sources = build_training_sources(ruler_files, args.musique, args.hotpot_n, args.seed, tok, cache_dir=pathlib.Path(args.out) / "cache")
    assert sources, "no training sources"
    data = MixedDataset(sources, tok, args.L, args.seed, cache_dir=pathlib.Path(args.out) / "cache")
    cfg = TrainConfig(arm=args.arm, seed=args.seed, backbone=args.backbone, L=args.L, total_steps=args.total_steps,
                      phase_a_steps=args.phase_a_steps, accum=args.accum, out_dir=args.out, mode=args.mode,
                      detector_json=args.detector if pathlib.Path(args.detector).exists() else None,
                      dry_run_steps=args.dry_run_steps, arm_kwargs=json.loads(args.arm_kwargs), checkpoint_layers=args.checkpoint_layers,
                      head_block=(args.head_block or None), chunk=args.chunk, log_every=args.log_every, eval_every=args.eval_every, ckpt_every=args.ckpt_every,
                      phase_a_patched=not args.phase_a_unpatched, phase_a_ckpt=args.phase_a_ckpt, phase_a_only=args.phase_a_only,
                      allow_phase_a_layer_change=args.allow_phase_a_layer_change)
    if args.layers: cfg.patched_layers = args.layers; cfg.detector_json = None
    quick = None
    if args.eval_ruler:
        det = json.load(open(args.detector)) if pathlib.Path(args.detector).exists() else None
        layers = det["patched_layers"] if det else cfg.patched_layers
        l_star, h_star = (det["l_star"], det["h_star"]) if det else (layers[0], 0)
        # a QUOTED glob reaches us unexpanded (run_s2.sh passes $EVAL unquoted, a hand-typed S1 command quoted it and died
        # with FileNotFoundError on the literal pattern -- pod 1, 2026-09-18); expand here as --ruler_train already does
        eval_files = [f for pat in args.eval_ruler for f in (sorted(glob.glob(pat)) if glob.has_magic(pat) else [pat])]
        if not eval_files:
            raise FileNotFoundError(f"--eval_ruler {args.eval_ruler} matches no file")
        exs = [build_example(r, tok) for f in eval_files for r in load_jsonl(f)]
        quick = quick_eval_factory(exs, [], tok, l_star, h_star, args.arm)
    train(cfg, data, quick_eval=quick)


if __name__ == "__main__":
    main()
