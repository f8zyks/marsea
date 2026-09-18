"""Measure the optimiser-step time at the real sequence length (spec Sec. 18 / record Sec. 44: the pre-committed rule is
> 1.2 s per sequence at 8K => drop to two patched layers or train at 4K).  Reports seconds per sequence and the implied
Phase-B wall clock for the committed step count."""
import argparse, json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-1.5B"); ap.add_argument("--L", type=int, default=8192)
    ap.add_argument("--layers", type=int, default=4); ap.add_argument("--mode", default="chunked", choices=["dense", "chunked"])
    ap.add_argument("--chunk", type=int, default=1024); ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--steps", type=int, default=5); ap.add_argument("--phase_b_steps", type=int, default=2000)
    ap.add_argument("--arm", default="marsea"); ap.add_argument("--out", default="runs/step_time.json")
    ap.add_argument("--rho0", type=float, default=0.05)
    ap.add_argument("--no_gate", action="store_true", help="report the D-23 rule without failing the process")
    ap.add_argument("--head_block", type=int, default=2,
                    help="Q-heads per block, as run_s2.sh runs it.  D-23's 1.2 s/sequence rule is now a hard failure, "
                         "so it must be measured on the forward the queue ACTUALLY runs: head blocking serialises the "
                         "head groups and moves the number (review 30372ae D).  0 disables it.")
    args = ap.parse_args()
    from marsea.train import TrainConfig, build_model, make_optimizer, train_step, warmup_calibrate
    from transformers import AutoConfig
    cfgm = AutoConfig.from_pretrained(args.backbone); L = cfgm.num_hidden_layers
    layers = sorted({int(round(x)) for x in torch.linspace(L // 3, L - 4, args.layers).tolist()})
    cfg = TrainConfig(arm=args.arm, backbone=args.backbone, patched_layers=layers, mode=args.mode, accum=args.accum,
                      log_every=10 ** 9, phase_a_steps=0, detector_json=None,
                      head_block=(args.head_block or None))
    model, tok, ctx, layers = build_model(cfg, phase_a=False); ctx.chunk = args.chunk
    warmup_calibrate(model, ctx, min(1024, args.L), rho0=args.rho0)   # the operating point a run has
    opt = make_optimizer(model, cfg, with_modules=True)
    ids = torch.randint(0, 1000, (1, args.L)).cuda()

    class _Seq:
        input_ids = ids
        labels = torch.cat([torch.full((1, args.L - 16), -100), ids[:, -16:].cpu()], 1)
        n_label_tokens = 16

    class _Data:
        def get(self, c): return _Seq()

    model.train()
    ts = []
    for i in range(args.steps + 1):
        torch.cuda.synchronize(); t0 = time.time()
        train_step(model, ctx, opt, _Data(), 0, i, cfg, lambda r: None)
        torch.cuda.synchronize()
        if i:                                                        # skip the warm-up step
            ts.append(time.time() - t0)
    per_step = sum(ts) / len(ts); per_seq = per_step / args.accum
    res = dict(L=args.L, patched_layers=layers, mode=args.mode, chunk=args.chunk, accum=args.accum,
               head_block=args.head_block or None,
               seconds_per_step=round(per_step, 2), seconds_per_sequence=round(per_seq, 3),
               peak_GB=round(torch.cuda.max_memory_allocated() / 2**30, 1),
               phase_b_hours=round(per_step * args.phase_b_steps / 3600, 2),
               rule="spec Sec. 18: > 1.2 s/sequence at 8K => two patched layers or 4K training",
               rule_triggered=bool(per_seq > 1.2 and args.L >= 8192))
    print(json.dumps(res, indent=1))
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(res, indent=1))
    if res["rule_triggered"]:
        # the rule used to be PRINTED and written to a file nothing reads, and the process exited 0 -- so the queue
        # behind it started anyway (review d5bd980 F).  It now fails, and --no_gate is the explicit opt-out.
        print("PRE-COMMITTED RULE TRIGGERED: drop to two patched layers or train at 4K (record Sec. 44).")
        if not args.no_gate:
            sys.exit(2)


if __name__ == "__main__":
    main()
