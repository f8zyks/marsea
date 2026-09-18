"""Memory profile for the S2 sizing (spec Sec. 6.5; read-through 2026-09-11 F-8).

Separates the two quantities the F-8 discussion conflates:
  PEAK      what the dense normaliser needs while it runs -- ~10 fp32 [1, H, T, T] tensors per patched layer, unaffected
            by head sub-selection (every head must be computed) and the number that decides whether a step fits at all;
  RETAINED  what the diagnostics hold after the call returns -- what head sub-selection and the field filter cut, and
            what accumulates when several passes are alive (B5's two-pass, the paired-E cache).
Both scale as T^2 in the patched layers, so two measured lengths fit the coefficient and extrapolate to 8K / 16K.

  python scripts/profile_memory.py --lengths 512 1024 --layers 2 --mode teacher
"""
import argparse, gc, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch


def _mem():
    return torch.cuda.max_memory_allocated() / 2**30, torch.cuda.memory_allocated() / 2**30


def _tensor_bytes(o):
    """bytes held by a tensor, or by every tensor inside a dict / list / tuple (extra["sparse"] is a DICT of tensors:
    the chunked path's whole sparse store lives there)."""
    if torch.is_tensor(o):
        return o.element_size() * o.nelement()
    if isinstance(o, dict):
        return sum(_tensor_bytes(v) for v in o.values())
    if isinstance(o, (list, tuple)):
        return sum(_tensor_bytes(v) for v in o)
    return 0


def _key(mode, sub):
    return f"{mode}_b5" if sub == "b5" else f"{mode}_head" if sub else mode


def gate_verdict(points: dict, extrapolation: dict, targets, gate_GB: float, gate_keys=None) -> dict:
    """The gate, over the GATED keys only: an OOM in a gated key fails it whatever the numbers say; otherwise the worst
    measured or extrapolated peak (at `targets`, and the 4-layer column when present) must be <= gate_GB.

    Keys nothing in the queue runs must not be gated: preflight's chunked evaluation probe gated `teacher` -- every
    head of l* kept, which only B5's pass 1 does, and B5 runs dense at 8K -- at 16K, where the H200 ran out of memory
    on a pass no job incurs while `teacher_head`, the pass the whole 16K queue runs, measured 54 GB (pod 1, 2026-09-18).
    A gated key that has no measurement at all is a failure too (the probe never ran it)."""
    gated = (lambda k: True) if gate_keys is None else (lambda k: k in gate_keys)
    worst, why, oom = 0.0, None, None
    seen = set()
    for T, entry in points.items():
        for k, v in entry.items():
            if not gated(k):
                continue
            seen.add(k)
            if v.get("oom"):
                oom = oom or f"OOM at T = {T} ({k})"
            if v.get("peak_GB", 0) > worst:
                worst, why = v["peak_GB"], f"measured {k} at T = {T}"
    for key, ex in extrapolation.items():
        if not gated(key):
            continue
        for t in targets:
            for kk in (str(t), f"{t}_at_4_layers"):
                if ex.get(kk) and ex[kk] > worst:
                    worst, why = ex[kk], f"extrapolated {key} at {kk}"
    missing = sorted(set(gate_keys or []) - seen)
    if missing and oom is None:
        oom = f"gated key(s) never measured: {missing}"
    passed = bool(oom is None and float(worst) <= gate_GB)          # a numpy bool crashed the JSON write
    return dict(passed=passed, gate_GB=gate_GB, worst_GB=round(float(worst), 2), why=oom or why,
                keys=list(gate_keys) if gate_keys is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--lengths", type=int, nargs="*", default=[512, 1024])
    ap.add_argument("--layers", type=int, nargs="+", default=[2],
                    help="patched-layer COUNTS to profile.  Two or more are differenced to get the MARGINAL cost of a "
                         "patched layer, which is the only honest way to rescale to a different layer count.")
    ap.add_argument("--head_block", type=int, default=None,
                    help="Q-heads per block (run_s2.sh's lever).  Unprofiled until now: cfg.head_block stayed None.")
    ap.add_argument("--sites", type=int, default=1,
                    help="how many measurement heads the teacher-forced profile keeps (D-9a scores the coreference "
                         "column at a second (layer, head), which doubles the retained dense tensors)")
    ap.add_argument("--gate_GB", type=float, default=None,
                    help="fail the process if any measured or extrapolated peak exceeds this (preflight's real gate)")
    ap.add_argument("--targets", type=int, nargs="*", default=[8192, 16384])
    ap.add_argument("--paired", action="store_true",
                    help="teacher mode: keep a SECOND model (a MarSea arm) resident and run its relation pass before the "
                         "measured one, as run_eval's PairedRelation does for B0/B1/B2/B4 -- two backbones, two LoRA stacks "
                         "and the paired pass's diagnostics on top of the arm's (review e16a843 F-4)")
    ap.add_argument("--gate_keys", nargs="*", default=None,
                    help="gate only these measurements (e.g. teacher_head: what the eval queue runs, not the all-heads "
                         "teacher pass nothing but B5's field-filtered trace incurs); default: all")
    ap.add_argument("--b5", action="store_true",
                    help="teacher mode: also measure B5's pass 1 (ctx.keep_dense on every patched layer, fields E and "
                         "supp_rel, every head) as key teacher_b5 -- the one all-heads pass the eval queue runs (dense, 8K)")
    ap.add_argument("--modes", nargs="*", default=["teacher", "train"])
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--out", default="runs/memory_profile.json")
    ap.add_argument("--arm", default="marsea"); ap.add_argument("--mode_impl", default="dense", choices=["dense", "chunked"])
    ap.add_argument("--checkpoint_layers", default="patched", choices=["patched", "all"])
    ap.add_argument("--rho0", type=float, default=0.05, help="relation coverage to calibrate to before profiling")
    args = ap.parse_args()
    # through the REAL training path: LoRA-only gradients, gradient checkpointing on the patched layers, logits_to_keep.
    # (Profiling a bare from_pretrained model instead makes every base weight trainable and checkpoints nothing, which
    # overstates the training peak by an order of magnitude.)
    from marsea.train import TrainConfig, build_model, train_step, make_optimizer
    from marsea.evaluate import teacher_forced_pass
    from transformers import AutoConfig
    cfgm = AutoConfig.from_pretrained(args.backbone)
    L = cfgm.num_hidden_layers; H = cfgm.num_attention_heads
    d = getattr(cfgm, "head_dim", cfgm.hidden_size // H)
    layer_counts = sorted(set(args.layers))
    def _layers_for(n):
        return sorted({int(round(x)) for x in torch.linspace(L // 3, L - 4, n).tolist()})
    layers = _layers_for(layer_counts[-1])
    tcfg = TrainConfig(arm=args.arm, backbone=args.backbone, patched_layers=layers, mode=args.mode_impl,
                       checkpoint_layers=args.checkpoint_layers, accum=1, log_every=10 ** 9, phase_a_steps=0,
                       detector_json=None, head_block=args.head_block)
    model, tok, ctx, layers = build_model(tcfg, phase_a=False)
    ctx.chunk = args.chunk                                   # the chunked path's key-chunk width
    # the relation at its operating point: the chunked path's footprint is proportional to coverage
    from marsea.train import warmup_calibrate
    b0s = warmup_calibrate(model, ctx, min(1024, max(args.lengths)), rho0=args.rho0)
    print(f"calibrated to rho_0 = {args.rho0}; b0 per patched layer: {b0s}")
    opt = make_optimizer(model, tcfg, with_modules=True)
    paired = None
    if args.paired:
        pcfg = TrainConfig(arm="marsea", backbone=args.backbone, patched_layers=layers, mode=args.mode_impl,
                           checkpoint_layers=args.checkpoint_layers, accum=1, log_every=10 ** 9, phase_a_steps=0,
                           detector_json=None, head_block=args.head_block)
        try:
            pm, _, pctx, _ = build_model(pcfg, phase_a=False); pctx.chunk = args.chunk
            warmup_calibrate(pm, pctx, min(1024, max(args.lengths)), rho0=args.rho0)
        except torch.OutOfMemoryError as e:
            # the characteristic failure of a two-model probe is running out of memory while the SECOND model is built,
            # which happened outside every try: a traceback, no JSON, and preflight (which tolerates only exit 3 here)
            # aborted the cap proof, the step time and the detector behind it (review 9bacef9 E).  It is a gate failure.
            res = dict(backbone=args.backbone, paired=True, impl=args.mode_impl, head_block=args.head_block, points={},
                       gate=dict(passed=False, gate_GB=args.gate_GB, worst_GB=None, why=f"OOM building the second (paired) model: {e}",
                                 keys=args.gate_keys))
            pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(args.out).write_text(json.dumps(res, indent=1))
            print(f"GATE FAILED: {res['gate']['why']}\n-> {args.out}"); sys.exit(3)
        pm.eval(); paired = (pm, pctx)
    base = torch.cuda.memory_allocated() / 2**30
    res = dict(backbone=args.backbone, heads=H, head_dim=d, patched_layers=layers, weights_GB=round(base, 2),
               rho0=args.rho0, impl=args.mode_impl, chunk=args.chunk, checkpoint=args.checkpoint_layers,
               head_block=args.head_block, sites=args.sites, paired=bool(args.paired), layer_counts=layer_counts,
               targets=list(args.targets),
               points={}, points_by_layers={})
    for T in args.lengths:
        ids = torch.randint(0, 1000, (1, T)).cuda()
        entry = {}
        for mode in args.modes:
            # teacher (all heads at l*, every field), teacher_head (the measurement sites only: what every queue job
            # runs) and, with --b5, teacher_b5: B5's pass 1 -- every head of EVERY patched layer kept, fields E and
            # supp_rel only, the one all-heads pass the eval queue incurs (dense, 8K, E2)
            subs = ([False, True] + (["b5"] if args.b5 else [])) if mode == "teacher" else [False]
            for sub in subs:
                gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                try:
                    if mode == "teacher" and sub == "b5":
                        model.eval(); ctx.keep_dense = True
                        try:
                            _, _, diags = teacher_forced_pass(model, ctx, ids[0, :-8].tolist(), ids[0, -8:].tolist(),
                                                              layers[-1], fields=("E", "supp_rel"))
                        finally:
                            ctx.keep_dense = False
                        peak, _ = _mem()
                        retained = sum(_tensor_bytes(v) for dd in diags.values()
                                       for v in list(dd.__dict__.values()) + [dd.extra]) / 2**30
                        del diags
                    elif mode == "teacher":
                        model.eval()
                        hs = list(range(3, 3 + max(1, args.sites))) if sub else None
                        sites_arg = None if hs is None else [(layers[-1], h) for h in hs]
                        pe = None
                        if paired is not None and sub:
                            # evaluate_ruler's order: the paired relation first, held on device, then the arm's pass
                            _, _, dgp = teacher_forced_pass(paired[0], paired[1], ids[0, :-8].tolist(), ids[0, -8:].tolist(),
                                                            layers[-1], h_star=hs[0], fields=("E",), sites=sites_arg)
                            from marsea.evaluate import _kept_head
                            pe = {(l, h): dgp[l].E[0, _kept_head(dgp[l], h)] for l, h in sites_arg}
                            del dgp
                        _, _, diags = teacher_forced_pass(model, ctx, ids[0, :-8].tolist(), ids[0, -8:].tolist(),
                                                          layers[-1], h_star=(hs[0] if hs else None), sites=sites_arg)
                        del pe
                        peak, _ = _mem()
                        # what the diagnostics themselves hold (F-8): the quantity head sub-selection and the field
                        # filter cut, and the one that accumulates when several passes are alive at once
                        retained = sum(_tensor_bytes(v) for dd in diags.values()
                                       for v in list(dd.__dict__.values()) + [dd.extra]) / 2**30
                        del diags
                    else:
                        model.train(); ctx.collect = False
                        class _Seq:                                   # one accumulation micro-batch, answer tokens at the end
                            input_ids = ids
                            labels = torch.cat([torch.full((1, T - 8), -100), ids[:, -8:].cpu()], 1)
                            n_label_tokens = 8
                        class _Data:
                            def get(self, c): return _Seq()
                        train_step(model, ctx, opt, _Data(), 0, 0, tcfg, lambda r: None)
                        peak, _ = _mem(); retained = 0.0
                        opt.zero_grad(set_to_none=True)
                    entry[_key(mode, sub)] = dict(peak_GB=round(peak, 3), retained_GB=round(max(0.0, retained), 3))
                except torch.OutOfMemoryError:
                    entry[_key(mode, sub)] = dict(oom=True)
                gc.collect(); torch.cuda.empty_cache()
        res["points"][T] = entry
        print(f"T = {T}: {json.dumps(entry)}")
    # fit (peak - weights) = a * T^p by least squares in log space and report the EXPONENT: the dense path is quadratic
    # in T (it materialises [H, T, T] per patched layer), the chunked path is designed to be linear (T x chunk), and a
    # two-point quadratic fit would extrapolate the chunked path as if it were dense.
    res["extrapolation"] = {}
    Ts = sorted(res["points"])
    for key in sorted({k for T in Ts for k in res["points"][T]}):
        pts = [(T, res["points"][T][key]["peak_GB"] - res["weights_GB"]) for T in Ts
               if "peak_GB" in res["points"][T].get(key, {}) and res["points"][T][key]["peak_GB"] > res["weights_GB"]]
        if len(pts) < 2:
            continue
        xs = np.log(np.array([t for t, _ in pts], dtype=float)); ys = np.log(np.array([v for _, v in pts], dtype=float))
        pexp, loga = np.polyfit(xs, ys, 1)
        a = float(np.exp(loga))
        fit = {str(t): round(res["weights_GB"] + a * t ** pexp, 1) for t in args.targets}
        fit["exponent_p"] = round(float(pexp), 2)
        fit["activation_GB_at_8K"] = round(a * 8192 ** pexp, 1)
        # NOT activation / n_layers: that charges the backbone's own activations to the patched layers.  The honest
        # per-layer number comes from differencing layer counts (marginal_per_patched_layer_GB, when >= 2 were run).
        fit["activation_over_layer_count_GB_at_8K"] = round(a * 8192 ** pexp / max(1, len(layers)), 1)
        fit["n_points"] = len(pts)
        # with two points the fit is exact and R2 is identically 1.0 BY CONSTRUCTION: reporting it as if it were
        # evidence of a good fit is the opposite of what it means (review d5bd980 F)
        fit["r2"] = (round(float(1 - ((ys - (pexp * xs + loga)) ** 2).sum() / max(1e-12, ((ys - ys.mean()) ** 2).sum())), 4)
                     if len(pts) > 2 else None)
        res["extrapolation"][key] = fit
    # the MARGINAL cost of one patched layer, by differencing layer counts at the largest profiled length.  Dividing
    # the whole activation term by the layer count (what this used to print) charges the backbone's own activations to
    # the patched layers, and gradient checkpointing means the marginal is nearly flat anyway (review d5bd980 F).
    if len(layer_counts) > 1:
        Tmax = max(args.lengths)
        for n in layer_counts[:-1]:
            gc.collect(); torch.cuda.empty_cache()
            del model, opt
            gc.collect(); torch.cuda.empty_cache()
            tcfg_n = TrainConfig(arm=args.arm, backbone=args.backbone, patched_layers=_layers_for(n), mode=args.mode_impl,
                                 checkpoint_layers=args.checkpoint_layers, accum=1, log_every=10 ** 9, phase_a_steps=0,
                                 detector_json=None, head_block=args.head_block)
            model, tok, ctx, _l = build_model(tcfg_n, phase_a=False); ctx.chunk = args.chunk
            warmup_calibrate(model, ctx, min(1024, Tmax), rho0=args.rho0)
            opt = make_optimizer(model, tcfg_n, with_modules=True)
            ids = torch.randint(0, 1000, (1, Tmax)).cuda()
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            model.train(); ctx.collect = False
            class _SeqN:
                input_ids = ids
                labels = torch.cat([torch.full((1, Tmax - 8), -100), ids[:, -8:].cpu()], 1)
                n_label_tokens = 8
            class _DataN:
                def get(self, c): return _SeqN()
            try:
                train_step(model, ctx, opt, _DataN(), 0, 0, tcfg_n, lambda r: None)
                res["points_by_layers"][n] = dict(T=Tmax, peak_GB=round(_mem()[0], 3))
            except torch.OutOfMemoryError:
                res["points_by_layers"][n] = dict(T=Tmax, oom=True)
            opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
        top = res["points"].get(Tmax, {}).get("train", {}).get("peak_GB")
        pts = [(n, v["peak_GB"]) for n, v in res["points_by_layers"].items() if "peak_GB" in v]
        if top is not None and pts:
            pts.append((layer_counts[-1], top))
            pts.sort()
            marg = (pts[-1][1] - pts[0][1]) / max(1, pts[-1][0] - pts[0][0])
            res["marginal_per_patched_layer_GB"] = dict(at_T=Tmax, GB=round(marg, 3), points=pts)
            print(f"\nmarginal cost of ONE patched layer at T = {Tmax}: {marg:.3f} GB  (points {pts})")
    marg_pre = (res.get("marginal_per_patched_layer_GB") or {}).get("GB")
    if marg_pre is not None:
        for key, ex in res["extrapolation"].items():
            for t in args.targets:
                base_t = ex.get(str(t))
                if base_t is not None:
                    ex[f"{t}_at_4_layers"] = round(base_t + (4 - len(layers)) * marg_pre * (t / max(args.lengths)) ** 2, 1)
    print("\nextrapolated PEAK (GB), fitted as weights + a*T^p with p FREE (printed per row), on",
          len(layers), "patched layers:")
    print(json.dumps(res["extrapolation"], indent=1))
    marg = marg_pre
    print(f"\narm={args.arm} impl={args.mode_impl} checkpoint={args.checkpoint_layers} head_block={args.head_block}; "
          f"measured on {len(layers)} patched layers"
          + (f"; marginal per patched layer {marg:.2f} GB, so the 4-layer column is carried through" if marg is not None
             else "; only ONE layer count was profiled, so no 4-layer rescaling is printed (pass --layers 2 4)") + ":")
    for key, ex in res["extrapolation"].items():
        # rescaling to a different layer count needs the MARGINAL per-layer term, not the whole activation term (the
        # backbone's activations do not scale with the patched-layer count), so the 4-layer column appears only when
        # two layer counts were actually measured and differenced.
        r2 = ex["r2"]
        tgt = "  |  ".join(f"{t}: {ex.get(str(t), float('nan')):.0f} GB"
                           + (f" ({ex[f'{t}_at_4_layers']:.0f} at 4 layers)" if f"{t}_at_4_layers" in ex else "")
                           for t in args.targets)
        print(f"  {key:14s} p = {ex['exponent_p']} (R2 {r2 if r2 is not None else 'n/a: 2 points, exact by construction'})"
              f"  |  {tgt}   (as measured, at {len(layers)} patched layers)")
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(res, indent=1))
    print("->", args.out)
    # the gate.  profile_memory used to swallow OutOfMemoryError and exit 0, so preflight's "peak < 72 GB" line
    # compared nothing at all (review d5bd980 F).
    if args.gate_GB is not None:
        # the verdict is written INTO the profile, so run_s2.sh can refuse a configuration preflight did not pass
        # (review e982f83 B-4) instead of trusting a comment
        g = res["gate"] = gate_verdict(res["points"], res["extrapolation"], args.targets, args.gate_GB, args.gate_keys)
        pathlib.Path(args.out).write_text(json.dumps(res, indent=1))
        hard = g["why"] if g["why"] and not g["why"].startswith(("measured", "extrapolated")) else None
        if hard:                                                          # an OOM, or a gated key the probe never ran
            print(f"GATE FAILED: {hard}"); sys.exit(3)
        print(f"GATE: worst peak {g['worst_GB']:.1f} GB ({g['why']}) against {args.gate_GB:.0f} GB")
        if not g["passed"]:
            print("GATE FAILED"); sys.exit(3)
        print("GATE PASSED")


if __name__ == "__main__":
    main()
