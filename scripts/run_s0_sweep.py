"""S0 as a whole-stack sweep: every (layer, head), span-tolerant matching, sink excluded (marsea/sweep.py).
  python scripts/run_s0_sweep.py --set "data/ruler/E2_L8192_K8_V1_*_s0,data/ruler/E2_L8192_K8_V4_*_s0" --n 40 --out runs/s0_sweep_base.json
  python scripts/run_s0_sweep.py --ckpt runs/phaseA_seed0.pt ...  --update_detector runs/detector.json
"""
import argparse, glob, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-1.5B"); ap.add_argument("--ckpt", default=None, help="Phase-A checkpoint (LoRA) to load")
    ap.add_argument("--set", required=True); ap.add_argument("--n", type=int, default=40); ap.add_argument("--out", default="runs/s0_sweep.json")
    ap.add_argument("--keep_sink", action="store_true"); ap.add_argument("--update_detector", default=None)
    ap.add_argument("--L_patch", type=int, default=4, help="the fixed size of PATCHED_LAYERS the run is budgeted for")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from marsea.data.ruler import load_jsonl, build_example
    from marsea.sweep import head_sweep, print_summary, USABLE_FRAC
    L_PATCH = args.L_patch
    tok = AutoTokenizer.from_pretrained(args.backbone)
    model = AutoModelForCausalLM.from_pretrained(args.backbone, dtype=torch.bfloat16, attn_implementation="eager").cuda()
    if args.ckpt:
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        from marsea.train import LORA_TARGETS
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, ck["lora"])
        print(f"loaded LoRA from {args.ckpt} (step {ck.get('step')})")
    model.eval()
    files = [f for pat in args.set.split(",") for f in sorted(glob.glob(pat if pat.endswith(".jsonl") else pat + "/validation.jsonl"))]
    exs = []
    per = max(1, args.n // max(1, len(files)))
    for f in files:
        exs += [build_example(r, tok) for r in load_jsonl(f)[:per]]
    det = json.load(open(args.update_detector)) if args.update_detector and pathlib.Path(args.update_detector).exists() else None
    ls, hs = (det["l_star"], det["h_star"]) if det else (None, None)
    res = head_sweep(model, exs, exclude_sink=not args.keep_sink, l_star=ls, h_star=hs)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True); res.save(args.out)
    print_summary(res); print(f"-> {args.out}")
    if det is not None:
        ind = res.frac.get("coref_mentions")
        if ind is not None:
            det["induction_scores"] = ind                                  # the induction score = coreference routing per head
        # the LICENCE is per T_j KIND (the split): the coreference columns need a head routing 'coref_mentions', the value
        # columns one routing 'C-a'; each is judged at (l*, h*) when given, else at its best head (per example, >= 80 %)
        ls_, hs_ = det["l_star"], det["h_star"]
        meas = {}
        for kind, cons in (("coref", "coref_mentions"), ("value", "C-a")):
            if cons not in res.frac: continue
            at_star = res.frac[cons][ls_][hs_]
            f, m, l, h = res.best[cons][0]
            # the licence must name the SITE it was granted at, or `usable_anywhere` licenses a measurement that is
            # then taken somewhere else: evaluate scored every kind at (l*, h*) regardless, so the gate failed OPEN
            # (review d5bd980 E).  run_eval passes this site back in as --coref_site.
            site = ([int(ls_), int(hs_)] if at_star >= USABLE_FRAC else ([int(l), int(h)] if f >= USABLE_FRAC else None))
            meas[kind] = dict(construction=cons, at_l_star_h_star=float(at_star), usable_at_l_star=bool(at_star >= USABLE_FRAC),
                              best=dict(layer=l, head=h, frac=f, mean=m), usable_anywhere=bool(f >= USABLE_FRAC),
                              site=site)
            print(f"{kind:6s} ({cons}): at (l*,h*)=({ls_},{hs_}) frac {at_star:.2f}; best ({l},{h}) frac {f:.2f}")
        det["column_measurable"] = {k: v["usable_at_l_star"] or v["usable_anywhere"] for k, v in meas.items()}
        det["s0_licence"] = meas
        det["s0_replication"] = res.replication_pair
        if not res.best.get("coref_mentions"):
            print("no coref_mentions construction in the sweep: leaving induction_best unset")
            json.dump(det, open(args.update_detector, "w"), indent=1)
            return
        f, m, l, h = res.best["coref_mentions"][0]
        det["induction_best"] = dict(construction="coref_mentions", layer=l, head=h, frac=f, mean=m)
        if f >= USABLE_FRAC and l not in det["patched_layers"]:
            # PATCHED_LAYERS has a fixed cardinality (L_patch = 4): every consumer -- the memory model, run_s2.sh's
            # budget, the E9 layer control -- sizes for it, so the induction layer REPLACES the weakest retrieval
            # layer rather than silently making five (review d5bd980 F).
            det["patched_layers"] = sorted(det["patched_layers"] + [l]); det["induction_layer_added"] = l
            if len(det["patched_layers"]) > L_PATCH:
                rank = {}                                    # each layer's BEST head score (the ranking is descending)
                for sc, ll, hh in det["ranking"]:
                    rank[ll] = max(rank.get(ll, -1.0), sc)
                drop = min((ll for ll in det["patched_layers"] if ll != l), key=lambda ll: rank.get(ll, -1.0))
                det["patched_layers"] = sorted(set(det["patched_layers"]) - {drop})
                det["induction_layer_replaced"] = drop
                print(f"  layer {drop} (weakest retrieval layer) leaves PATCHED_LAYERS to keep |PATCHED_LAYERS| = {L_PATCH}")
            print(f"induction head ({l}, {h}) routes coreference cleanly (frac {f:.2f}): layer {l} joins PATCHED_LAYERS")
        print("column measurable:", det["column_measurable"])
        json.dump(det, open(args.update_detector, "w"), indent=1)


if __name__ == "__main__":
    main()
