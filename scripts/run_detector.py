"""Sec. 6.7: generate the 200-prompt S-NIAH detector set (4K, words, seed 0), run the retrieval-head detector on the
UNPATCHED eager backbone, write runs/detector.json with per-(layer, head) scores, PATCHED_LAYERS and (l*, h*)."""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--L", type=int, default=4096)
    ap.add_argument("--L_patch", type=int, default=4)
    ap.add_argument("--out", default="runs/detector.json")
    ap.add_argument("--data_dir", default="data/ruler")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from marsea.data.ruler import NIAHConfig, generate, load_jsonl, build_example
    from marsea.detector import run_detector
    tok = AutoTokenizer.from_pretrained(args.backbone)
    cfg = NIAHConfig("DET", args.L, 1, 1, 1, type_needle_v="words", seed=0, num_samples=args.n, gold_depth=None)
    path = generate(cfg, args.backbone, pathlib.Path(args.data_dir))
    exs = []
    for rec in load_jsonl(path):
        ex = build_example(rec, tok)
        exs.append(dict(prompt_ids=ex.prompt_ids, gold_ids=ex.gold_ids,
                        needle_token_positions=[list(range(a, b)) for (a, b) in ex.value_spans]))
    model = AutoModelForCausalLM.from_pretrained(args.backbone, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    res = run_detector(model, tok, exs, L_patch=args.L_patch).stamp(args.backbone)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.save(args.out)
    print(f"stamped: spec {res.spec_version} git {res.git} at {res.created}")
    print(f"PATCHED_LAYERS = {res.patched_layers}; (l*, h*) = ({res.l_star}, {res.h_star}); prompts used {res.n_prompts}")
    print("top heads:", [(round(s, 3), l, h) for s, l, h in res.ranking[:12]])
    n_ret = sum(1 for s, _, _ in res.ranking if s > res.threshold)
    print(f"heads with score > {res.threshold}: {n_ret} of {len(res.scores) * len(res.scores[0])} "
          f"({100 * n_ret / (len(res.scores) * len(res.scores[0])):.1f} %; the paper expects < 5 %)")


if __name__ == "__main__":
    main()
