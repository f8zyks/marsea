"""Apply the --gold_depth patch to RULER's niah.py (spec Sec. 13.1 [v4.1], round-2 C-2) by exact string
replacement, and regenerate third_party/ruler_gold_depth.patch from the result.  Idempotent."""
import pathlib, subprocess, sys, shutil
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "third_party/RULER/scripts/data/synthetic/niah.py"
ORIG = SRC.with_suffix(".py.orig")
PATCH = ROOT / "third_party/ruler_gold_depth.patch"

MARKER = "## MarSea-patch-v3"        # bumped whenever a hunk is added: ruler.ensure_patched() re-applies on a miss
# v3 (2026-09-16): the haystack-size rate estimate.  v2's sets were ~29 % short of the requested length
# because RULER's tokens-per-haystack probe charges the needles' fixed cost to the per-word rate, so the
# shortfall grew with the needle count and would have confounded E3's n-sweep with sequence length.


def apply(text: str) -> str:
    if MARKER in text:
        return text
    a = 'parser.add_argument("--model_template_token", type=int, default=0, help=\'used for nemo skills, minus num of model template token\')\n'
    assert a in text
    text = text.replace(a, a + 'parser.add_argument("--gold_depth", type=float, default=None,\n'
        '                    help=\'MarSea position control (spec Sec. 13.1 [v4.1]): place the QUERIED key\\\'s needles at this relative depth in [0,1]; distractor needles keep RULER\\\'s random depths.\')\n')
    a = "    keys, values, needles = [], [], []\n    for _ in range(args.num_needle_k):\n"
    assert a in text
    text = text.replace(a, "    keys, values, needles = [], [], []\n    needle_key_of = []\n    for _ in range(args.num_needle_k):\n")
    a = "                value=value[-1],\n            ))\n        values.append(value)\n\n    random.Random(args.random_seed).shuffle(needles)\n"
    assert a in text
    text = text.replace(a, "                value=value[-1],\n            ))\n            needle_key_of.append(len(keys) - 1)\n        values.append(value)\n\n"
        "    perm = list(range(len(needles)))\n    random.Random(args.random_seed).shuffle(perm)\n"
        "    needles = [needles[p] for p in perm]\n    needle_key_of = [needle_key_of[p] for p in perm]\n\n"
        "    ## MarSea: the query is decided BEFORE placement so the gold needles can be positioned (--gold_depth)\n"
        "    indices = random.sample(range(args.num_needle_k), args.num_needle_q)\n")
    a = ("        insertion_positions = [0] + \\\n"
         "                              sorted([int(len(document_sents) * (depth / 100)) for depth in random.sample(DEPTHS, len(needles))]) + \\\n"
         "                              [len(document_sents)]\n")
    assert a in text, "depth-sampling lines not found"
    text = text.replace(a,
        "        # MarSea: stock RULER samples depths WITHOUT replacement from 40 values and cannot place > 40 needles (E2 at m >= 8,\n"
        "        # E3 at n >= 16 with V = 4); with more needles the depths are drawn with replacement (ties insert consecutively)\n"
        "        depths = random.sample(DEPTHS, len(needles)) if len(needles) <= len(DEPTHS) else random.choices(DEPTHS, k=len(needles))\n"
        "        if args.gold_depth is not None:\n"
        "            # MarSea: the queried key's needles go to the requested depth (consecutive), distractors stay random\n"
        "            gold_pos = int(len(document_sents) * args.gold_depth)\n"
        "            gold_needles = [needles[n] for n in range(len(needles)) if needle_key_of[n] in indices]\n"
        "            other = [(int(len(document_sents) * (depths[n] / 100)), needles[n]) for n in range(len(needles)) if needle_key_of[n] not in indices]\n"
        "            placed = sorted(other + [(gold_pos, nd) for nd in gold_needles], key=lambda x: x[0])\n"
        "            needles = [nd for _, nd in placed]\n"
        "            insertion_positions = [0] + [p for p, _ in placed] + [len(document_sents)]\n"
        "        else:\n"
        "            insertion_positions = [0] + \\\n"
        "                              sorted([int(len(document_sents) * (depth / 100)) for depth in depths]) + \\\n"
        "                              [len(document_sents)]\n")
    a = "    ## Query and Answer\n    indices = random.sample(range(args.num_needle_k), args.num_needle_q)\n"
    assert a in text
    text = text.replace(a, "    ## Query and Answer  (indices drawn above)\n")
    a = ("            except:\n"
         "                if used_haystack > incremental:\n"
         "                    used_haystack -= incremental\n")
    assert a in text, "per-sample retry loop not found"
    text = text.replace(a,
        "            except Exception as _e:\n"
        "                " + MARKER + ": stock RULER retries the SAME failing call forever once `used_haystack`\n"
        "                # has bottomed out at `incremental` -- the needles alone do not fit and no haystack size helps.\n"
        "                if used_haystack > incremental:\n"
        "                    used_haystack -= incremental\n"
        "                else:\n"
        "                    raise RuntimeError(\n"
        "                        f'niah.py cannot fit {args.num_needle_k * args.num_needle_v} needles in '\n"
        "                        f'max_seq_length={max_seq_length} even at the minimum haystack ({incremental}): {_e}')\n")
    # ---------------------------------------------------------------- v3: the haystack-size rate estimate
    a = '    sample_input_text, _ = generate_input_output(incremental)\n    sample_tokens = len(TOKENIZER.text_to_tokens(sample_input_text))\n    tokens_per_haystack = sample_tokens / incremental\n'
    assert a in text, "tokens_per_haystack estimate not found"
    text = text.replace(a, '    ## MarSea: stock RULER divides the WHOLE probe\'s token count by the haystack size, charging the needles\'\n    # and template\'s FIXED cost to the per-word rate.  At 128 needles that inflates it ~5x, the 3x slack below\n    # does not absorb it, the binary search converges to its own upper bound, and a 16K request yields ~11.6K\n    # tokens -- under max_seq_length, so the assert never fires.  Two-point differencing cancels every\n    # needle- and template-sized constant exactly (measured 2026-09-16; see scripts/patch_ruler_length.py).\n    _t1 = len(TOKENIZER.text_to_tokens(generate_input_output(incremental)[0]))\n    _t2 = len(TOKENIZER.text_to_tokens(generate_input_output(2 * incremental)[0]))\n    tokens_per_haystack = max((_t2 - _t1) / incremental, 1e-6)\n    logger.info(f"MarSea rate by differencing: ({_t2} - {_t1}) / {incremental} = {tokens_per_haystack:.3f} tok/unit")\n')
    return text

if __name__ == "__main__":
    src = SRC.read_text()
    if not ORIG.exists():
        ORIG.write_text(src)
    new = apply(ORIG.read_text())
    SRC.write_text(new)
    compile(new, str(SRC), "exec")
    res = subprocess.run(["diff", "-u", str(ORIG), str(SRC)], capture_output=True, text=True)
    PATCH.write_text(res.stdout.replace(str(ORIG), "a/scripts/data/synthetic/niah.py").replace(str(SRC), "b/scripts/data/synthetic/niah.py"))
    print(f"patched {SRC}; diff -> {PATCH} ({len(res.stdout.splitlines())} lines)")
