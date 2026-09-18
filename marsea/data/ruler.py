"""
RULER NIAH (spec Sec. 13.1 [v4.1], round-2 C-1..C-3, D-27).

Generation drives the (patched) RULER script `scripts/data/synthetic/niah.py` with the BACKBONE tokenizer.
  E2 (m-sweep)    K = 8 keys, V = m in {1,2,4,8,16}, Q = 1, L = 8K
  E3 (n-sweep)    K = n in {8,16,32,64,128}, V = 4, Q = 1, L = 16K constant; inert-padding control K = 1, V = 4
  E4 (m = 1)      K = V = Q = 1 (niah_single_2), L = 8K
  depth control   --gold_depth d (our patch); default 0.5; profile {0.1,0.3,0.5,0.7,0.9}
  training pool   seeds 100-102; evaluation seeds 0-2 (never overlap)
jsonl fields: index, input (answer_prefix STRIPPED), outputs (list), length, answer_prefix, token_position_answer.
prompt = input + answer_prefix;  gold = " " + ", ".join(outputs)  (RULER's own template; no chat template).

Ground truth for teacher forcing (Sec. 12.2 / 12.3, D-9):
  answer rows      the positions whose NEXT token is the first token of each gold value (one row per value)
  K_i (row truth)  token spans of the gold needle values (value-level); needle sentences (sentence-level) logged too
  T_j (column)     coreference by string match, SPLIT (2026-09-09): key j = first token of the FIRST mention of the queried
                   key phrase; T_j = first tokens of every LATER mention (the other gold needles, the question, the answer
                   prefix) -> m_j = m + 1 on MV-NIAH.  The answer positions attach to the VALUE columns: key = first token
                   of needle value v, T_j = the row emitting v (m_j = 1).
"""
from __future__ import annotations
import json, os, pathlib, subprocess, sys, re
from dataclasses import dataclass, field, asdict
from typing import Optional
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
RULER_DIR = ROOT / "third_party" / "RULER"
NIAH = RULER_DIR / "scripts" / "data" / "synthetic" / "niah.py"
NIAH_TEMPLATE = ("Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. "
                 "I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic "
                 "{type_needle_v} for {query} mentioned in the provided text?{answer_prefix}")
# RULER's niah.py formats its template with {context}, {query} (and {type_needle_v}); the answer prefix is appended
# by the script through TASKS['niah']['answer_prefix'] and then STRIPPED into its own field.

EVAL_SEEDS = (0, 1, 2)
TRAIN_SEEDS = (100, 101, 102)
QUICK_SEED = 3          # the 250-step in-training eval: held out from BOTH (it used to reuse eval seed 0, i.e. E2's own test set)


@dataclass
class NIAHConfig:
    name: str
    max_seq_length: int
    num_needle_k: int
    num_needle_v: int
    num_needle_q: int = 1
    type_haystack: str = "essay"
    type_needle_k: str = "words"
    type_needle_v: str = "numbers"
    gold_depth: Optional[float] = 0.5
    num_samples: int = 200
    seed: int = 0
    tokens_to_generate: int = 128

    def save_name(self) -> str:
        d = "rand" if self.gold_depth is None else f"{self.gold_depth:.1f}"
        return f"{self.name}_L{self.max_seq_length}_K{self.num_needle_k}_V{self.num_needle_v}_Q{self.num_needle_q}_d{d}_s{self.seed}"


def grid_E2(L=8192, seeds=EVAL_SEEDS, n_samples=200):
    return [NIAHConfig("E2", L, 8, m, 1, seed=s, num_samples=n_samples) for m in (1, 2, 4, 8, 16) for s in seeds]


def grid_E3(L=16384, seeds=EVAL_SEEDS, n_samples=200, depth=0.5):
    cfgs = [NIAHConfig("E3", L, n, 4, 1, seed=s, num_samples=n_samples, gold_depth=depth) for n in (8, 16, 32, 64, 128) for s in seeds]
    cfgs += [NIAHConfig("E3pad", L, 1, 4, 1, seed=s, num_samples=n_samples, gold_depth=depth) for s in seeds]   # inert padding
    return cfgs


def grid_E3_depth(L=16384, seeds=EVAL_SEEDS, n_samples=200, n=32):
    return [NIAHConfig("E3depth", L, n, 4, 1, seed=s, num_samples=n_samples, gold_depth=d) for d in (0.1, 0.3, 0.5, 0.7, 0.9) for s in seeds]


def grid_E4(L=8192, seeds=EVAL_SEEDS, n_samples=200):
    return [NIAHConfig("E4", L, 1, 1, 1, seed=s, num_samples=n_samples) for s in seeds]


def grid_detector(L=4096, n_samples=200):
    return [NIAHConfig("DET", L, 1, 1, 1, type_needle_v="words", seed=0, num_samples=n_samples, gold_depth=None)]


def grid_training(L=8192, seeds=TRAIN_SEEDS, n_samples=400):
    """training procedure Sec. 2.1: uniform mixture over the grid; feasibility K*V <= 128 sentences; random depth."""
    cfgs = []
    for s in seeds:
        for K in (1, 2, 4, 8, 16, 32):
            for V in (1, 2, 4, 8):
                if K * V > 128:
                    continue
                cfgs.append(NIAHConfig("TRAIN", L, K, V, 1, seed=s, num_samples=n_samples, gold_depth=None))
    return cfgs


# Feasibility (RULER hazard).  niah.py's per-sample retry loop shrinks the haystack by `incremental` until the
# sample fits; when the NEEDLES alone leave no room it retries the same failing call forever.  scripts/patch_ruler.py
# (v2) turns that hang into a RuntimeError; this is the cheap pre-check, and it is measured rather than guessed.
# With the Qwen2.5 tokenizer a needle sentence ("One of the special magic X for <adj-noun> is: <value>.") costs 22
# tokens for `numbers`, 18 for `words`, ~40 for `uuids`; and niah.py's essay haystack is a list of WORDS, not
# sentences, so its `incremental` floor of 500 words is only ~613 tokens.  The old rule (16 tokens/needle against
# L // 4) was wrong in both directions: it under-counted the needle and over-counted the floor, and it rejected
# E3 at n = 128 -- 512 needles, ~12.1K tokens of a 16K budget -- which spec Sec. 13.1 sizes as legal.
TOK_PER_NEEDLE = {"numbers": 22, "words": 18, "uuids": 40}
MIN_HAYSTACK_TOK = {"essay": 640, "noise": 25 * 24, "needle": 25 * 22}   # `incremental` * tokens per unit
TEMPLATE_TOK = 96                                                        # NIAH_TEMPLATE + answer prefix + query
# The only length check RULER makes is its own `length <= max_seq_length`: nothing ever verified that a set
# REACHES the length it claims.  On 2026-09-16 a needle-contaminated rate estimate silently produced 11,598
# tokens for a 16,384 request, and because the shortfall scales with the needle count it would have made
# E3's sequence length a function of the swept variable n.  generate() now refuses such a set.
LENGTH_FLOOR_FRAC = float(os.environ.get("MARSEA_LENGTH_FLOOR", "0.95"))


def needle_budget(cfg: "NIAHConfig") -> tuple[int, int]:
    """(tokens the needles and the minimum haystack demand, tokens available).  Feasible iff demand <= available."""
    n_needles = cfg.num_needle_k * cfg.num_needle_v
    demand = (n_needles * TOK_PER_NEEDLE.get(cfg.type_needle_v, 22)
              + MIN_HAYSTACK_TOK.get(cfg.type_haystack, 640) + TEMPLATE_TOK + cfg.tokens_to_generate)
    return demand, cfg.max_seq_length


def feasible(cfg: "NIAHConfig") -> bool:
    d, a = needle_budget(cfg)
    return d <= a


def ensure_patched():
    src = NIAH.read_text()
    import importlib.util
    spec = importlib.util.spec_from_file_location("_marsea_patch_ruler", ROOT / "scripts" / "patch_ruler.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)   # load_module() is deprecated/removed
    marker = mod.MARKER
    if marker not in src:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "patch_ruler.py")], check=True)


def generate(cfg: NIAHConfig, tokenizer_path: str, out_dir: pathlib.Path, force: bool = False, timeout: int = 3600) -> pathlib.Path:
    """Runs RULER's niah.py; returns the jsonl path.  Idempotent.

    An existing set is returned BEFORE the generator is touched: the pods have the uploaded data and no
    third_party/RULER (setup_runpod.sh clones it only when generation is needed), and ensure_patched() used to run
    first, so preflight step 6 on pod 1 died with FileNotFoundError on niah.py with the DET set sitting in data/ruler."""
    out_dir = pathlib.Path(out_dir).resolve()               # niah.py runs with cwd = its own directory
    save_name = cfg.save_name()
    target = out_dir / save_name / "validation.jsonl"
    if target.exists() and not force:
        return target
    ensure_patched()
    demand, avail = needle_budget(cfg)
    if demand > avail:
        raise ValueError(f"{cfg.save_name()}: {cfg.num_needle_k * cfg.num_needle_v} needle sentences plus the minimum "
                         f"haystack demand ~{demand} tokens, over L = {avail} (see needle_budget)")
    essay = NIAH.parent / "json" / "PaulGrahamEssays.json"
    if cfg.type_haystack == "essay" and not essay.exists():
        raise FileNotFoundError(f"{essay} missing: run third_party/RULER/scripts/data/synthetic/json/download_paulgraham_essay.py")
    template = ("Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. "
                "I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic "
                "{type_needle_v} for {query} mentioned in the provided text? The special magic {type_needle_v} for {query} "
                "mentioned in the provided text are")
    cmd = [sys.executable, str(NIAH), "--save_dir", str(out_dir), "--save_name", save_name, "--subset", "validation",
           "--tokenizer_path", tokenizer_path, "--tokenizer_type", "hf", "--max_seq_length", str(cfg.max_seq_length),
           "--tokens_to_generate", str(cfg.tokens_to_generate), "--num_samples", str(cfg.num_samples),
           "--random_seed", str(cfg.seed), "--template", template,
           "--num_needle_k", str(cfg.num_needle_k), "--num_needle_v", str(cfg.num_needle_v), "--num_needle_q", str(cfg.num_needle_q),
           "--type_haystack", cfg.type_haystack, "--type_needle_k", cfg.type_needle_k, "--type_needle_v", cfg.type_needle_v]
    if cfg.gold_depth is not None:
        cmd += ["--gold_depth", str(cfg.gold_depth)]
    env = dict(os.environ, PYTHONPATH=str(NIAH.parent.parent) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    try:
        res = subprocess.run(cmd, cwd=str(NIAH.parent), env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"niah.py exceeded {timeout}s for {save_name} (RULER's retry loop can spin forever)") from e
    if res.returncode != 0:
        raise RuntimeError(f"niah.py failed:\n{res.stdout[-2000:]}\n{res.stderr[-4000:]}")
    assert target.exists(), target
    # ---- the length gate (see LENGTH_FLOOR_FRAC).  Measured here rather than trusted, and recorded.
    lens = [json.loads(l)["length"] for l in target.read_text().splitlines() if l.strip()]
    stats = {}
    if lens:
        stats = dict(measured_length_mean=sum(lens) / len(lens), measured_length_min=min(lens),
                     measured_length_max=max(lens), measured_length_n=len(lens),
                     length_floor_frac=LENGTH_FLOOR_FRAC)
        if stats["measured_length_mean"] < LENGTH_FLOOR_FRAC * cfg.max_seq_length:
            target.unlink()                     # do NOT leave a short set on disk: generate() skips existing targets
            raise RuntimeError(
                f"{save_name}: mean length {stats['measured_length_mean']:.0f} tokens is below "
                f"{LENGTH_FLOOR_FRAC:.2f} x L = {LENGTH_FLOOR_FRAC * cfg.max_seq_length:.0f}. The set was deleted. "
                f"Re-run scripts/patch_ruler.py (v3 fixes RULER's haystack-size estimate); if the shortfall is "
                f"intended, set MARSEA_LENGTH_FLOOR.")
    (out_dir / save_name / "config.json").write_text(json.dumps(dict(asdict(cfg), **stats), indent=1))
    return target


def load_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_set(path) -> list[dict]:
    """load a generated set and attach its config (num_needle_k / num_needle_v / gold_depth / seed) to every record, so
    the evaluation rows can be stratified by n (E3) and depth (the position profile)."""
    path = pathlib.Path(path)
    recs = load_jsonl(path)
    cfg_path = path.parent / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    for r in recs:
        r["_config"] = cfg
    return recs


# --------------------------------------------------------------------------- prompt assembly and ground truth
NEEDLE_RE = re.compile(r"One of the special magic (\w+) for ([\w\-]+) is: ([\w\-]+)\.")


@dataclass
class RULERExample:
    prompt: str
    gold: str                          # " v1, v2, ..."
    outputs: list
    query_key: str
    prompt_ids: list
    gold_ids: list
    answer_rows: list                  # teacher-forced positions emitting the FIRST token of each gold value
    value_spans: list                  # [(start, end)] token spans of the gold needle VALUES in the prompt
    sentence_spans: list               # [(start, end)] token spans of the gold needle sentences
    key_mentions: list                 # token index of the first token of every mention of the query key (in order)
    key_mention_spans: list = field(default_factory=list)   # [(start, end)] token spans of every mention, in text order
    answer_value_rows: list = field(default_factory=list)   # per gold value: the rows emitting EACH of its tokens
    question_mention: Optional[int] = None    # first token of the key phrase in the question
    T_j: dict = field(default_factory=dict)    # {key_token_index j: [target rows]}: coref columns (later mentions) + value columns (answer rows)
    T_j_kind: dict = field(default_factory=dict)   # {j: 'coref' | 'value'}
    meta: dict = field(default_factory=dict)


MENTION_CACHE: dict = {}
MENTION_CACHE_MAX = 4096        # bounded: the annotator sees a new key phrase per example and the cache never shrank


def word_bounded_mentions(needle: str, hay: str, start: int = 0, end: Optional[int] = None) -> list:
    """char offsets of every occurrence of `needle` in hay[start:end] that is not inside a longer word.

    Plain `re.escape(needle)` matched substrings of haystack words: the queried key is an adjective-noun pair drawn
    from wonderwords, and a haystack hit RELOCATES key_mentions[0], which anchors the whole D-9a coreference column
    (review d5bd980 D-4).  Lookarounds rather than \b, because the key phrase may begin or end with a non-word
    character (hyphens, commas and spaces all occur in it)."""
    if not needle:
        return []
    pat = MENTION_CACHE.get(needle)
    if pat is None:
        if len(MENTION_CACHE) >= MENTION_CACHE_MAX:
            MENTION_CACHE.clear()
        # the guard is on both sides and on BOTH classes: \w after a word character, and a repeat of the needle's own
        # last character -- "c++" otherwise matches inside "c+++" (harmless on RULER's adjective-noun keys, reachable
        # on a MuSiQue bridge string; review 30372ae E)
        head = r"(?<!\w)" if needle[:1].isalnum() or needle[:1] == "_" else r"(?<!" + re.escape(needle[0]) + r")"
        tail = r"(?!\w)" if needle[-1:].isalnum() or needle[-1:] == "_" else r"(?!" + re.escape(needle[-1]) + r")"
        pat = MENTION_CACHE[needle] = re.compile(head + re.escape(needle) + tail)
    seg = hay[start:end] if end is not None else hay[start:]
    return [start + m.start() for m in pat.finditer(seg)]


def _char_to_token(offsets, c):
    """first token whose span contains char c (or starts after c)."""
    for t, (a, b) in enumerate(offsets):
        if a <= c < b or a >= c:
            return t
    return len(offsets) - 1


def build_example(rec: dict, tok, add_gold: bool = True) -> RULERExample:
    prompt = rec["input"] + rec["answer_prefix"]
    outputs = [str(o) for o in rec["outputs"]]
    gold = " " + ", ".join(outputs)
    enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    gold_ids = tok(gold, add_special_tokens=False)["input_ids"]
    # the queried key: last "for {key} mentioned" in the answer prefix
    m = re.search(r"for ([\w\-, ]+?) mentioned in the provided text (?:are|is)", rec["answer_prefix"])   # "is" when Q*V == 1
    query_key = m.group(1).strip() if m else ""
    # gold needle sentences and value spans by exact string search (values are unique by construction)
    value_spans, sentence_spans = [], []
    for v in outputs:
        for mm in NEEDLE_RE.finditer(rec["input"]):
            if mm.group(3) == v and mm.group(2) == query_key:
                sentence_spans.append((_char_to_token(offsets, mm.start()), _char_to_token(offsets, mm.end() - 1) + 1))
                value_spans.append((_char_to_token(offsets, mm.start(3)), _char_to_token(offsets, mm.end(3) - 1) + 1))
                break
    # every mention of the key phrase (needles first, then the question / answer prefix), in text order
    mention_starts = word_bounded_mentions(query_key, prompt)
    key_mention_spans = [(_char_to_token(offsets, c), _char_to_token(offsets, c + len(query_key) - 1) + 1)
                         for c in mention_starts]
    key_mentions = [a for a, _ in key_mention_spans]
    tail = [c for c in mention_starts if c >= len(rec["input"]) - 400]
    question_mention = _char_to_token(offsets, tail[0]) if tail else None
    # answer rows: position emitting the first token of each gold value inside " v1, v2, ..."
    answer_rows = []; answer_value_rows = []
    genc = tok(gold, return_offsets_mapping=True, add_special_tokens=False)
    for v in outputs:
        c = gold.find(v)
        if c < 0:
            continue
        t = _char_to_token(genc["offset_mapping"], c)
        t_end = _char_to_token(genc["offset_mapping"], c + len(v) - 1) + 1
        answer_rows.append(len(ids) + t - 1)                       # the position that EMITS token t of the gold
        answer_value_rows.append([len(ids) + tt - 1 for tt in range(t, t_end)])   # every token of the value
    # T_j (D-9): key j = first token of the FIRST mention; targets = later mentions' first tokens + answer rows
    # the SPLIT (spec v4.7 / paper Sec. 2.3, 2026-09-09): the coreference column of the key phrase's first mention has the
    # LATER MENTIONS as targets -- (m - 1) other needles + the question + the answer prefix = m + 1 on MV-NIAH -- and the
    # answer positions belong to the VALUE columns (key = first token of needle value v; target = the row emitting v)
    T_j = {}; T_j_kind = {}
    if key_mentions:
        j = key_mentions[0]
        T_j[j] = sorted(set(key_mentions[1:])); T_j_kind[j] = "coref"
    for (a, b), row in zip(value_spans, answer_rows):
        if a not in T_j:
            T_j[a] = [row]; T_j_kind[a] = "value"
    return RULERExample(prompt=prompt, gold=gold, outputs=outputs, query_key=query_key, prompt_ids=ids, gold_ids=gold_ids,
                        answer_rows=answer_rows, value_spans=value_spans, sentence_spans=sentence_spans,
                        key_mentions=key_mentions, key_mention_spans=key_mention_spans, answer_value_rows=answer_value_rows,
                        question_mention=question_mention, T_j=T_j, T_j_kind=T_j_kind,
                        meta=dict(length=rec.get("length"), index=rec.get("index"),
                                  num_needle_k=rec.get("_config", {}).get("num_needle_k"), num_needle_v=rec.get("_config", {}).get("num_needle_v"),
                                  gold_depth=rec.get("_config", {}).get("gold_depth"), seed=rec.get("_config", {}).get("seed")))


def exact_set_accuracy(generated: str, outputs: list) -> bool:
    """Pre-committed E2 criterion: the generated list, split on ', ', equals the gold set (no distractor admitted)."""
    text = generated.strip().split("\n")[0]
    text = re.split(r"[.]|\bmentioned\b", text)[0]
    pred = {p.strip().strip(".,") for p in text.split(",") if p.strip()}
    return pred == {str(o) for o in outputs}


def ruler_recall_score(generated: str, outputs: list) -> float:
    """RULER's own (recall-based) score: fraction of gold values present in the generation."""
    return sum(1.0 for o in outputs if str(o) in generated) / max(1, len(outputs))
