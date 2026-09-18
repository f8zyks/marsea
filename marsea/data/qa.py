"""
MuSiQue-Answerable and HotpotQA-distractor (spec Sec. 13.1; training procedure Sec. 2).

MuSiQue   musique_ans_v1.0_{train,dev}.jsonl (github.com/StonyBrookNLP/musique release).  Fields: question, answer,
          paragraphs[{idx, title, paragraph_text, is_supporting}], question_decomposition[{question, answer,
          paragraph_support_idx}] -- the intermediate answers are the BRIDGE strings for T_j (D-9).
          m = number of is_supporting paragraphs (2-4).  E3-style n-sweeps sample distractor paragraphs from OTHER
          questions' non-supporting paragraphs (seeded).
HotpotQA  HF "hotpot_qa" / "distractor": question, answer, context{title, sentences}, supporting_facts{title, sent_id}.
Prompt    "{context}\\n\\nQuestion: {question}\\nAnswer:" -> " {answer}";  context = passages "Title: {t}\\n{text}" joined
          by "\\n\\n"; gold passages in HOP ORDER, distractors interleaved at fixed slots by seed (Sec. 12.3 / 13.1).
"""
from __future__ import annotations
import hashlib, json, re, string, collections, random, pathlib
from dataclasses import dataclass, field
from typing import Optional

MUSIQUE_URL = "https://github.com/StonyBrookNLP/musique  (release: musique_v1.0.zip -> data/musique_ans_v1.0_{train,dev}.jsonl)"


@dataclass
class QAExample:
    prompt: str
    gold: str
    answer: str
    question: str
    passages: list                         # [(title, text, is_gold)] in prompt order
    gold_slots: list                       # indices (in prompt order) of the gold passages
    m: int
    prompt_ids: list = field(default_factory=list)
    gold_ids: list = field(default_factory=list)
    passage_spans: list = field(default_factory=list)      # token spans of each passage in prompt order
    sentence_spans: list = field(default_factory=list)     # HotpotQA: [(passage_slot, sent_id, (a, b))] gold sentences
    answer_rows: list = field(default_factory=list)        # every position emitting a gold answer token
    bridge_strings: list = field(default_factory=list)
    T_j: dict = field(default_factory=dict)                # key token j -> [target rows]: coref (later mentions) + value (answer rows)
    T_j_kind: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def _interleave(gold: list, distractors: list, seed: int):
    """gold passages in hop order; distractors at fixed slots drawn by seed (the same slots for every arm)."""
    rng = random.Random(seed)
    n = len(gold) + len(distractors)
    slots = sorted(rng.sample(range(n), len(gold)))
    out = []; gi = 0; di = 0
    dis = list(distractors); rng.shuffle(dis)
    for s in range(n):
        if gi < len(gold) and s == slots[gi]:
            out.append(gold[gi]); gi += 1
        else:
            out.append(dis[di]); di += 1
    return out, slots


def _stable_hash(s) -> int:
    """A per-record seed that survives a new process.  Python's hash() is SALTED for str (PYTHONHASHSEED), so every
    arm -- each a separate process -- laid its passages out differently and a resumed run did not reproduce its own
    order; spec Sec. 13.1 requires fixed slots by seed (review d5bd980 D-3)."""
    return int(hashlib.md5(str(s).encode()).hexdigest()[:8], 16)


from .ruler import word_bounded_mentions        # word-bounded string matching, shared with the RULER annotator

def format_prompt(passages, question):
    ctx = "\n\n".join(f"Title: {t}\n{x}" for t, x, _ in passages)
    return f"{ctx}\n\nQuestion: {question}\nAnswer:"


# --------------------------------------------------------------------------- MuSiQue
def load_musique(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def musique_example(rec: dict, seed: int, n_total: Optional[int] = None, distractor_pool: Optional[list] = None) -> QAExample:
    """n_total: for E3-style sweeps, the total number of passages (gold + distractors sampled from `distractor_pool`
    = other questions' non-supporting paragraphs); None = the record's own 20 paragraphs."""
    paras = rec["paragraphs"]
    # gold in HOP order: the decomposition's paragraph_support_idx sequence
    hop_idx = [d.get("paragraph_support_idx") for d in rec.get("question_decomposition", []) if d.get("paragraph_support_idx") is not None]
    gold_idx = [i for i in hop_idx if i is not None]
    if not gold_idx:
        gold_idx = [p["idx"] for p in paras if p["is_supporting"]]
    by_idx = {p["idx"]: p for p in paras}
    gold = [(by_idx[i]["title"], by_idx[i]["paragraph_text"], True) for i in gold_idx if i in by_idx]
    own_dis = [(p["title"], p["paragraph_text"], False) for p in paras if not p["is_supporting"]]
    if n_total is not None:
        need = max(0, n_total - len(gold))
        rng = random.Random(seed * 7919 + _stable_hash(rec["id"]) % 10007)
        pool = distractor_pool if distractor_pool is not None else own_dis
        dis = rng.sample(pool, min(need, len(pool)))
    else:
        dis = own_dis
    passages, slots = _interleave(gold, dis, seed + (_stable_hash(rec["id"]) % 100003))
    bridges = [d["answer"] for d in rec.get("question_decomposition", [])[:-1] if d.get("answer")]
    return QAExample(prompt=format_prompt(passages, rec["question"]), gold=" " + rec["answer"], answer=rec["answer"],
                     question=rec["question"], passages=passages, gold_slots=slots, m=len(gold), bridge_strings=bridges,
                     meta=dict(id=rec["id"], hops=len(gold), answerable=rec.get("answerable", True)))


def musique_distractor_pool(records: list[dict], exclude_id: Optional[str] = None) -> list:
    return [(p["title"], p["paragraph_text"], False) for r in records if r["id"] != exclude_id for p in r["paragraphs"] if not p["is_supporting"]]


# --------------------------------------------------------------------------- HotpotQA
def load_hotpot(split: str = "train", n: Optional[int] = None, seed: int = 0, cache_dir=None):
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, cache_dir=cache_dir)
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    return ds


def hotpot_example(rec: dict, seed: int) -> QAExample:
    titles = rec["context"]["title"]; sents = rec["context"]["sentences"]
    sup_titles = rec["supporting_facts"]["title"]; sup_ids = rec["supporting_facts"]["sent_id"]
    gold_titles = list(dict.fromkeys(sup_titles))                    # order of first appearance in the annotation
    passages_all = {t: "".join(s) for t, s in zip(titles, sents)}
    gold = [(t, passages_all[t], True) for t in gold_titles if t in passages_all]
    dis = [(t, passages_all[t], False) for t in titles if t not in gold_titles]
    passages, slots = _interleave(gold, dis, seed + (_stable_hash(rec["id"]) % 100003))
    ex = QAExample(prompt=format_prompt(passages, rec["question"]), gold=" " + rec["answer"], answer=rec["answer"],
                   question=rec["question"], passages=passages, gold_slots=slots, m=len(gold),
                   meta=dict(id=rec["id"], level=rec.get("level"), type=rec.get("type"),
                             gold_sentences=[(t, int(i)) for t, i in zip(sup_titles, sup_ids)],
                             sentences={t: s for t, s in zip(titles, sents)}))
    return ex


# --------------------------------------------------------------------------- tokenisation, spans, T_j
def _char_to_token(offsets, c):
    for t, (a, b) in enumerate(offsets):
        if a <= c < b or a >= c:
            return t
    return len(offsets) - 1


def tokenize_example(ex: QAExample, tok) -> QAExample:
    enc = tok(ex.prompt, return_offsets_mapping=True, add_special_tokens=False)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    ex.prompt_ids = ids
    ex.gold_ids = tok(ex.gold, add_special_tokens=False)["input_ids"]
    ex.answer_rows = [len(ids) + t - 1 for t in range(len(ex.gold_ids))]          # every gold token (QA; C-29)
    # passage spans
    spans = []; c = 0
    for (t, x, _) in ex.passages:
        block = f"Title: {t}\n{x}"
        a = ex.prompt.find(block, c); b = a + len(block); c = b
        spans.append((_char_to_token(offsets, a), _char_to_token(offsets, b - 1) + 1))
    ex.passage_spans = spans
    # HotpotQA gold sentences at sentence granularity
    if "gold_sentences" in ex.meta:
        sent_spans = []
        for (t, sid) in ex.meta["gold_sentences"]:
            slot = next((k for k, (tt, _, _) in enumerate(ex.passages) if tt == t), None)
            if slot is None: continue
            sl = ex.meta["sentences"][t]
            if sid >= len(sl): continue
            pre = f"Title: {t}\n" + "".join(sl[:sid])
            a = ex.prompt.find(f"Title: {t}\n") + len(pre); b = a + len(sl[sid])
            sent_spans.append((slot, sid, (_char_to_token(offsets, a), _char_to_token(offsets, max(a, b - 1)) + 1)))
        ex.sentence_spans = sent_spans
    # T_j by coreference string match (D-9), SPLIT (spec v4.7): the bridge string's first mention has the LATER
    # mentions as targets (the hop-(h+1) passage, the question), and the VALUE column -- the first mention of the gold
    # answer string in the context -- has the row that emits its first token, m_j = 1, exactly as on RULER
    T = {}; kind = {}
    for s in ex.bridge_strings:
        if not s or len(s) < 2: continue
        ms = word_bounded_mentions(s, ex.prompt)               # not a substring of a longer word (review d5bd980 D-4)
        if len(ms) < 2: continue                                        # a bridge with no later mention has no column
        j = _char_to_token(offsets, ms[0])
        T[j] = sorted({_char_to_token(offsets, c) for c in ms[1:]}); kind[j] = "coref"
    if ex.answer and len(ex.answer) >= 2:
        ctx_end = ex.prompt.rfind("\n\nQuestion:")
        cs = word_bounded_mentions(ex.answer, ex.prompt, 0, ctx_end if ctx_end > 0 else None)
        c = cs[0] if cs else -1
        if c >= 0:
            j = _char_to_token(offsets, c)
            if j not in T:
                # the VALUE column matches RULER's: key = the first token of the answer's mention in the context,
                # target = the row that EMITS it, so m_j = 1.  It used to take every gold-token row, giving
                # m_j = len(gold_ids) here and 1 on RULER while aggregate_ruler pooled both under "value"
                # (review d5bd980 E).
                T[j] = list(ex.answer_rows[:1]); kind[j] = "value"
    ex.T_j = T; ex.T_j_kind = kind
    return ex


# --------------------------------------------------------------------------- metrics (official-style)
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred: str, gold: str):
    p = normalize_answer(pred).split(); g = normalize_answer(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0, 0.0, 0.0
    prec = ns / len(p); rec = ns / len(g)
    return 2 * prec * rec / (prec + rec), prec, rec


def em_score(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def support_prf(pred_set, gold_set):
    pred_set, gold_set = set(pred_set), set(gold_set)
    tp = len(pred_set & gold_set)
    P = tp / len(pred_set) if pred_set else 0.0
    R = tp / len(gold_set) if gold_set else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F, float(pred_set == gold_set)


def hotpot_joint(ans_pred, ans_gold, sp_pred, sp_gold):
    """official HotpotQA joint metrics: P_joint = P_ans P_sp, R_joint = R_ans R_sp; joint EM demands both exact."""
    f_a, p_a, r_a = f1_score(ans_pred, ans_gold); em_a = em_score(ans_pred, ans_gold)
    p_s, r_s, f_s, em_s = support_prf(sp_pred, sp_gold)
    pj, rj = p_a * p_s, r_a * r_s
    fj = 2 * pj * rj / (pj + rj) if (pj + rj) else 0.0
    return dict(ans_em=em_a, ans_f1=f_a, sp_em=em_s, sp_f1=f_s, sp_p=p_s, sp_r=r_s, joint_em=em_a * em_s, joint_f1=fj, joint_p=pj, joint_r=rj)


def parse_answer(generated: str) -> str:
    return generated.strip().split("\n")[0].strip()
