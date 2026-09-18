"""
Training procedure Sec. 2: the 1:1:1 mix, one example per sequence at its own length (B = 1: NO padding, NO packing),
sequences longer than L dropped (never truncated), a per-seed permutation of each source consumed in lock-step,
the data cursor part of the checkpoint so a restart resumes the identical order.

  text   = prompt + gold + EOS;   labels = -100 on every prompt token; the answer tokens and the EOS carry the label
"""
from __future__ import annotations
import json, pathlib, random, hashlib
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch

SOURCES = ("ruler", "musique", "hotpot")


@dataclass
class Sequence:
    input_ids: torch.Tensor          # [1, T]
    labels: torch.Tensor             # [1, T]
    source: str
    index: int
    n_label_tokens: int


class MixedDataset:
    """sources: {name: list of (prompt: str, gold: str)}.  Lengths are tokenised once and cached to `cache_dir`."""

    def __init__(self, sources: dict, tok, L: int, seed: int, cache_dir: Optional[pathlib.Path] = None, eos_id=None):
        self.tok = tok; self.L = L; self.seed = seed
        self.eos_id = eos_id if eos_id is not None else tok.eos_token_id
        self.sources = {k: v for k, v in sources.items() if len(v)}
        self.names = [n for n in SOURCES if n in self.sources] or list(self.sources)
        self.perms = {}
        for name in self.names:
            items = self.sources[name]
            lengths = self._lengths(name, items, cache_dir)
            keep = [i for i, ln in enumerate(lengths) if ln <= L]
            rng = np.random.default_rng(seed * 1000003 + sum(map(ord, name)))
            self.perms[name] = [keep[i] for i in rng.permutation(len(keep))]
            print(f"[mix] {name}: {len(keep)}/{len(items)} sequences <= L={L}")
            if not keep:
                raise ValueError(f"[mix] source {name!r} has no sequence <= L={L}: sequences are dropped, never truncated")

    def _lengths(self, name, items, cache_dir):
        key = hashlib.md5((name + str(len(items)) + str(self.L) + items[0][0][:200] + items[-1][0][:200]).encode()).hexdigest()[:12]
        if cache_dir is not None:
            f = pathlib.Path(cache_dir) / f"lengths_{name}_{key}.json"
            if f.exists():
                return json.loads(f.read_text())
        lengths = []
        for (p, g) in items:
            lengths.append(len(self.tok(p, add_special_tokens=False)["input_ids"]) + len(self.tok(g, add_special_tokens=False)["input_ids"]) + 1)
        if cache_dir is not None:
            pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(lengths))
        return lengths

    def __len__(self):
        return len(self.names) * min(len(self.perms[n]) for n in self.names)

    def get(self, cursor: int) -> Sequence:
        """deterministic: source = cursor mod #sources, index = perm[source][cursor // #sources]."""
        name = self.names[cursor % len(self.names)]
        perm = self.perms[name]
        idx = perm[(cursor // len(self.names)) % len(perm)]
        prompt, gold = self.sources[name][idx]
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        g_ids = self.tok(gold, add_special_tokens=False)["input_ids"] + [self.eos_id]
        ids = torch.tensor([p_ids + g_ids])
        labels = torch.tensor([[-100] * len(p_ids) + g_ids])
        return Sequence(ids, labels, name, idx, len(g_ids))


def build_training_sources(ruler_jsonls: list, musique_path: Optional[str], hotpot_n: int, seed: int, tok=None,
                           hotpot_split: str = "train", cache_dir=None) -> dict:
    """assemble (prompt, gold) pairs per source (training procedure Sec. 2.1)."""
    from .ruler import load_jsonl
    from .qa import load_musique, musique_example, load_hotpot, hotpot_example
    out = {}
    ruler = []
    for path in ruler_jsonls:
        for rec in load_jsonl(path):
            ruler.append((rec["input"] + rec["answer_prefix"], " " + ", ".join(str(o) for o in rec["outputs"])))
    if ruler: out["ruler"] = ruler
    # every source the caller NAMES must arrive.  MuSiQue used to be skipped when its file was absent (it is a manual
    # Google-Drive download nothing verifies) and HotpotQA's failure was caught and printed, so an arm could train on
    # a different mixture from the procedure's and nothing noticed (review e982f83 I).
    if ruler_jsonls and not ruler:
        raise RuntimeError(f"RULER training files named but empty: {ruler_jsonls[:3]}...")
    if musique_path:
        if not pathlib.Path(musique_path).exists():
            raise FileNotFoundError(f"MuSiQue training file {musique_path} is missing (manual download; see RUNBOOK_nebius.md)")
        recs = load_musique(musique_path)
        out["musique"] = [(ex.prompt, ex.gold) for ex in (musique_example(r, seed) for r in recs)]
        if not out["musique"]:
            raise RuntimeError(f"MuSiQue training file {musique_path} yielded no examples")
    if hotpot_n:
        ds = load_hotpot(hotpot_split, n=hotpot_n, seed=seed, cache_dir=cache_dir)     # raises if unavailable
        out["hotpot"] = [(ex.prompt, ex.gold) for ex in (hotpot_example(r, seed) for r in ds)]
        if not out["hotpot"]:
            raise RuntimeError("HotpotQA yielded no training examples")
    return out
