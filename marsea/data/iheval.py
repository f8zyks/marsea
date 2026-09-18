"""
IHEval (Zhang et al. 2025; github.com/ytyz1307zzh/IHEval), the multi-turn rule-following split (E6, S4).
The number of conflicting tiers is read from each example's metadata.  Base checkpoint, no chat template:
    "System: {system}\\n\\nUser: {user}\\n\\nAssistant:"     (spec Sec. 13.1 [v4.1])
E6 reports accuracy against conflicting-tier count as a SLOPE, never a pooled mean.

The repository's file layout is read at run time (`load_iheval(path)` accepts a jsonl / json file of records); each
record must expose: system (str), turns (list of {role, content}) or user (str), reference/answer, and a metadata
field carrying the tier count ("num_tiers" / "n_conflicts" / "depth").  Records lacking a tier count are dropped.
"""
from __future__ import annotations
import json, pathlib
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IHExample:
    prompt: str
    reference: str
    tiers: int
    meta: dict = field(default_factory=dict)


def _tier_count(rec: dict):
    for k in ("num_tiers", "n_tiers", "n_conflicts", "num_conflicts", "depth", "conflict_depth"):
        if k in rec: return int(rec[k])
        if isinstance(rec.get("metadata"), dict) and k in rec["metadata"]: return int(rec["metadata"][k])
    return None


def load_iheval(path, stats: Optional[dict] = None) -> list[IHExample]:
    """`stats` (if given) receives {n_records, n_kept, n_dropped_no_tier}: a record with no tier count is dropped,
    and a PARTIAL drop used to be invisible -- a set half of whose records lack the key would silently halve E6
    (review d5bd980 F)."""
    path = pathlib.Path(path)
    text = path.read_text()
    recs = json.loads(text) if text.lstrip().startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]
    out, dropped = [], 0
    for r in recs:
        tiers = _tier_count(r)
        if tiers is None:
            dropped += 1
            continue
        system = r.get("system", r.get("system_prompt", ""))
        if "turns" in r:
            user = "\n\n".join(f"{t.get('role', 'User').title()}: {t['content']}" for t in r["turns"])
            prompt = f"System: {system}\n\n{user}\n\nAssistant:"
        else:
            prompt = f"System: {system}\n\nUser: {r.get('user', r.get('input', ''))}\n\nAssistant:"
        ref = r.get("reference", r.get("answer", r.get("output", "")))
        out.append(IHExample(prompt=prompt, reference=str(ref), tiers=tiers, meta=dict(id=r.get("id"))))
    if stats is not None:
        stats.update(n_records=len(recs), n_kept=len(out), n_dropped_no_tier=dropped)
    if dropped:
        print(f"[iheval] {dropped} of {len(recs)} records carry no tier count and were dropped")
    return out


def accuracy_by_tier(rows: list[dict]) -> dict:
    """rows: [{tiers, correct}] -> {tiers: accuracy}, plus the fitted slope of accuracy against tier count."""
    import numpy as np
    by = {}
    for r in rows:
        by.setdefault(r["tiers"], []).append(float(r["correct"]))
    acc = {t: float(np.mean(v)) for t, v in sorted(by.items())}
    if len(acc) >= 2:
        xs = np.array(list(acc.keys()), float); ys = np.array(list(acc.values()))
        slope = float(np.polyfit(xs, ys, 1)[0])
    else:
        slope = float("nan")
    return dict(accuracy_by_tier=acc, slope=slope, n_by_tier={t: len(v) for t, v in by.items()})
