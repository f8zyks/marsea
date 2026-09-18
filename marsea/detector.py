"""
Sec. 6.7: the retrieval-head detector (Wu et al. 2024, arXiv:2404.15574) on the UNPATCHED backbone.

Detector set: 200 RULER S-NIAH prompts at 4K (type_needle_v = words, seed 0), teacher-forced, eager attention.
score(l, h) = mean over prompts of the fraction of gold-value answer positions whose row-argmax (over the whole
visible row) lands on the needle's copy of the token being emitted; a head "retrieves" if score > 0.1.
PATCHED_LAYERS = the L_patch = 4 distinct layers holding the top-scoring heads; (l*, h*) = the top head of the
top-scoring patched layer.  Scores per (layer, head) are recorded with the run.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, fields, asdict
from typing import Optional
import torch


@dataclass
class DetectorResult:
    scores: list                       # [L][H] mean score
    patched_layers: list
    l_star: int
    h_star: int
    n_prompts: int
    threshold: float
    ranking: list                      # [(score, layer, head)] descending
    induction_scores: Optional[list] = None      # [L][H] coreference routing fraction (marsea/sweep.py), added by run_s0_sweep --update_detector
    induction_best: Optional[dict] = None
    induction_layer_added: Optional[int] = None
    n_heads_above_threshold: Optional[int] = None    # how many (l, h) clear `threshold` (Sec. 6.7's "a head retrieves")
    layers_below_threshold: list = field(default_factory=list)   # layers that entered PATCHED_LAYERS without clearing it
    usable: Optional[bool] = None                    # False = NO head retrieves anywhere: App. H's fallback, not a run
    extra: dict = field(default_factory=dict)        # keys written by later stages (the S0 licence)
    # provenance, as on every checkpoint: this artefact chooses PATCHED_LAYERS and (l*, h*) for the whole programme,
    # and a run must be able to say which version of the detector produced the one it used (review 30372ae D)
    spec_version: Optional[str] = None
    git: Optional[str] = None
    created: Optional[str] = None
    backbone: Optional[str] = None

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=1)

    @staticmethod
    def load(path):
        """Tolerates keys this class does not declare: run_s0_sweep writes the S0 licence into the same file, and
        DetectorResult(**json.load(f)) rejected them -- which made this loader dead code."""
        with open(path) as f:
            raw = json.load(f)
        known = {f.name for f in fields(DetectorResult)}
        res = DetectorResult(**{k: v for k, v in raw.items() if k in known})
        res.extra = {k: v for k, v in raw.items() if k not in known}
        return res

    def stamp(self, backbone: Optional[str] = None):
        """record which code produced this detector."""
        import datetime
        from . import SPEC_VERSION
        from .train import git_hash
        self.spec_version = SPEC_VERSION
        self.git = git_hash()
        self.created = datetime.datetime.now().isoformat(timespec="seconds")
        self.backbone = backbone
        return self


def answer_and_needle_positions(prompt_ids: list[int], gold_ids: list[int], needle_token_positions: list[list[int]]):
    """Teacher forcing: the position that emits gold token t is len(prompt) + t - 1.  For each emitted gold token,
    the needle copy = the position(s) in the context holding the same token id inside a gold needle span.
    Returns [(answer_pos, [copy_positions])] for answer tokens that have a copy."""
    out = []
    span_pos = [p for span in needle_token_positions for p in span]
    for t, tid in enumerate(gold_ids):
        pos = len(prompt_ids) + t - 1
        copies = [p for p in span_pos if prompt_ids[p] == tid]
        if copies:
            out.append((pos, copies))
    return out


@torch.no_grad()
def run_detector(model, tok, examples: list[dict], threshold: float = 0.1, L_patch: int = 4, device="cuda",
                 max_prompts: int = 200) -> DetectorResult:
    """examples: dicts with prompt_ids (list[int]), gold_ids (list[int]), needle_token_positions (list[list[int]]).
    The model must be loaded with attn_implementation="eager" (attention weights are needed) and UNPATCHED."""
    model.eval()
    cfg = model.config
    L, H = cfg.num_hidden_layers, cfg.num_attention_heads
    tot = torch.zeros(L, H, dtype=torch.float64)
    n_used = 0
    base = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    attn_mods = [layer.self_attn for layer in base.layers]
    argmax_store = {}

    def make_hook(li):
        def hook(mod, args, out):
            w = out[1]                                     # [1,H,T,T] attention weights (eager path)
            if w is None:
                return
            rows = hook.rows
            wr = w[0, :, rows, :].clone()
            # D-31 / sweep.py:62: position 0 is the attention sink of a BOS-less decoder and is very often the row
            # maximum in a base model, so an argmax that includes it measures the sink, not retrieval.  The routing
            # sweep excluded it; this gate -- which chooses PATCHED_LAYERS and (l*, h*) for everything -- did not.
            wr[..., 0] = float("-inf")
            argmax_store[li] = wr.argmax(-1).cpu()                 # [H, n_rows]
        return hook
    handles = []
    for li, m in enumerate(attn_mods):
        h = make_hook(li); h.rows = None
        handles.append((h, m.register_forward_hook(h)))
    try:
        for ex in examples[:max_prompts]:
            pairs = answer_and_needle_positions(ex["prompt_ids"], ex["gold_ids"], ex["needle_token_positions"])
            if not pairs:
                continue
            ids = torch.tensor([ex["prompt_ids"] + ex["gold_ids"]], device=device)
            rows = [p for p, _ in pairs]
            for h, _ in handles:
                h.rows = rows
            argmax_store.clear()
            # eager_attention_forward returns the weights to the attention module regardless of output_attentions; the
            # hooks read them layer by layer (keeping all 28 [H,T,T] tensors alive would need ~11 GB at 4K)
            model(input_ids=ids, use_cache=False)
            hit = torch.zeros(L, H, dtype=torch.float64)
            for li in range(L):
                am = argmax_store[li]                       # [H, n_rows]
                for r, (pos, copies) in enumerate(pairs):
                    cset = torch.tensor(copies)
                    hit[li] += (am[:, r].unsqueeze(-1) == cset.unsqueeze(0)).any(-1).double()
            tot += hit / len(pairs)
            n_used += 1
    finally:
        for _, hd in handles:
            hd.remove()
    scores = (tot / max(1, n_used))
    ranking = sorted([(float(scores[l, h]), l, h) for l in range(L) for h in range(H)], reverse=True)
    # `threshold` is the "a head retrieves if score > 0.1" rule of Sec. 6.7.  It used to be stored and printed and
    # never applied, so a layer whose best head scored 0.01 entered PATCHED_LAYERS exactly as one scoring 0.6 did.
    # It is now a REJECTION: layers are taken in ranking order from the heads that clear it, and if fewer than
    # L_patch clear it the result says so rather than quietly filling from the rest.
    above = [r for r in ranking if r[0] > threshold]
    layers, weak = [], []
    for sc, l, h in above:
        if l not in layers:
            layers.append(l)
        if len(layers) == L_patch:
            break
    if len(layers) < L_patch:                     # not enough retrieval layers: fill, but record what was filled
        for sc, l, h in ranking:
            if l not in layers:
                layers.append(l); weak.append(dict(layer=l, head=h, score=sc))
            if len(layers) == L_patch:
                break
    # (l*, h*) is the top-ranked head.  `src = above or ranking` looked like a filter and was not one: `ranking` is sorted
    # descending and `above` is its threshold prefix, so above[0] IS ranking[0] whenever above is non-empty (review
    # e982f83 I).  The guard against naming a head that does not retrieve is `usable` below, which preflight.sh turns
    # into a hard failure -- that is App. H's fallback case, not a run.
    l_star, h_star = ranking[0][1], ranking[0][2]
    return DetectorResult(scores=scores.tolist(), patched_layers=sorted(layers), l_star=l_star, h_star=h_star,
                          n_prompts=n_used, threshold=threshold, ranking=ranking[:64],
                          n_heads_above_threshold=len(above), layers_below_threshold=weak,
                          usable=bool(ranking and ranking[0][0] > threshold))


def uniform_middle_layers(n_layers: int, L_patch: int) -> list:
    """E9 control: L_patch layers evenly spaced through the middle third of the stack."""
    lo, hi = n_layers // 3, 2 * n_layers // 3
    return sorted(set(int(round(x)) for x in torch.linspace(lo, hi, L_patch).tolist()))
