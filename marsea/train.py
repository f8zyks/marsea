"""
Training procedure (records/marsea_training_procedure.md), one recipe for every arm.

  Phase A  steps 0..499   normaliser = SoftmaxNorm in the patched layers (== plain LoRA), G1 only; ONE checkpoint per
                          seed, forked by every arm (B0 continues it).
  Phase B  steps 500..T-1 the arm's normaliser live; fresh modules; b0 / TauK-bias calibration on 8 sequences at
                          the data cursor (not consumed); sanity 5(a)/(b); G2/G3 join the ONE cosine schedule.
  loss     next-token CE on answer tokens only, TOKEN-MEAN over the 16-sequence accumulation window
  optim    AdamW (0.9, 0.95), eps 1e-8; G1 LoRA lr 2e-4 wd 0.01; G2 module weights lr 1e-3 wd 0.01; G3 biases /
           b0 / LN gains lr 1e-3 wd 0; 3 % warm-up then cosine to 10 % at the last step; clip 1.0
  ckpt     every 250 steps: LoRA (peft), arm modules, optimizer, scheduler, cursor, RNG, step, spec version, git hash
  failure  NaN -> restart from the last checkpoint with the identical order; 2nd time halve the head LR; 3rd stop
"""
from __future__ import annotations
import json, math, os, pathlib, random, subprocess, time, dataclasses
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F

from . import SPEC_VERSION
from .backbone import load_backbone, patch_model, normalizers, MarSeaContext, _base_model
from .baselines import make_normalizer
from .normalizer import MarSeaNormalizer
from .heads import inv_softplus

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


@dataclass
class TrainConfig:
    arm: str = "marsea"                       # marsea | B0 | B1 | B2 | B3 | B4 (B5 has no training)
    seed: int = 0
    backbone: str = "Qwen/Qwen2.5-1.5B"
    patched_layers: list = field(default_factory=lambda: [12, 14, 16, 18])   # PROVISIONAL until the detector runs
    L: int = 8192
    total_steps: int = 3500                   # 2500 in the reduced programme (D-24)
    phase_a_steps: int = 500
    accum: int = 16
    lr_lora: float = 2e-4
    lr_modules: float = 1e-3
    warmup_frac: float = 0.03
    final_lr_frac: float = 0.10
    clip: float = 1.0
    rho0: float = 0.05
    eval_every: int = 250
    log_every: int = 50
    ckpt_every: int = 250
    out_dir: str = "runs"
    phase_a_ckpt: Optional[str] = None        # fork from this file (default: runs/phaseA_seed{seed}.pt)
    resume: bool = True
    arm_kwargs: dict = field(default_factory=dict)       # MarSeaNormalizer / MESH knobs (E9)
    head_block: Optional[int] = None            # Q-heads processed at a time (memory / H_block; identical numbers)
    mode: str = "dense"                        # "chunked" for 16K
    init_gap_policy: str = "stop"              # Phase-B init sanity 5(b): "stop" (the procedure) | "record" (the OWNER has
                                               # inspected the stop and decided to continue; written into the README)
    chunk: int = 1024                          # the chunked path's key-chunk width (MarSeaContext.chunk; pod 1 measured
                                               # 2048 with head_block None 17 % faster than 1024 with head_block 2 at 8K)
    checkpoint_layers: str = "patched"         # "patched" | "all"
    device: str = "cuda"
    dry_run_steps: Optional[int] = None        # for smoke tests: stop Phase B after this many steps (NEVER Phase A)
    phase_a_only: bool = False                 # train / verify the Phase-A checkpoint and stop (run_s2.sh's pre-job)
    calib_sequences: int = 8
    calib_subsample: int = 2_000_000           # visible pairs per sequence used for the b0 bisection
    detector_json: Optional[str] = None        # overrides patched_layers when present
    allow_phase_a_layer_change: bool = False   # fork a Phase-A file saved under a different PATCHED_LAYERS (after S0)
    phase_a_patched: bool = True               # Phase A runs the PATCHED SoftmaxNorm layers, i.e. exactly the forward
                                               # Phase B forks into (B0 is Phase A continued, so S2 pays this memory
                                               # anyway).  False runs unpatched SDPA -- cheaper, but a different
                                               # kernel: ~9e-3 relative on bf16 logits, measured and recorded.


def git_hash():
    """HEAD, with "+dirty" when tracked files differ from it: a provenance hash that names a commit the numbers did not
    come from is worse than none."""
    root = str(pathlib.Path(__file__).parents[1])
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True,
                                        stderr=subprocess.DEVNULL).strip()
        return h + ("+dirty" if dirty else "")
    except Exception:
        return "n/a"


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                numpy=np.random.get_state(), python=random.getstate())


def set_rng_state(st):
    torch.set_rng_state(st["torch"])
    if st["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])
    np.random.set_state(st["numpy"]); random.setstate(st["python"])


# --------------------------------------------------------------------------- schedule
def lr_multiplier(step: int, total: int, warmup: int, final_frac: float) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


# --------------------------------------------------------------------------- model assembly
def build_model(cfg: TrainConfig, phase_a: bool):
    from peft import LoraConfig, get_peft_model
    model, tok, facts = load_backbone(cfg.backbone, device=cfg.device)
    d = facts["head_dim"]
    layers = cfg.patched_layers
    if cfg.detector_json and pathlib.Path(cfg.detector_json).exists():
        layers = json.load(open(cfg.detector_json))["patched_layers"]
    # write the RESOLVED layers back: the checkpoint's config is asdict(cfg), and it used to carry the provisional
    # default while the model had been patched from the detector -- so nothing could check that an evaluation's
    # detector is the one the checkpoint trained against (review e982f83 I)
    cfg.patched_layers = [int(l) for l in layers]
    torch.manual_seed(cfg.seed * 7 + 1)
    ctx = MarSeaContext(mode=cfg.mode)
    ctx.chunk = int(cfg.chunk)                 # was MarSeaContext's default 1024 whatever the queue asked (review e982f83 I)
    if phase_a and not cfg.phase_a_patched:
        # Phase A on the UNPATCHED model: the layers are not swapped at all and stock SDPA runs.  No relation either
        # way (ctx.phase_a routes a patched layer to SoftmaxNorm); this only chooses the kernel -- phase_a_kernel_check
        # measures the difference and writes it to the run README.
        model.marsea_ctx = ctx; ctx.patched_layers = []
    else:
        akw = dict(cfg.arm_kwargs)
        if cfg.head_block and cfg.arm.lower() in ("marsea", "b3"):
            akw.setdefault("head_block", cfg.head_block)
        ctx = patch_model(model, layers, lambda l: make_normalizer(cfg.arm, d, **akw).to(cfg.device), ctx,
                          checkpoint_patched=(cfg.checkpoint_layers == "patched"))
    ctx.phase_a = phase_a
    for p in model.parameters():
        p.requires_grad_(False)
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    # LoRA in fp32 master copies; arm modules in fp32
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = p.data.float(); p.requires_grad_(True)
    for l, nm in normalizers(model).items():
        nm.float()
        for p in nm.parameters():
            p.requires_grad_(not phase_a)
    if cfg.checkpoint_layers == "all" or (phase_a and not cfg.phase_a_patched):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.marsea_ctx = ctx
    return model, tok, ctx, layers


def param_groups(model, cfg: TrainConfig, with_modules: bool):
    lora = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    groups = [dict(params=lora, lr=cfg.lr_lora, weight_decay=0.01, name="G1")]
    if with_modules:
        w, o = [], []
        for l, nm in normalizers(model).items():
            if hasattr(nm, "new_module_params"):
                ww, oo = nm.new_module_params(); w += ww; o += oo
        if w: groups.append(dict(params=w, lr=cfg.lr_modules, weight_decay=0.01, name="G2"))
        if o: groups.append(dict(params=o, lr=cfg.lr_modules, weight_decay=0.0, name="G3"))
    return groups


def make_optimizer(model, cfg, with_modules):
    return torch.optim.AdamW(param_groups(model, cfg, with_modules), betas=(0.9, 0.95), eps=1e-8)


def apply_lr(opt, step, cfg, head_lr_scale=1.0):
    m = lr_multiplier(step, cfg.total_steps, int(cfg.warmup_frac * cfg.total_steps), cfg.final_lr_frac)
    for g in opt.param_groups:
        base = cfg.lr_lora if g.get("name") == "G1" else cfg.lr_modules * head_lr_scale
        g["lr"] = base * m
    return m


# --------------------------------------------------------------------------- checkpointing
def module_state(model):
    """the arm's own modules, per patched layer.  Arms without modules (B0/B1/B4's SoftmaxNorm etc.) contribute nothing:
    an empty entry would later be loaded into an arm that DOES have modules and fail on every missing key."""
    if not (getattr(model, "marsea_ctx", None) and model.marsea_ctx.patched_layers):
        return {}
    return {l: sd for l, nm in normalizers(model).items() if len(sd := nm.state_dict()) > 0}


def load_module_state(model, st):
    """Load the arm's modules where the checkpoint has them.  A Phase-A checkpoint carries none (its normaliser is
    SoftmaxNorm), and the spec has the arm's modules created FRESH at the first Phase-B step, so an absent or empty
    entry is the expected case and is skipped rather than loaded."""
    if not st or not model.marsea_ctx.patched_layers:
        return
    for l, nm in normalizers(model).items():
        sd = st.get(l, st.get(str(l)))
        if sd:
            nm.load_state_dict(sd)


def save_checkpoint(path, model, opt, step, cursor, cfg, extra=None):
    from peft import get_peft_model_state_dict
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    # written under a temporary name and renamed: `last.pt` is overwritten in place every 250 steps, and a pod lost
    # DURING that write left an unloadable file that every later relaunch raised on (ops playbook review B).  The
    # rename is atomic on the same filesystem, so the file is either the old checkpoint or the new one, never a mix.
    # ... and DURABLE against node death, not only process death: os.replace publishes the directory entry at once,
    # but nothing forces the temporary file's data out of the page cache first, so a pod that dies in that window can
    # come back with last.pt at the right name and size and unwritten blocks.  fsync the data before the rename and
    # the directory after it; ~52 MB once per 250 steps (ops playbook review 2, item 1).
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        torch.save(dict(lora=get_peft_model_state_dict(model), modules=module_state(model), optimizer=opt.state_dict() if opt else None,
                        step=step, cursor=cursor, rng=rng_state(), spec_version=SPEC_VERSION, git=git_hash(), config=asdict(cfg),
                        detector=detector_stamp(cfg.detector_json), extra=extra or {}), f)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        dfd = os.open(str(path.parent), os.O_RDONLY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    except OSError:                                   # a filesystem that cannot fsync a directory: the data fsync stands
        pass


def unit_cap_record(L: int = 0) -> dict:
    """the unit cap's binding rule in force, recorded in every run README and eval table: E8's frac_cap_binds /
    frac_rows_over_unit read the cap's binding flag and moved by ~50 % at low coverage when the rule changed (review
    e982f83 D-5), so runs on either side of a change are not comparable and this is where a reader finds out."""
    from .normalizer import CAP_TOL
    from .invariants import cap_tolerance, validate_tol_cap_override, tol_cap_ceiling
    if L:
        validate_tol_cap_override(L)          # at STARTUP: an override above L eps / 4 refuses the run here, not 30 h in
    return dict(rule="binds iff excess = sum_{E_i.} (Atil - A_sm) > CAP_TOL (fp64; independent of the softmax kernel's "
                     "row-sum error, review e16a843) AND the projection's own fp32 arithmetic finds mass above the target "
                     "(theta > 0); a binding row is projected onto its own fp64 softmax mass sum_j A_sm_ij, not onto 1 "
                     "(review 7814665 C).  The theta guard's window is a device property measured by check_unit_cap.py "
                     "(runs/unit_cap_device.json: theta_guard_window)", CAP_TOL=CAP_TOL, target="sum_j A_sm_ij (fp64)",
                theta_guard=True,
                # the INV-3 / INV-3a tolerance in force: the default log law, or the measured-window override the gate's
                # failure branch sets (review e6ca44f E) -- recorded so runs under different tolerances are not compared
                tol_cap_at_L=(cap_tolerance(L) if L else None), tol_cap_ceiling_at_L=(tol_cap_ceiling(L) if L else None),
                MARSEA_TOL_CAP=os.environ.get("MARSEA_TOL_CAP") or None)


def detector_stamp(path) -> Optional[dict]:
    """what an evaluation checks its detector against: the file's layers and measurement site (not its hash -- S0
    legitimately adds its licence to the same file after Phase A has been saved)."""
    if not path or not pathlib.Path(path).exists():
        return None
    import hashlib
    raw = pathlib.Path(path).read_bytes(); d = json.loads(raw)
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), patched_layers=d.get("patched_layers"),
                l_star=d.get("l_star"), h_star=d.get("h_star"), spec_version=d.get("spec_version"), created=d.get("created"))


def load_checkpoint(path, model, opt=None):
    from peft import set_peft_model_state_dict
    ck = torch.load(path, map_location="cpu", weights_only=False)
    set_peft_model_state_dict(model, ck["lora"])
    load_module_state(model, ck["modules"])
    if opt is not None and ck.get("optimizer") is not None:
        try:
            opt.load_state_dict(ck["optimizer"])
        except Exception as e:
            print(f"[ckpt] optimizer state not loaded ({e})")
    set_rng_state(ck["rng"])
    return ck


# --------------------------------------------------------------------------- Phase-B initialisation (Sec. 6)
INIT_GAP_BOUND = 0.05


def bias_record(t) -> "float | list":
    """A head's last-layer bias for the README: a float for the shared heads, a list for the per-head arm (E9 per_head:
    TauK / TauQ carry one bias per Q-head, shape [H, 1]).  `.item()` on that tensor killed the per_head arm inside
    phase_b_init the first time it ever reached it (2026-09-19) -- the arm the queue's own comment says "must be run,
    not unit-tested"."""
    v = [float(x) for x in t.detach().flatten().tolist()]
    return v[0] if len(v) == 1 else v


def init_gap_verdict(sanity: dict, policy: str = "stop") -> str:
    """Phase-B init sanity 5(b).  A non-finite live loss always raises.  A gap over the bound raises under "stop" -- the
    procedure: "STOP and inspect, do not train through it" -- and returns "recorded" under "record", which exists for one
    case only: the owner HAS inspected the stop and decided to continue, and the run README says so.  On the first real
    grid (2026-09-18) the bound stopped seed 2 of MarSea and B3 at 5.2 % (seeds 0 and 1: 4.8 %, 1.5 %; every gap
    negative), the key-only ablation at -12.9 % and the uniform-quota ablation at +50.6 %."""
    if policy not in ("stop", "record"):
        raise ValueError(f"init_gap_policy must be 'stop' or 'record', got {policy!r}")
    if not sanity["finite"]:
        raise RuntimeError(f"Phase-B init sanity 5(b) failed: non-finite live loss: {sanity}")
    if sanity["rel_gap_5b"] <= INIT_GAP_BOUND:
        return "ok"
    if policy == "stop":
        raise RuntimeError(f"Phase-B init sanity 5(b) failed: {sanity} -- STOP and inspect, do not train through it")
    return "recorded"


@torch.no_grad()
def phase_b_init(model, ctx: MarSeaContext, data, cursor: int, cfg: TrainConfig, readme: dict):
    """calibrate b0 (bisection to rho0) and TauK's bias (median 1/std) per patched layer on `calib_sequences`
    sequences at the cursor (NOT consumed); sanity checks 5(a)/(b).  Returns the sanity dict."""
    norms = normalizers(model)
    model.eval()
    seqs = [data.get(cursor + i) for i in range(cfg.calib_sequences)]
    mode_saved = ctx.mode; ctx.mode = "dense"                               # calibration captures S/K/Q on the dense path (review C)
    ntok = sum(s.n_label_tokens for s in seqs)
    if not any(isinstance(nm, MarSeaNormalizer) for nm in norms.values()):
        # B2 (MESH) and the arms without modules: record the init and the Phase-A / live losses (no calibration to do)
        ctx.phase_a = True; loss_pa = sum(_answer_loss(model, s, cfg.device) for s in seqs) / max(1, ntok)
        ctx.phase_a = False; loss_live = sum(_answer_loss(model, s, cfg.device) for s in seqs) / max(1, ntok)
        note = ("no calibration: B2's h_a, h_b are fresh with last bias 0, i.e. uniform marginals" if cfg.arm.lower() == "b2"
                else f"no calibration: {cfg.arm} has no modules of its own (Phase B is Phase A continued)")
        readme["phase_b_init"] = dict(arm=cfg.arm, note=note,
                                      sanity=dict(loss_phaseA=loss_pa, loss_live=loss_live, finite=math.isfinite(loss_live)))
        print(f"[phase B init] {json.dumps(readme['phase_b_init'])}")
        ctx.mode = mode_saved; model.train()
        if not math.isfinite(loss_live):
            raise RuntimeError("Phase-B init: non-finite live loss")
        return readme["phase_b_init"]["sanity"]
    # sanity 5(a), first half: the SoftmaxNorm (Phase-A) path on the calibration batch
    ctx.phase_a = True; ctx.force_empty = False
    loss_pa = sum(_answer_loss(model, s, cfg.device) for s in seqs) / max(1, ntok)
    raw_vals = {l: [] for l in norms}; stds = {l: [] for l in norms}
    ctx.phase_a = False; ctx.force_empty = True; ctx.capture_inputs = True; ctx.collect = False
    loss_a = 0.0; ntok = 0
    for s in seqs:
        loss_a += _answer_loss(model, s, cfg.device)
        ntok += s.n_label_tokens
        for l, nm in norms.items():
            S, vis, K, Q = ctx.last_S[l]
            # F-6: the SAME candidate rule as RelationHead.calibrate_b0 -- visible pairs minus the D-31 sink pairs.  The
            # sequences have their own lengths (B = 1, no padding), so the pooled quantity is the 1-D value vector.
            from .relation import candidate_logits
            vals = candidate_logits(nm.relation.raw(K.float(), Q.float()), vis)
            if vals.numel() > cfg.calib_subsample:
                vals = vals[torch.randint(0, vals.numel(), (cfg.calib_subsample,), device=vals.device)]
            raw_vals[l].append(vals.cpu())
            from .heads import column_stats
            st = column_stats(S, vis)[..., 3]
            stds[l].append(st[vis.expand_as(S).sum(-2) > 1].cpu())
    loss_a /= max(1, ntok)
    calib = {}
    for l, nm in norms.items():
        # F-6: b0 through the one bisection RelationHead.calibrate_b0 uses, over the one candidate set (vis minus the
        # D-31 sink pairs).  Counting sink pairs as candidates put realised coverage below rho_0 and wrote a coverage
        # number to the README that the run could not reproduce.
        vals = torch.cat(raw_vals[l]); sd = torch.cat(stds[l])
        nm.relation.calibrate_b0(None, None, None, rho0=cfg.rho0, iters=30, raw=vals)
        target = nm.tauK.calibrate(None, None, stds=sd)
        nm.tauQ.set_init_tau(1.0)
        calib[l] = dict(b0=float(nm.relation.b0), coverage=float((vals + nm.relation.b0.item() > 0).float().mean()),
                        n_candidates=int(vals.numel()), tauK_target=float(target),
                        tauK_bias=bias_record(nm.tauK.last_bias), tauQ_bias=bias_record(nm.tauQ.last_bias))
    ctx.capture_inputs = False
    # sanity 5(a): E forced empty reproduces Phase A's loss; 5(b): live loss within 5 %, no NaN
    loss_live = 0.0; cov = {l: [] for l in norms}
    ctx.force_empty = False; ctx.collect = True
    for s in seqs:
        loss_live += _answer_loss(model, s, cfg.device)
        for l in norms:
            e8 = ctx.diags[l].extra.get("e8")
            if e8 and "rho" in e8: cov[l].append(e8["rho"])
    loss_live /= max(1, ntok)
    ctx.collect = False
    sanity = dict(loss_phaseA=loss_pa, loss_forced_empty=loss_a, gap_5a=abs(loss_a - loss_pa),
                  loss_live=loss_live, rel_gap_5b=abs(loss_live - loss_a) / max(1e-8, abs(loss_a)),
                  signed_rel_gap_5b=(loss_live - loss_a) / max(1e-8, abs(loss_a)),     # negative: the live relation LOWERS the loss
                  finite=math.isfinite(loss_live), coverage_live={l: float(np.mean(v)) if v else None for l, v in cov.items()})
    readme["phase_b_init"] = dict(calibration=calib, sanity=sanity)
    print(f"[phase B init] calibration {json.dumps(calib)}\n[phase B init] sanity {json.dumps(sanity)}")
    ctx.mode = mode_saved
    if sanity["gap_5a"] > 1e-4 * max(1.0, abs(loss_pa)):                     # 5(a): E forced empty == Phase A's path (INV-9 at model level)
        raise RuntimeError(f"Phase-B init sanity 5(a) failed: forced-empty loss {loss_a} vs Phase-A path {loss_pa}")
    verdict = init_gap_verdict(sanity, cfg.init_gap_policy)                  # 5(b): live within 5 %, no NaN
    if verdict == "recorded":
        ev = dict(step=cfg.phase_a_steps, event="init sanity 5(b) exceeded; continued by the owner's decision (init_gap_policy=record)",
                  rel_gap_5b=sanity["rel_gap_5b"], signed_rel_gap_5b=sanity["signed_rel_gap_5b"], bound=INIT_GAP_BOUND)
        readme.setdefault("events", []).append(ev)
        print(f"[phase B init] 5(b) EXCEEDED ({sanity['signed_rel_gap_5b']:+.4f} against +-{INIT_GAP_BOUND}); continuing: init_gap_policy=record")
    model.train()
    return sanity


# --------------------------------------------------------------------------- E8 summary hook
def e8_summary(ctx: MarSeaContext) -> dict:
    """the per-layer E8 summaries.  Read from the snapshot train_step took after micro-batch 0 (ctx.diags itself is
    cleared by the model's forward pre-hook on every forward -- F-2); falls back to ctx.diags for single-forward callers."""
    snap = getattr(ctx, "extra_log", {}).get("e8")
    if snap:
        return snap
    out = {l: dg.extra["e8"] for l, dg in ctx.diags.items() if dg is not None and "e8" in dg.extra}
    # e8_missing is the field that makes E8's ABSENCE audible; it was written at two sites and read at none, so the
    # fix for a silent loss wrote to somewhere nobody opens (review 30372ae E).  It goes into the log beside the rest.
    for l, dg in ctx.diags.items():
        if dg is not None and "e8" not in dg.extra and "e8_missing" in dg.extra:
            out[l] = {"missing": dg.extra["e8_missing"]}
    return out


class NaNError(RuntimeError):
    pass


@torch.no_grad()
def warmup_calibrate(model, ctx, T: int = 1024, rho0: float = 0.05, device="cuda") -> dict:
    """Put the relation at its operating point (rho_0 coverage) before profiling memory or step time.  A fresh head has
    b0 = 0, i.e. about half of all visible pairs in the relation -- an operating point no run has, and one whose |E| the
    chunked path pays for directly (its footprint is proportional to coverage)."""
    from .backbone import normalizers
    from .normalizer import MarSeaNormalizer
    ids = torch.randint(0, 1000, (1, T), device=device)
    saved_mode, ctx.mode = ctx.mode, "dense"                     # the calibration reads S/K/Q, captured on the dense path
    ctx.capture_inputs = True; ctx.collect = True
    model.eval(); model(input_ids=ids, use_cache=False)
    out = {}
    for l, nm in normalizers(model).items():
        if isinstance(nm, MarSeaNormalizer) and l in ctx.last_S:
            S_l, vis_l, K_l, Q_l = ctx.last_S[l]
            nm.relation.calibrate_b0(K_l.float(), Q_l.float(), vis_l, rho0)
            nm.tauK.calibrate(S_l, vis_l); nm.tauQ.set_init_tau(1.0)
            out[l] = float(nm.relation.b0)
    ctx.capture_inputs = False; ctx.collect = False; ctx.mode = saved_mode
    ctx.last_S = {}; ctx.diags = {}
    model.train()
    return out


@torch.no_grad()
def phase_a_kernel_check(model, ctx, data, cfg, layers, n_seq: int = 2, tol: float = 1e-4) -> dict:
    """Phase A runs the UNPATCHED model by default (cfg.phase_a_patched = False) because SDPA carries 8K where the dense
    eager path does not.  "Every arm forks the same Phase A" then rests on unpatched SDPA == the patched SoftmaxNorm arm
    -- a comparison T0 never makes (T0 compares the patched arm with the EAGER twin).  Measure it here, once, and write
    the number into the run README (read-through 2026-09-11, Sec. 3)."""
    if cfg.phase_a_patched or not layers:
        return dict(skipped="Phase A is running the patched SoftmaxNorm layers")
    from .backbone import patch_model
    from .baselines import SoftmaxNorm
    model.eval()
    seqs = [data.get(i) for i in range(n_seq)]
    base = [model(input_ids=s.input_ids.to(cfg.device), use_cache=False).logits.float() for s in seqs]
    saved = {l: model.get_decoder().layers[l].self_attn if hasattr(model, "get_decoder") else None for l in layers}
    ctx2 = patch_model(model, layers, lambda l: SoftmaxNorm())
    ctx2.phase_a = True
    rel = 0.0
    for s, ref in zip(seqs, base):
        out = model(input_ids=s.input_ids.to(cfg.device), use_cache=False).logits.float()
        rel = max(rel, float((out - ref).abs().max() / ref.abs().max().clamp_min(1e-6)))
    dec = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    for l, mod in saved.items():                                           # restore the unpatched modules
        if mod is not None:
            dec.layers[l].self_attn = mod
    if getattr(dec, "_marsea_hook", None) is not None:
        dec._marsea_hook.remove(); dec._marsea_hook = None
    model.marsea_ctx = ctx
    model.train()
    res = dict(max_rel_logit_diff=rel, tol=tol, within_tol=bool(rel <= tol), n_sequences=n_seq,
               note="unpatched SDPA (what this Phase A trains) vs the patched SoftmaxNorm arm (what Phase B forks into); "
                    "bf16 kernels differ at ~1e-2 relative on logits, so this is a recorded quantity, not a gate")
    print(f"[phase A] kernel check: max rel logit diff {rel:.2e} (tol {tol:g}) -- "
          f"{'within tolerance' if res['within_tol'] else 'ABOVE tolerance: Phase A trained under a different kernel'}")
    if not res["within_tol"]:
        print("[phase A] WARNING: --phase_a_patched is the default for S2 precisely so that the warm start is the forward "
              "every arm forks into; this run opted out and the delta above is recorded in the run README.")
    return res


def anneal_gate_temperature(model, step, cfg):
    """E9 hard-concrete gate: temperature 1.0 -> 0.2 linearly over Phase B (spec Sec. 4.3)."""
    frac = min(1.0, max(0.0, (step - cfg.phase_a_steps) / max(1, cfg.total_steps - cfg.phase_a_steps)))
    for nm in normalizers(model).values():
        if getattr(nm, "gate", "st") == "hard_concrete":
            nm.gate_temperature = 1.0 + (0.2 - 1.0) * frac


@torch.no_grad()
def _answer_loss(model, s, device) -> float:
    labels = s.labels.to(device)
    keep = torch.nonzero(labels[0, 1:] != -100).flatten()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=s.input_ids.to(device), use_cache=False, logits_to_keep=keep)
    return float(F.cross_entropy(out.logits[0].float(), labels[0, keep + 1], reduction="sum"))


# --------------------------------------------------------------------------- the loop
def train(cfg: TrainConfig, data, quick_eval=None):
    """data: a MixedDataset (data.get(cursor) -> Sequence).  quick_eval: callable(model, ctx, step) -> dict."""
    set_seed(cfg.seed)
    out = pathlib.Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    run_dir = out / f"{cfg.arm}_seed{cfg.seed}"; run_dir.mkdir(exist_ok=True)
    readme_path = out / f"{cfg.arm}_seed{cfg.seed}" / "README.json"
    if cfg.resume and readme_path.exists():
        readme = json.load(open(readme_path)); readme.setdefault("events", []).append(dict(event="resumed", time=time.time()))
    else:
        readme = dict(config=asdict(cfg), spec_version=SPEC_VERSION, git=git_hash(), events=[],
                      unit_cap=unit_cap_record(cfg.L))
    phase_a_path = pathlib.Path(cfg.phase_a_ckpt or (out / f"phaseA_seed{cfg.seed}.pt"))
    log_f = open(run_dir / "train_log.jsonl", "a")
    dev = cfg.device

    def log(rec):
        rec["time"] = time.time(); log_f.write(json.dumps(rec) + "\n"); log_f.flush()

    # ---------------- Phase A (once per seed)
    if not phase_a_path.exists():
        print(f"[phase A] training {phase_a_path}")
        model, tok, ctx, layers = build_model(cfg, phase_a=True)
        readme["phase_a_kernel_check"] = phase_a_kernel_check(model, ctx, data, cfg, layers)
        opt = make_optimizer(model, cfg, with_modules=False)
        cursor = 0
        model.train()
        steps_done = 0
        for step in range(cfg.phase_a_steps):                            # the dry-run guard never applies to Phase A (A1)
            cursor, loss = train_step(model, ctx, opt, data, cursor, step, cfg, log)
            steps_done += 1
            if step % cfg.log_every == 0: print(f"[A] step {step} loss {loss:.4f}")
        if steps_done != cfg.phase_a_steps:
            raise RuntimeError(f"Phase A ran {steps_done} of {cfg.phase_a_steps} steps; refusing to save an incomplete warm start")
        save_checkpoint(phase_a_path, model, opt, cfg.phase_a_steps, cursor, cfg, extra={"phase": "A", "steps": steps_done})
        readme["events"].append(dict(step=cfg.phase_a_steps, event="phase A saved", path=str(phase_a_path)))
        del model, opt; torch.cuda.empty_cache()
    else:
        ckA = torch.load(phase_a_path, map_location="cpu", weights_only=False)
        assert ckA.get("step") == cfg.phase_a_steps and ckA.get("extra", {}).get("phase") == "A", f"{phase_a_path} is not a complete Phase-A checkpoint"
    if cfg.phase_a_only:
        # the README used to be written only after phase_b_init, and under the runbook Phase A ONLY runs here -- so no
        # README in the programme carried phase_a_kernel_check (review 7814665 F)
        (run_dir / "README.json").write_text(json.dumps(readme, indent=1, default=str))
        print(f"[phase A] complete: {phase_a_path}"); log_f.close(); return
    if "ckA" in locals():
            # S0 legitimately changes PATCHED_LAYERS (the induction layer joins, the weakest retrieval layer leaves) AFTER
            # Phase A was saved.  That does not corrupt the fork -- the checkpoint is LoRA on every layer and the arm modules
            # start fresh -- but Phase A then adapted under a different kernel mix than the one Phase B runs, and the README's
            # phase_a_kernel_check describes a layer set Phase A never used.  Say so, and make continuing a decision
            # (review e16a843 J: provenance, not repair; retraining Phase A is ~6 GPU-h).
            layers_now = (json.load(open(cfg.detector_json))["patched_layers"]
                          if cfg.detector_json and pathlib.Path(cfg.detector_json).exists() else cfg.patched_layers)
            layers_A = (ckA.get("detector") or {}).get("patched_layers")
            if layers_A is None:
                print(f"[phase B] WARNING: {phase_a_path} carries no detector stamp; its patched layers are unknown")
            elif [int(x) for x in layers_A] != [int(x) for x in layers_now]:
                msg = (f"{phase_a_path} was trained with patched layers {layers_A}; Phase B would patch {list(layers_now)} "
                       f"(S0 changed PATCHED_LAYERS).  Retrain Phase A (delete the file), or pass --allow_phase_a_layer_change.")
                if not cfg.allow_phase_a_layer_change:
                    raise RuntimeError(msg)
                print(f"[phase B] NOTE: {msg}  -- continuing as allowed")
                readme["events"].append(dict(event="phase A layer set differs", phase_a_layers=layers_A,
                                             phase_b_layers=list(layers_now), allowed=True))

    if cfg.arm.lower() == "b5":
        print("B5 has no training (evaluation-only on B0's weights, D-29)"); return
    # ---------------- Phase B
    model, tok, ctx, layers = build_model(cfg, phase_a=False)
    ck = load_checkpoint(phase_a_path, model)
    cursor = ck["cursor"]; step0 = cfg.phase_a_steps
    opt = make_optimizer(model, cfg, with_modules=False)
    if ck.get("optimizer"):
        try: opt.load_state_dict(ck["optimizer"])
        except Exception as e: print(f"[phase B] G1 optimizer state not restored: {e}")
    if cfg.arm.lower() == "b0":
        ctx.phase_a = True                                  # B0 IS Phase A continued
    last_ckpt = None
    resume_path = run_dir / "last.pt"
    head_lr_scale = 1.0; nan_count = {}
    if cfg.resume and resume_path.exists():
        ck2 = load_checkpoint(resume_path, model)
        step0 = ck2["step"]; cursor = ck2["cursor"]; head_lr_scale = ck2["extra"].get("head_lr_scale", 1.0)
        opt = make_optimizer(model, cfg, with_modules=True)
        try: opt.load_state_dict(ck2["optimizer"])
        except Exception as e:
            # a resume without its Adam moments is a different recipe from the arms that ran straight through, and it
            # used to be a line on stdout only (ops playbook review C): recorded in the README so collect.py's reader
            # and the run record can see it, and written now rather than at the end of the run
            print(f"[resume] optimizer state not restored: {e}")
            readme["events"].append(dict(step=step0, event="optimizer state not restored on resume", detail=str(e)))
        (run_dir / "README.json").write_text(json.dumps(readme, indent=1, default=str))     # carries the "resumed" event too
        print(f"[phase B] resumed at step {step0}, cursor {cursor}")
        last_ckpt = resume_path
    else:
        torch.manual_seed(cfg.seed * 7 + 1)                 # fresh, seeded modules (created at patch time)
        ctx.phase_a = (cfg.arm.lower() == "b0")
        phase_b_init(model, ctx, data, cursor, cfg, readme)
        for g in param_groups(model, cfg, with_modules=True)[1:]:
            opt.add_param_group(g)                          # G2/G3 join with fresh Adam moments (C-24)
        (run_dir / "README.json").write_text(json.dumps(readme, indent=1, default=str))
        # B1: a NaN before the first Phase-B checkpoint must rewind to THIS state (modules + G2/G3 moments), not to Phase A
        save_checkpoint(resume_path, model, opt, step0, cursor, cfg, extra={"head_lr_scale": head_lr_scale, "phase": "B-init"})
        last_ckpt = resume_path
    model.train()
    step = step0
    while step < cfg.total_steps:
        if cfg.dry_run_steps is not None and step >= step0 + cfg.dry_run_steps: break
        try:
            cursor_new, loss = train_step(model, ctx, opt, data, cursor, step, cfg, log, head_lr_scale)
        except NaNError as e:
            key = step; nan_count[key] = nan_count.get(key, 0) + 1
            readme["events"].append(dict(step=step, event="nan", count=nan_count[key], detail=str(e)))
            print(f"[NaN] at step {step} ({nan_count[key]}): {e}")
            if nan_count[key] >= 3:
                readme["events"].append(dict(step=step, event="stopped: third NaN"))
                (run_dir / "README.json").write_text(json.dumps(readme, indent=1, default=str))
                raise
            if nan_count[key] == 2:
                head_lr_scale *= 0.5; readme["events"].append(dict(step=step, event="head lr halved", scale=head_lr_scale))
            if last_ckpt is not None:
                ck3 = load_checkpoint(last_ckpt, model, opt); step = ck3["step"]; cursor = ck3["cursor"]
            else:
                ck3 = load_checkpoint(phase_a_path, model); cursor = ck3["cursor"]; step = cfg.phase_a_steps
            opt.zero_grad(set_to_none=True)
            continue
        cursor = cursor_new
        anneal_gate_temperature(model, step, cfg)
        if step % cfg.log_every == 0:
            e8 = e8_summary(ctx)
            b0s = {l: float(nm.relation.b0) for l, nm in normalizers(model).items() if hasattr(nm, "relation")}
            log(dict(step=step, loss=loss, e8=e8, head_lr_scale=head_lr_scale, grad_norms=ctx.extra_log.get("grad_norms"),
                     nonfinite_grads=ctx.extra_log.get("nonfinite_grads"), b0=b0s))
            miss = [l for l, v in e8.items() if "missing" in v]
            print(f"[B] step {step} loss {loss:.4f} "
                  + (f"rho {[round(v['rho'], 4) for v in e8.values() if 'rho' in v]}" if e8 else "")
                  + (f"  E8 MISSING on layers {miss}" if miss else ""))
        step += 1
        if step % cfg.ckpt_every == 0 or step == cfg.total_steps:
            save_checkpoint(resume_path, model, opt, step, cursor, cfg, extra={"head_lr_scale": head_lr_scale, "phase": "B"})
            save_checkpoint(run_dir / f"step{step}.pt", model, opt, step, cursor, cfg, extra={"head_lr_scale": head_lr_scale})
            for old in sorted(run_dir.glob("step*.pt"), key=lambda q: int(q.stem[4:]))[:-2]:
                old.unlink()
            last_ckpt = resume_path
            if quick_eval is not None:
                model.eval()
                res = quick_eval(model, ctx, step); log(dict(step=step, eval=res)); print(f"[eval] step {step}: {res}")
                model.train()
    save_checkpoint(run_dir / "final.pt", model, opt, step, cursor, cfg, extra={"head_lr_scale": head_lr_scale})
    (run_dir / "README.json").write_text(json.dumps(readme, indent=1, default=str))
    log_f.close()
    return model, tok, ctx


def train_step(model, ctx, opt, data, cursor, step, cfg, log, head_lr_scale=1.0):
    dev = cfg.device
    opt.zero_grad(set_to_none=True)
    seqs = [data.get(cursor + i) for i in range(cfg.accum)]
    tokens_in_window = sum(s.n_label_tokens for s in seqs)
    ctx.collect = (step % cfg.log_every == 0)
    loss_sum = 0.0
    for micro, s in enumerate(seqs):
        ids = s.input_ids.to(dev); labels = s.labels.to(dev)
        keep = torch.nonzero(labels[0, 1:] != -100).flatten()                   # positions that PREDICT a labelled token
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # no KV cache (a checkpointed recompute must not append to it); only the answer positions' logits are built
            logits = model(input_ids=ids, use_cache=False, logits_to_keep=keep).logits          # [1, n_keep, V]
        loss_tok = F.cross_entropy(logits[0].float(), labels[0, keep + 1], reduction="sum")
        if not torch.isfinite(loss_tok):
            raise NaNError(f"non-finite loss at step {step} micro {micro}")
        # backward must NOT run inside the autocast block: PyTorch dispatches autocast-eligible ops in the backward
        # (and in any checkpointed recompute) under the ambient autocast state, which would run the programs' gradients
        # in bf16 -- measured at ~3e-2 relative against ~5e-7 for the correct ordering (review 2026-09-12 follow-up)
        assert not torch.is_autocast_enabled(), "backward under autocast: the programs' gradients would run in bf16"
        (loss_tok / tokens_in_window).backward()
        loss_sum += float(loss_tok.detach())
        if ctx.collect:
            # F-2 (read-through 2026-09-11): the model's forward pre-hook calls ctx.reset_forward on EVERY forward, which
            # clears ctx.diags -- so micro-batches 1..15 wipe what micro 0 collected and e8_summary() saw {}.  Snapshot the
            # per-layer E8 payloads here; that is the whole of E8 as an experiment (training procedure Sec. 8.1).
            if not hasattr(ctx, "extra_log"): ctx.extra_log = {}
            ctx.extra_log["e8"] = {l: (dg.extra["e8"] if "e8" in dg.extra else {"missing": dg.extra["e8_missing"]})
                                   for l, dg in ctx.diags.items()
                                   if dg is not None and ("e8" in dg.extra or "e8_missing" in dg.extra)}
        ctx.collect = False                                   # E8 from the first micro-batch of the window
    params = [p for g in opt.param_groups for p in g["params"]]
    with torch.no_grad():                                                     # B8: per-group norms BEFORE clipping; NaN/inf counts
        norms = {}; nonfinite = 0
        for g in opt.param_groups:
            gs = [p.grad for p in g["params"] if p.grad is not None]
            norms[g.get("name", "?")] = float(torch.norm(torch.stack([x.norm() for x in gs]))) if gs else 0.0
            nonfinite += sum(int((~torch.isfinite(x)).sum()) for x in gs)
        if not hasattr(ctx, "extra_log"): ctx.extra_log = {}
        ctx.extra_log["grad_norms"] = norms; ctx.extra_log["nonfinite_grads"] = nonfinite
    gn = torch.nn.utils.clip_grad_norm_(params, cfg.clip)
    if not torch.isfinite(gn):
        raise NaNError(f"non-finite gradient norm at step {step}")
    apply_lr(opt, step, cfg, head_lr_scale)
    opt.step()
    return cursor + cfg.accum, loss_sum / max(1, tokens_in_window)
