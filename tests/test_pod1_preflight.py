"""Pod 1 (1 x H200, 2026-09-18): preflight's chunked evaluation probe failed its memory gate on `teacher` at 16K -- every
head of l* kept, a pass no 16K job runs -- while `teacher_head`, the pass every E3 job runs, measured 54 GB.  The gate
is now a function over the gated keys, the chunked probe gates teacher_head, and B5's pass 1 (the one all-heads pass in
the queue, dense at 8K) is measured and gated where it runs."""
import pathlib, re, sys
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from profile_memory import gate_verdict, _key  # noqa: E402

# exactly what pod 1 measured (runs/preflight_eval_chunked.json, 2026-09-18 02:20)
POD1_POINTS = {
    "4096": {"teacher": {"peak_GB": 11.907, "retained_GB": 6.225}, "teacher_head": {"peak_GB": 8.013, "retained_GB": 2.141}},
    "8192": {"teacher": {"peak_GB": 40.05, "retained_GB": 26.212}, "teacher_head": {"peak_GB": 23.917, "retained_GB": 8.969}},
    "16384": {"teacher": {"oom": True}, "teacher_head": {"peak_GB": 53.954, "retained_GB": 41.042}},
}
POD1_EXTRAP = {
    "teacher": {"8192": 40.0, "16384": 156.4, "exponent_p": 2.05, "n_points": 2, "r2": None},
    "teacher_head": {"8192": 20.5, "16384": 58.7, "exponent_p": 1.66, "n_points": 3, "r2": 0.9827},
}


def test_pod1_verdict_reproduced_ungated_and_passes_on_the_pass_the_queue_runs():
    g = gate_verdict(POD1_POINTS, POD1_EXTRAP, [8192, 16384], 72.0)
    assert g["passed"] is False and g["why"] == "OOM at T = 16384 (teacher)" and g["worst_GB"] == 156.4
    g = gate_verdict(POD1_POINTS, POD1_EXTRAP, [8192, 16384], 72.0, ["teacher_head"])
    assert g["passed"] is True and g["worst_GB"] == 58.7 and g["why"] == "extrapolated teacher_head at 16384"
    assert g["keys"] == ["teacher_head"]


def test_gate_still_fails_on_the_numbers_and_on_a_key_the_probe_never_ran():
    g = gate_verdict(POD1_POINTS, POD1_EXTRAP, [8192, 16384], 50.0, ["teacher_head"])
    assert g["passed"] is False and g["why"].startswith("extrapolated") and g["worst_GB"] == 58.7
    g = gate_verdict(POD1_POINTS, POD1_EXTRAP, [8192], 72.0, ["teacher_head", "teacher_b5"])
    assert g["passed"] is False and "never measured" in g["why"] and "teacher_b5" in g["why"]
    # an OOM in a gated key wins over any number
    pts = {"8192": {"teacher_b5": {"oom": True}, "teacher_head": {"peak_GB": 1.0}}}
    g = gate_verdict(pts, {}, [8192], 72.0, ["teacher_head", "teacher_b5"])
    assert g["passed"] is False and g["why"] == "OOM at T = 8192 (teacher_b5)"
    assert isinstance(g["passed"], bool)


def test_keys():
    assert _key("teacher", False) == "teacher" and _key("teacher", True) == "teacher_head"
    assert _key("teacher", "b5") == "teacher_b5" and _key("train", False) == "train"


def test_preflight_gates_each_probe_on_what_the_queue_runs_there():
    pf = (ROOT / "scripts/preflight.sh").read_text()
    probes = [ln for ln in pf.split("\n$PY ") if "profile_memory.py" in ln]
    chunked = [ln for ln in probes if "preflight_eval_chunked.json" in ln][0]
    dense = [ln for ln in probes if "preflight_eval_dense.json" in ln][0]
    assert "--targets 8192 16384" in chunked and "--gate_keys teacher_head" in chunked and "--b5" not in chunked
    assert "--b5" in dense and re.search(r"--gate_keys teacher_head teacher_b5", dense) and "--targets 8192 " in dense
    # the B5 job in the eval queue is dense (run_eval's default) at 8K, teacher-forced only -- what the dense probe gates
    es = (ROOT / "scripts/run_evalsuite.sh").read_text()
    b5 = [ln for ln in es.splitlines() if "--arm B5" in ln][0]
    assert "--mode chunked" not in b5 and "E2_L8192" in es.split("--arm B5")[1].split("\n")[1]


def test_profile_memory_b5_branch_matches_b5_two_pass():
    pm = (ROOT / "scripts/profile_memory.py").read_text()
    ev = (ROOT / "marsea/evaluate.py").read_text()
    b5 = ev.split("def b5_two_pass")[1].split("def ")[0]
    assert "ctx_m.keep_dense = True" in b5 and 'fields=("E", "supp_rel")' in b5
    branch = pm.split('if mode == "teacher" and sub == "b5":')[1].split("elif mode")[0]
    assert "ctx.keep_dense = True" in branch and 'fields=("E", "supp_rel")' in branch and "ctx.keep_dense = False" in branch
    assert "sites=" not in branch, "B5's pass 1 keeps every head; a sites= argument would measure something else"


def test_gate_gb_is_the_cards_memory_with_headroom_not_the_h100s_literal():
    """pod 1 measured the paired evaluation at 86 GB at 16K: a pass on the 143 GB H200, a failure against 72 -- the
    80 GB card's number, hard-coded."""
    pf = (ROOT / "scripts/preflight.sh").read_text()
    line = [ln for ln in pf.splitlines() if ln.startswith("GATE_GB=")][0]
    assert "GATE_GB:-72" not in line and "get_device_properties" in line and "0.9 *" in line and "min(" in line
    assert pf.index("PY=${PY:-") < pf.index(line), "the derivation runs $PY, which must be defined first"
    assert 'echo "GATE_GB=$GATE_GB' in pf, "the ceiling in force must be in the log"
    # the arithmetic the line does, on the two cards that matter
    for mib, expect in ((81559, 71), (143771, 126), (81920, 72)):
        assert int(0.9 * mib * 2**20 / 2**30) == expect


def test_the_training_configuration_is_one_set_of_knobs_read_by_preflight_and_the_queue():
    """Pod 1 measured the 8K recipe at 7.6 s/sequence (D-23 ceiling 1.2) and the levers at 4K; whichever is taken,
    preflight must measure the configuration run_s2.sh runs: the same L, CHUNK and HEAD_BLOCK, and CHUNK must reach
    the context (it was set in run_s2.sh and passed nowhere -- review e982f83 I)."""
    from marsea.train import TrainConfig
    assert TrainConfig().chunk == 1024
    tr = (ROOT / "marsea/train.py").read_text()
    assert "ctx.chunk = int(cfg.chunk)" in tr
    rt = (ROOT / "scripts/run_train.py").read_text()
    assert '"--chunk"' in rt and "chunk=args.chunk" in rt and "head_block=(args.head_block or None)" in rt
    s2 = (ROOT / "scripts/run_s2.sh").read_text()
    common = s2.split("COMMON=")[1].split("\n")[0]
    assert "--L $L" in common and "--head_block $HEAD_BLOCK" in common and "--chunk $CHUNK" in common
    assert "UNIFORM_L=${UNIFORM_L:-$L}" in s2, "the dense E9 arms must follow the grid's length"
    pf = (ROOT / "scripts/preflight.sh").read_text()
    assert "L=${L:-8192}; CHUNK=${CHUNK:-1024}; HEAD_BLOCK=${HEAD_BLOCK:-2}" in pf
    ts = [ln for ln in pf.split("\n$PY ") if "time_step.py" in ln and "--mode chunked" in ln][0]
    assert "--L $L" in ts and "--chunk $CHUNK" in ts and "--head_block $HEAD_BLOCK" in ts
    tp = [ln for ln in pf.split("\n$PY ") if "preflight_train_chunked.json" in ln][0]
    assert "--lengths 2048 4096 $L" in tp and "--targets $L" in tp and "--chunk $CHUNK" in tp
    # the evaluation probes are NOT retargeted: evaluation runs at 8K and 16K whatever the training length, with ITS
    # OWN head blocking (pod 1 run 3: unblocked, the 16K sites-only pass OOMs; run_evalsuite.sh reads EVAL_HEAD_BLOCK)
    for name in ("preflight_eval_chunked.json", "preflight_eval_dense.json", "preflight_eval_paired.json"):
        ev = [ln for ln in pf.split("\n$PY ") if name in ln][0]
        assert "--head_block $EVAL_HEAD_BLOCK" in ev and "$HEAD_BLOCK " not in ev and "--chunk $CHUNK" not in ev, name
    ev = [ln for ln in pf.split("\n$PY ") if "preflight_eval_chunked.json" in ln][0]
    assert "--lengths 4096 8192 16384" in ev and "--targets 8192 16384" in ev
    assert "EVAL_HEAD_BLOCK=${EVAL_HEAD_BLOCK:-2}" in pf
    # the training default is defined ONCE, at the top, and every probe reads $HEAD_BLOCK (8887dff had replaced the
    # definition itself, so an unset HEAD_BLOCK became an empty --head_block argument)
    assert pf.count("${HEAD_BLOCK:-2}") == 1 and "HEAD_BLOCK=${HEAD_BLOCK:-2}" in pf
    es = (ROOT / "scripts/run_evalsuite.sh").read_text()
    assert "HEAD_BLOCK=${EVAL_HEAD_BLOCK:-2}" in es and "HEAD_BLOCK=${HEAD_BLOCK:-2}" not in es
    from conftest import RUN_KNOBS
    assert "EVAL_HEAD_BLOCK" in RUN_KNOBS
    pm = (ROOT / "scripts/profile_memory.py").read_text()
    assert "for T in sorted(set(args.lengths))" in pm


def test_the_suite_does_not_read_the_operators_run_knobs():
    """preflight step 2 runs pytest in the shell that has L / CHUNK / HEAD_BLOCK exported for the queue (pod 1: four
    script tests failed on them).  conftest strips the knobs from the process environment before any test runs."""
    import os
    from conftest import RUN_KNOBS
    for k in ("L", "CHUNK", "HEAD_BLOCK", "STEPS", "MARSEA_TOL_CAP", "GATE_GB"):
        assert k in RUN_KNOBS and k not in os.environ


def test_generate_returns_an_uploaded_set_without_touching_the_generator(tmp_path, monkeypatch):
    """Pod 1, preflight step 6: run_detector -> generate() called ensure_patched() before checking that the DET set
    already existed, and the pod has no third_party/RULER by design (the data is uploaded)."""
    import marsea.data.ruler as R
    cfg = R.NIAHConfig("DET", 4096, 1, 1, 1, type_needle_v="words", seed=0, num_samples=200, gold_depth=None)
    assert cfg.save_name() == "DET_L4096_K1_V1_Q1_drand_s0", cfg.save_name()   # the uploaded set's name
    d = tmp_path / cfg.save_name(); d.mkdir(); (d / "validation.jsonl").write_text("{}\n")
    monkeypatch.setattr(R, "NIAH", tmp_path / "no_such_dir" / "niah.py")
    monkeypatch.setattr(R, "ensure_patched", lambda: (_ for _ in ()).throw(AssertionError("generator touched")))
    assert R.generate(cfg, "Qwen/Qwen2.5-1.5B", tmp_path) == (d / "validation.jsonl").resolve()
    body = (ROOT / "marsea/data/ruler.py").read_text().split("def generate(")[1].split('"""')[2]   # after the docstring
    assert body.index("if target.exists() and not force:") < body.index("ensure_patched()")


def test_the_queues_data_globs_follow_the_training_length():
    """Pod 1: the 4K lever needs TRAIN_L4096_* beside TRAIN_L8192_*; with the length-blind default glob an 8K run would
    silently train on a mixture of lengths, and a 4K run saw 0/27600 sequences <= L (mix.py never truncates)."""
    s2 = (ROOT / "scripts/run_s2.sh").read_text()
    assert 'DATA_RULER=${DATA_RULER:-"data/ruler/TRAIN_L${L}_*"}' in s2
    assert 'EVAL=${EVAL:-"data/ruler/QUICK_L${L}_*/validation.jsonl"}' in s2
    assert s2.index("L=${L:-8192}") < s2.index("DATA_RULER=${DATA_RULER:-"), "L must be defined before the globs use it"
    # the mixer refuses rather than truncates: the failure mode that stopped Phase A
    mix = (ROOT / "marsea/data/mix.py").read_text()
    assert "sequences are dropped, never truncated" in mix


def test_run_train_expands_a_quoted_eval_glob():
    """S1 on pod 1 was typed with --eval_ruler "data/ruler/QUICK_L4096_*/validation.jsonl" (quoted): the shell did not
    expand it, run_train.py did not either, and the run died on the literal pattern before building the model."""
    rt = (ROOT / "scripts/run_train.py").read_text()
    assert "glob.has_magic(pat)" in rt and "matches no file" in rt
    assert "for f in args.eval_ruler for r in load_jsonl(f)" not in rt


def test_per_head_biases_are_recorded_as_lists_not_item():
    """The per_head E9 arm died in phase_b_init on its first real run: TauK.last_bias is [H, 1] there and the README
    record called .item() on it."""
    import torch
    from marsea.train import bias_record
    from marsea.heads import TauK, TauQ
    assert bias_record(torch.tensor([0.25])) == 0.25
    assert bias_record(torch.zeros(12, 1)) == [0.0] * 12
    for cls in (TauK, TauQ):
        shared, per = cls(8), cls(8, per_head=12)
        assert isinstance(bias_record(shared.last_bias), float)
        rec = bias_record(per.last_bias); assert isinstance(rec, list) and len(rec) == 12
    src = (ROOT / "marsea/train.py").read_text()
    assert "last_bias.item()" not in src


def test_generation_prefill_gets_E_under_head_blocking():
    """Eval smoke test, 2026-09-19: dense-path generation with --head_block 2 crashed in init_from_prefill on ~E with
    E = None.  The head-block merge includes E only when want_E is set, and the backbone set it from ctx.collect and
    want_dense_diag alone; generation sets neither."""
    import torch
    from conftest import rand_case
    import numpy as np
    from marsea.normalizer import MarSeaNormalizer
    case = rand_case(np.random.default_rng(3), n_q=12, n_k=12, H=4, Hkv=2, B=1, D=16)
    nm = MarSeaNormalizer(16, head_block=2)
    nm.want_dense_diag = False
    nm.want_E = False
    _, d = nm.normalize(case["S"], case["vis"], case["K_kv"], case["Q"])
    assert d.E is None, "the merge drops E unless asked: that is the saving, and why the prefill must ask"
    nm.want_E = True
    _, d = nm.normalize(case["S"], case["vis"], case["K_kv"], case["Q"])
    assert torch.is_tensor(d.E) and d.E.shape == case["S"].shape and d.tau_j is not None and d.cbar_j is not None and d.nu is not None
    src = (ROOT / "marsea/backbone.py").read_text()
    assert "want_E = bool(ctx.collect or self.normalizer.want_dense_diag or ctx.generation)" in src
