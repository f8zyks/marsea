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
