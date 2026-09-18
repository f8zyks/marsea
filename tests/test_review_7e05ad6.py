"""Regression tests for the 7e05ad6 review: the evaluation path validates and records the tolerance at its length."""
import os, pathlib, subprocess, sys
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_eval_provenance_carries_the_tolerance_at_the_jobs_length_and_validates_the_override():
    """run_eval.py called unit_cap_record() with no length, so the eval tables recorded None for the tolerance and the
    ceiling and an override the ceiling exists to refuse passed the whole queue unremarked (review 7e05ad6 2)."""
    import run_eval
    class Ex:
        def __init__(self, n, g): self.prompt_ids = list(range(n)); self.gold_ids = list(range(g))
    assert run_eval.eval_length([Ex(100, 8), Ex(4000, 8)], "data/ruler/E2_L8192_K8_V4_s0") == 4008
    assert run_eval.eval_length([], "data/ruler/E3_L16384_K4_s1") == 16384
    assert run_eval.eval_length([], "data/musique/musique_ans_v1.0_dev.jsonl") == 0
    src = (ROOT / "scripts/run_eval.py").read_text()
    assert "unit_cap_record(eval_length(exs, args.set))" in src and "unit_cap_record()" not in src
    from marsea.train import unit_cap_record
    rec = unit_cap_record(16384)
    assert rec["tol_cap_at_L"] is not None and rec["tol_cap_ceiling_at_L"] is not None
    # the eval path now refuses the same value training refuses, at the job's length
    r = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
                        "from marsea.train import unit_cap_record; unit_cap_record(16384)"],
                       capture_output=True, text=True, env=dict(os.environ, MARSEA_TOL_CAP="1e-3"))
    assert r.returncode != 0 and "exceeds n_k eps / 4" in r.stderr
