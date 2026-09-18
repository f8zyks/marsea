"""Regression tests for the 4f754ed review: the MARSEA_TOL_CAP override is bounded by the arithmetic and by the measured
window, and parsed once at startup."""
import os, pathlib, subprocess, sys
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _tol(env, expr):
    r = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {str(ROOT)!r}); from marsea import invariants as I; print({expr})"],
                       capture_output=True, text=True, env=dict(os.environ, **env))
    return r


def test_override_is_refused_above_the_scan_ceiling_and_parsed_at_import():
    """max(default, override) had no ceiling: MARSEA_TOL_CAP=1 made INV-3a admit any row, silently, for a 30-hour queue;
    a malformed value raised inside an invariant check mid-run (review 4f754ed 2)."""
    from marsea.invariants import tol_cap_ceiling
    eps = torch.finfo(torch.float32).eps
    assert abs(tol_cap_ceiling(16384) - 16384 * eps / 4) < 1e-15
    # a typo fails at IMPORT, naming the variable
    r = _tol({"MARSEA_TOL_CAP": "1e-5x"}, "I.cap_tolerance(8192)")
    assert r.returncode != 0 and "MARSEA_TOL_CAP='1e-5x' is not a number" in r.stderr, r.stderr
    r = _tol({"MARSEA_TOL_CAP": "-1e-6"}, "I.cap_tolerance(8192)")
    assert r.returncode != 0 and "must be a positive number" in r.stderr
    # a whole row of mass is refused at the run's length by the startup validation unit_cap_record performs
    r = _tol({"MARSEA_TOL_CAP": "1"}, "I.validate_tol_cap_override(8192)")
    assert r.returncode != 0 and "not a physically possible" in r.stderr and "exceeds n_k eps / 4" in r.stderr
    r = _tol({"MARSEA_TOL_CAP": "1"}, "__import__('marsea.train', fromlist=['unit_cap_record']).unit_cap_record(8192)")
    assert r.returncode != 0 and "MARSEA_TOL_CAP" in r.stderr, "unit_cap_record must refuse the override at startup"
    # the intended value (4 x the dev GPU's window) clears the ceiling, the measured-window bar, and the record carries it
    r = _tol({"MARSEA_TOL_CAP": "8e-6"}, "(I.validate_tol_cap_override(8192, window=2e-6), I.cap_tolerance(8192))")
    assert r.returncode == 0 and "8e-06" in r.stdout, r.stdout + r.stderr
    # a value justified on another device (more than 8 x the window measured here) is refused by the gate
    r = _tol({"MARSEA_TOL_CAP": "1e-4"}, "I.validate_tol_cap_override(16384, window=1.2e-7)")
    assert r.returncode != 0 and "measured on THIS device" in r.stderr
    r = _tol({}, "I.validate_tol_cap_override(8192, window=1e-7)")
    assert r.returncode == 0 and r.stdout.strip() == "None"


def test_gate_refuses_a_stale_override_instead_of_self_certifying(tmp_path):
    """window_ok compared against cap_tolerance(n), which includes the override, so once set the gate passed by
    construction; now it re-validates the override against the window it has just measured."""
    cu = (ROOT / "scripts/check_unit_cap.py").read_text()
    assert "validate_tol_cap_override(n, gw[\"window\"])" in cu and "override_refused" in cu
    r = subprocess.run([sys.executable, str(ROOT / "scripts/check_unit_cap.py"), "--lengths", "2048", "--elements", "1e5",
                        "--rhos", "0.05", "--out", str(tmp_path / "uc.json")],
                       capture_output=True, text=True, env=dict(os.environ, MARSEA_TOL_CAP="5e-5", CUDA_VISIBLE_DEVICES=""))
    assert r.returncode == 3 and "measured on THIS device" in r.stdout and "GATE FAILED" in r.stdout, r.stdout[-800:]
    if not (ROOT / "RUNBOOK_nebius.md").exists():
        pytest.skip("RUNBOOK_nebius.md is not in the handover bundle (code-only by design)")
    rb = (ROOT / "RUNBOOK_nebius.md").read_text()
    assert "L·ε/4" in rb and "8× the window measured" in rb
