"""Regression tests for the e6ca44f review: one verify_all.py (D), the window gate's defined failure branch (E), the
manifest covering the whole tree (H)."""
import inspect, os, pathlib, subprocess, sys
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_verify_all_has_one_copy_and_carries_both_halves():
    """The repo carried two verify_all.py that drifted in both directions: the code copy had the exit code preflight
    gates on, the paper's copy had Prop. A(iii) re-targeted to the row's own softmax mass; neither had both, and the
    copy that runs printed the label tab:verification no longer uses (review e6ca44f D)."""
    code = ROOT / "scripts/verify_all.py"; doc = ROOT / "documents/verify_all.py"
    if not (ROOT / "documents").is_dir():
        pytest.skip("documents/ is not in the handover bundle (code-only by design); the symlink is checked in the repo")
    assert doc.is_symlink() and doc.resolve() == code.resolve(), "documents/verify_all.py must be a symlink to the code copy, not a file"
    va = code.read_text()
    assert "row mass never exceeds the row's own softmax mass" in va and "never exceeds one unit" not in va
    assert "_proj_le(At[i, :], sm_mass_i)" in va and "sys.exit(1)" in va and "TOTAL VIOLATIONS" in va
    assert not (ROOT / "documents/verify_all_MERGED_use_this.py").exists(), "the transient merged copy would be a third file"


def test_cap_tolerance_has_a_measured_override_and_the_gate_names_it(monkeypatch):
    """`8 eps log2(n_k)` is not a derived law; if the measured guard window exceeds it on the target device the defined
    branch is MARSEA_TOL_CAP := 4 x window, recorded in every README and eval table (review e6ca44f E)."""
    from marsea import invariants
    from marsea.invariants import cap_tolerance
    from marsea.train import unit_cap_record
    # the override is parsed ONCE at import (review 4f754ed 2.3): the test sets the parsed value, not the variable
    monkeypatch.setattr(invariants, "TOL_CAP_OVERRIDE", None)
    base = cap_tolerance(8192)
    assert abs(base - 8 * torch.finfo(torch.float32).eps * 13) < 1e-12
    monkeypatch.setattr(invariants, "TOL_CAP_OVERRIDE", 8e-6)
    assert cap_tolerance(8192) == base, "an override BELOW the default must not loosen nothing / tighten the gate"
    monkeypatch.setattr(invariants, "TOL_CAP_OVERRIDE", 2e-5); monkeypatch.setenv("MARSEA_TOL_CAP", "2e-5")
    assert cap_tolerance(8192) == 2e-5 and cap_tolerance(8192, is64=True) < 1e-12
    rec = unit_cap_record(8192)                      # 2e-5 is under the 8192 eps / 4 ceiling: accepted and recorded
    assert rec["MARSEA_TOL_CAP"] == "2e-5" and rec["tol_cap_at_L"] == 2e-5 and rec["tol_cap_ceiling_at_L"] > 2e-5
    cu = (ROOT / "scripts/check_unit_cap.py").read_text()
    assert "MARSEA_TOL_CAP=" in cu and "4 * gw['window']" in cu, "the gate's failure branch must print the number to set"
    if not (ROOT / "RUNBOOK_nebius.md").exists():
        pytest.skip("RUNBOOK_nebius.md is not in the handover bundle (code-only by design)")
    rb = (ROOT / "RUNBOOK_nebius.md").read_text()
    assert "MARSEA_TOL_CAP" in rb and "If this gate fails" in rb


def test_handoff_manifest_covers_every_shipped_file():
    """two PDFs under scripts/ shipped in the tarball with no manifest line (review e6ca44f H)."""
    hs = (ROOT / "scripts/handoff.sh").read_text()
    assert "-name '*.py' -o -name '*.sh'" not in hs and "find marsea scripts tests -type f" in hs
    listed = subprocess.run(["bash", "-c", "cd '%s' && find marsea scripts tests -type f -not -path '*/__pycache__/*' "
                             "-not -name '*.pyc' -not -path '*/runs/*' | sort" % ROOT], capture_output=True, text=True).stdout.split()
    assert "scripts/pr_curve.pdf" in listed and "scripts/v1_probe.pdf" in listed
