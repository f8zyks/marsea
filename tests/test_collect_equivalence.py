"""collect.py --equivalent_commits (2026-09-20): the evaluation queue was fixed mid-campaign by a commit that does not touch
the pooled arms' path, so a job's seeds carry two commits.  Pooling them is an owner's decision with evidence -- recorded in
collected.json, never silent, and never wider than the commits it names."""
import importlib.util, json, pathlib, sys
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("collect", ROOT / "scripts/collect.py")
collect = importlib.util.module_from_spec(spec); spec.loader.exec_module(collect)
A, B, C = "e4541bdcfa4ecdc26acda0039f4ef890d5884d0d", "39439641c42bc4ff94e306069130cbb532080160", "1234567aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def _tables(d, gits):
    for seed, (git, prec) in enumerate(zip(gits, (0.5, 0.7, 0.9))):
        tag = f"marsea_s{seed}_E4"
        (d / f"{tag}_table.json").write_text(json.dumps(dict(
            provenance=dict(tag=tag, seed=seed, arm="marsea", experiment="E4", git=git, detector_sha256="d", stratify="m"),
            table={"1": dict(row_precision=prec, n=200)})))


def _run(d, decl=None):
    sys.argv = ["collect.py", "--eval_dir", str(d)] + (["--equivalent_commits", str(decl)] if decl else [])
    collect.main()


def test_two_commits_pool_only_under_a_recorded_declaration(tmp_path):
    _tables(tmp_path, (A, B, B + "+dirty"))
    with pytest.raises(SystemExit):
        _run(tmp_path)                                                      # without it: the old refusal
    assert json.loads((tmp_path / "collected.json").read_text())["jobs"]["marsea_E4"]["1"]["row_precision"]["mean"] is None
    decl = tmp_path / "equiv.json"
    decl.write_text(json.dumps(dict(commits=[A[:7], B], evidence="three E4 jobs re-run: rows identical", decided_by="owner", date="2026-09-20")))
    _run(tmp_path, decl)
    out = json.loads((tmp_path / "collected.json").read_text())
    st = out["jobs"]["marsea_E4"]["1"]["row_precision"]
    assert st["n_seeds"] == 3 and abs(st["mean"] - 0.7) < 1e-12 and not out["provenance_conflicts"]
    ce = out["commit_equivalence"]
    assert ce["pooled_jobs"] == ["marsea_E4"] and ce["evidence"].startswith("three E4") and ce["decided_by"] == "owner"
    assert "declared equivalent" in (tmp_path / "collected.md").read_text()


def test_a_third_commit_is_still_a_conflict(tmp_path):
    _tables(tmp_path, (A, B, C))
    decl = tmp_path / "equiv.json"
    decl.write_text(json.dumps(dict(commits=[A, B], evidence="x")))
    with pytest.raises(SystemExit):
        _run(tmp_path, decl)
    out = json.loads((tmp_path / "collected.json").read_text())
    assert "marsea_E4" in out["provenance_conflicts"] and out["commit_equivalence"]["pooled_jobs"] == []


@pytest.mark.parametrize("bad", [dict(commits=[A], evidence="x"), dict(commits=[A, B], evidence=" "), dict(commits=["e45", B], evidence="x")])
def test_a_declaration_needs_two_real_hashes_and_evidence(tmp_path, bad):
    _tables(tmp_path, (A, B, B))
    decl = tmp_path / "equiv.json"; decl.write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="equivalent_commits needs"):
        _run(tmp_path, decl)


def test_run_eval_records_the_mode_the_arm_ran_on():
    src = (ROOT / "scripts/run_eval.py").read_text()
    assert "mode_effective=ctx.mode" in src and "mode=args.mode" in src
