"""The evaluation queue on several pods and volumes (2026-09-19): SHARD by a hash of the tag, MULTI_POD claims between pods on
one volume, GPUS to evaluate beside a card that is still training, and paired jobs deferred until their MarSea seed exists."""
import os, shutil, subprocess, sys
import pytest
from test_review_e982f83 import _sim_queue, ROOT


def _tags(log):
    return [ln.split("--tag ", 1)[1].split()[0] for ln in log]


def test_shards_partition_the_eval_jobs_and_are_stable_under_a_changing_inventory(tmp_path):
    r_all, log_all, _ = _sim_queue(tmp_path / "all", ngpu=4)
    all_tags = set(_tags(log_all)); assert len(all_tags) == len(log_all) >= 90
    got = []
    for k in (0, 1):
        r, log, mx = _sim_queue(tmp_path / f"s{k}", ngpu=4, extra_env={"SHARD": f"{k}/2"})
        assert "share is done (SHARD=%d/2" % k in r.stdout and mx <= 4
        got.append(set(_tags(log)))
    assert not (got[0] & got[1]) and got[0] | got[1] == all_tags and min(len(got[0]), len(got[1])) > 30
    # a job keeps its shard when an arm is missing (the split hashes the TAG, not the launch index)
    r, log, _ = _sim_queue(tmp_path / "m", ngpu=4, e9_missing=("e9_per_head",), extra_env={"SHARD": "0/2", "ALLOW_MISSING_ARMS": "1"})
    assert set(_tags(log)) == got[0] - {"e9_e9_per_head_E2"}
    for bad in ("2/2", "x"):
        r, _, _ = _sim_queue(tmp_path / f"bad{bad[0]}", ngpu=4, extra_env={"SHARD": bad})
        assert r.returncode == 1 and "SHARD must be" in r.stderr


def test_two_pods_on_one_volume_never_run_an_eval_job_twice(tmp_path):
    r0, log0, _ = _sim_queue(tmp_path, ngpu=2, extra_env={"SHARD": "0/97"})            # builds the sim; a sliver of the list
    sim = tmp_path / "sim"
    for f in ("conc.log", "conc.max", "conc.cur"):
        (sim / f).unlink(missing_ok=True)
    shutil.rmtree(sim / "runs/eval", ignore_errors=True)
    base = dict(os.environ, PY=str(sim / "fakepy"), SIM=str(sim), REAL_PY=sys.executable, EVAL_DATA_CHECK="0", PYTHONPATH=str(ROOT), MULTI_POD="1")
    procs = [subprocess.Popen(["bash", str(sim / "scripts/run_evalsuite.sh"), n], cwd=sim, env=dict(base, POD_ID=pid),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for pid, n in (("podA", "3"), ("podB", "2"))]
    outs = [p.communicate(timeout=900) for p in procs]
    tags = _tags((sim / "conc.log").read_text().splitlines())
    assert len(tags) == len(set(tags)) >= 90, (len(tags), len(set(tags)))
    ran = [sum(1 for ln in o[0].splitlines() if ln.startswith("[gpu ")) for o in outs]
    assert sum(ran) == len(tags) and min(ran) > 10, ran
    owners = {d.name: (d / "owner").read_text().strip() for d in (sim / "runs/eval/claims").iterdir()}
    assert set(owners.values()) == {"podA", "podB"} and len(owners) == len(tags)


def test_gpus_places_jobs_on_the_named_cards_and_paired_jobs_wait_for_their_marsea_seed(tmp_path):
    r, log, mx = _sim_queue(tmp_path / "g", ngpu=8, extra_env={"GPUS": "1", "SHARD": "0/5"})
    assert {ln.split()[0] for ln in log} == {"1"} and mx == 1
    # rolling evaluation: marsea seed 2 has not finished -> its dense arms and B5 are deferred, everything else runs
    r0, _, _ = _sim_queue(tmp_path / "d0", ngpu=4, extra_env={"SHARD": "0/97"})
    sim = tmp_path / "d0/sim"
    (sim / "runs/marsea_seed2/final.pt").unlink(); shutil.rmtree(sim / "runs/eval", ignore_errors=True); (sim / "conc.log").unlink(missing_ok=True)
    env = dict(os.environ, PY=str(sim / "fakepy"), SIM=str(sim), REAL_PY=sys.executable, EVAL_DATA_CHECK="0", PYTHONPATH=str(ROOT), ALLOW_MISSING_ARMS="1")
    r = subprocess.run(["bash", str(sim / "scripts/run_evalsuite.sh"), "4"], cwd=sim, env=env, capture_output=True, text=True, timeout=900)
    tags = _tags((sim / "conc.log").read_text().splitlines())
    assert not [t for t in tags if t.startswith(("B0_s2", "B2_s2", "B5_s2", "marsea_s2"))], "seed 2's paired jobs ran without their MarSea checkpoint"
    assert any(t.startswith("B3_s2") for t in tags) and any(t.startswith("B0_s1") for t in tags)
    assert "defer B0_s2" in r.stdout and "defer B5_s2" in r.stdout and "deferred" in r.stdout
