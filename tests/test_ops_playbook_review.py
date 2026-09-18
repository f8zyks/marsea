"""Regression tests for the ops-playbook review (2026-09-17): run_s2.sh's GPU round-robin restarts at 0 after every wave
and the dense E9 arms share the chunked arms' waves (A); checkpoints are written atomically (B); a resume that loses its
optimizer state is recorded in the README (C)."""
import json, os, pathlib, shutil, subprocess, sys
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]

FAKE_PY = r'''#!/usr/bin/env bash
if [ "$1" = "-" ] || [ "$1" = "-c" ]; then exec "$REAL_PY" "$@"; fi
exec 8>"$SIM/gpu$CUDA_VISIBLE_DEVICES.lock"
flock -n 8 || echo "overlap on gpu $CUDA_VISIBLE_DEVICES" >> "$SIM/overlap.log"
( flock 9; echo "$CUDA_VISIBLE_DEVICES $*" >> "$SIM/launch.log" ) 9>"$SIM/launch.lock"
sleep 0.5      # long enough that two jobs of one wave on the same GPU WOULD overlap in wall time (review 2, item 3)
exit 0
'''


def _sim_s2(tmp_path, ngpu=8, skip_dense=False):
    """run the real run_s2.sh past its gates with every trainer replaced by a recorder of CUDA_VISIBLE_DEVICES."""
    if shutil.which("flock") is None:
        pytest.skip("flock not available")
    sim = tmp_path / "sim"; (sim / "scripts").mkdir(parents=True); (sim / "runs").mkdir()
    shutil.copy(ROOT / "scripts/run_s2.sh", sim / "scripts/")
    det = {"patched_layers": [1, 2], "l_star": 1, "h_star": 0, "s0_licence": {}, "column_measurable": {}}
    (sim / "runs/detector.json").write_text(json.dumps(det))
    (sim / "runs/preflight_train_dense.json").write_text(json.dumps({"targets": [8192], "gate": {"passed": True}}))
    for s in range(3):                                  # complete Phase-A files stamped with the detector's layers
        torch.save({"step": 500, "extra": {"phase": "A"}, "detector": {"patched_layers": [1, 2]}}, sim / f"runs/phaseA_seed{s}.pt")
    d = sim / "data/ruler/TRAIN_L8192_K1_V1_Q1_drand_s100"; d.mkdir(parents=True); (d / "validation.jsonl").write_text("{}\n")
    q = sim / "data/ruler/QUICK_L8192_K8_V1_Q1_d0.5_s3"; q.mkdir(parents=True); (q / "validation.jsonl").write_text("{}\n")
    (sim / "data/musique").mkdir(); (sim / "data/musique/musique_ans_v1.0_train.jsonl").write_text("{}\n")
    fake = sim / "fakepy"; fake.write_text(FAKE_PY); fake.chmod(0o755)
    env = dict(os.environ, PY=str(fake), SIM=str(sim), REAL_PY=sys.executable, HOTPOT_N="0", PYTHONPATH=str(ROOT),
               SKIP_DENSE_E9=("1" if skip_dense else "0"))
    r = subprocess.run(["bash", str(sim / "scripts/run_s2.sh"), str(ngpu)], cwd=sim, env=env, capture_output=True, text=True, timeout=600)
    launches = [ln.split(" ", 1) for ln in (sim / "launch.log").read_text().splitlines()] if (sim / "launch.log").exists() else []
    return r, launches, sim


@pytest.mark.parametrize("ngpu", [8, 4])
def test_s2_round_robin_restarts_every_wave_and_the_dense_arms_join_the_first_wave(tmp_path, ngpu):
    """LAUNCH_I was never reset, so Phase A's three launches shifted every Phase-B wave: 5 / 8 / 4 chunked jobs with the
    dense arms on GPUs 4-6 as a fourth wave -- and the playbook's hand-launch on GPUs 1-3 would have doubled up three
    H200s (ops playbook review A).  Now: dense first, 8 / 8 / 4 on 8 GPUs and 4 / 4 / 4 / 4 / 4 on the plan's 4-GPU
    pivot, and the first launch of every wave is on GPU 0."""
    r, launches, sim = _sim_s2(tmp_path, ngpu=ngpu)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    assert not (sim / "overlap.log").exists(), (sim / "overlap.log").read_text()
    phase_a = [l for l in launches if "--phase_a_only" in l[1]]
    phase_b = [l for l in launches if "--phase_a_only" not in l[1]]
    assert len(phase_a) == 3 and len(phase_b) == 20, (len(phase_a), len(phase_b))
    # launch.log is in COMPLETION order (concurrent jobs); the launch ORDER is the script's own "[gpu N] tag" plan
    plan = [ln for ln in r.stdout.splitlines() if ln.startswith("[gpu ") and "phaseA" not in ln]
    gpus = [int(ln.split()[1].rstrip("]")) for ln in plan]
    assert gpus == [i % ngpu for i in range(20)], gpus                # 8 / 8 / 4 (or 4 x 5) -- the round-robin restarted at 0
    assert [ln.split("] ", 1)[1] for ln in plan[:3]] == ["e9_uniform_quota", "e9_uniform_quota_control", "e9_hierarchical_K64"], plan[:3]
    assert sum("--mode dense" in cmd for _, cmd in phase_b) == 3
    assert sorted(int(g) for g, _ in phase_b) == sorted(i % ngpu for i in range(20))


def test_s2_skip_dense_leaves_17_chunked_jobs_in_three_waves(tmp_path):
    """SKIP_DENSE_E9=1 used to be consulted only when the dense memory verdict had FAILED; with a passed verdict the
    dense arms ran regardless (review 2, item 5).  It now drops them whatever the verdict, which still prints."""
    r, launches, sim = _sim_s2(tmp_path, skip_dense=True)
    assert r.returncode == 0, r.stderr[-800:]
    phase_b = [l for l in launches if "--phase_a_only" not in l[1]]
    plan = [ln for ln in r.stdout.splitlines() if ln.startswith("[gpu ") and "phaseA" not in ln]
    assert len(phase_b) == 17 and [int(ln.split()[1].rstrip("]")) for ln in plan] == [i % 8 for i in range(17)]
    assert "dense E9 arms skipped" in r.stdout and "dropped by request" in r.stdout and "gate passed" in r.stdout


def test_checkpoint_write_that_dies_midway_leaves_the_previous_checkpoint_loadable(tmp_path, monkeypatch):
    """`last.pt` is overwritten in place every 250 steps; a pod lost during the write left a file every relaunch raised
    on (review B).  Behavioural: torch.save is made to write a few bytes and then die; the file at `last.pt` must still
    be the OLD checkpoint, and the temporary must not be mistaken for a checkpoint (review 2, item 3)."""
    import inspect, sys, types
    from marsea import train
    old = tmp_path / "last.pt"; torch.save({"step": 250, "marker": "old"}, old)
    fake_peft = types.ModuleType("peft"); fake_peft.get_peft_model_state_dict = lambda m: {}
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setattr(train, "module_state", lambda m: {})
    monkeypatch.setattr(train, "detector_stamp", lambda p: None)
    monkeypatch.setattr(train, "git_hash", lambda: "test")
    real_save = torch.save
    def dying_save(obj, f, *a, **k):
        f.write(b"PK\x03\x04 partial"); raise OSError("node died mid-write")
    monkeypatch.setattr(torch, "save", dying_save)
    cfg = train.TrainConfig(detector_json=None)
    with pytest.raises(OSError):
        train.save_checkpoint(old, model=None, opt=None, step=500, cursor=8000, cfg=cfg)
    assert torch.load(old, map_location="cpu", weights_only=False)["marker"] == "old", "the previous last.pt was damaged"
    assert list(tmp_path.glob("step*.pt")) == [] and any(p.name.startswith("last.pt.tmp") for p in tmp_path.iterdir())
    # and the good path: the write is fsync'd and renamed over the old file
    monkeypatch.setattr(torch, "save", real_save)
    fsyncs = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (fsyncs.append(fd), real_fsync(fd)))
    train.save_checkpoint(old, model=None, opt=None, step=500, cursor=8000, cfg=cfg, extra={"phase": "B"})
    ck = torch.load(old, map_location="cpu", weights_only=False)
    # two fsyncs: the data, then the directory (the second only where the filesystem allows fsync on a directory fd)
    try:
        dfd = os.open(str(tmp_path), os.O_RDONLY); real_fsync(dfd); os.close(dfd); dir_fsync_ok = True
    except OSError:
        dir_fsync_ok = False
    assert ck["step"] == 500 and ck["extra"]["phase"] == "B" and len(fsyncs) >= (2 if dir_fsync_ok else 1), "an fsync did not run"
    src_t = inspect.getsource(train.train)
    assert 'event="optimizer state not restored on resume"' in src_t


def test_setup_script_checks_the_machine_and_pins_the_cache_to_the_volume():
    """setup_runpod.sh (then setup_nebius.sh) predated the network-volume design: no bash / CUDA / SXM checks, the HF cache on the container
    disk, RULER + the Paul Graham download unconditionally, `runs/` never created, and "next" pointing at gen_data.sh."""
    src = (ROOT / "scripts/setup_runpod.sh").read_text()
    assert subprocess.run(["bash", "-n", str(ROOT / "scripts/setup_runpod.sh")]).returncode == 0
    for needle in ("BASH_VERSINFO", "CUDA Version", 'grep -iE "NVL|PCIe"', "MIN_GPU_MIB", "EXPECT_NGPU", "mkdir -p runs",
                   "HF_HOME", "pod_env.sh", "NEED_RULER", "measured_length_mean", "musique_ans_v1.0_", "preflight.sh",
                   "REQUIRE_DATA", "runs/backbone.json", 'fsid "$HF_HOME"'):
        assert needle in src, needle
    assert 'grep -qiv "SXM"' not in src, "an allow-list on the product string warns on the correct hardware (review acda7b6 1)"
    assert "gen_data.sh" not in src.split("=== done")[1], "the next step must be preflight, not data generation on the pod"
    gi = (ROOT / ".gitignore").read_text()
    assert "pod_env.sh" in gi and "\nruns/\n" in gi and "!runs/*.json" not in gi


def test_runs_directory_is_output_and_the_gate_4b_runbook_is_in_the_tracked_tree():
    """S0 writes into runs/detector.json; while runs/*.json were tracked, git_hash() would have stamped <hash>+dirty on
    every README, checkpoint and eval table from S0 on.  And the two doc-code guards on gate 4b read RUNBOOK_nebius.md
    at the repo root, which had moved under the ignored documents/ (review acda7b6 2)."""
    tracked = subprocess.run(["git", "ls-files", "runs/"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    assert tracked == [], f"output files tracked under runs/: {tracked}"
    assert (ROOT / "RUNBOOK_nebius.md").exists(), "RUNBOOK_nebius.md must be in the tracked tree (the pod has no documents/)"
    rb = (ROOT / "RUNBOOK_nebius.md").read_text()
    assert "MARSEA_TOL_CAP" in rb and "If this gate fails" in rb
