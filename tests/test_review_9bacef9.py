"""Regression tests for the 9bacef9 review: the three launch blockers (C-1 preflight's exit status, C-2 the E9 inventory,
C-3 collect.py's per-seed labels), the theta guard's window (D), and the invariant / reporting residuals (E)."""
import inspect, json, os, pathlib, shutil, subprocess, sys
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))


# --------------------------------------------------------------------------- C-1: preflight's tail
def _preflight_tail(dense_note, paired_note, time_note=""):
    pf = (ROOT / "scripts/preflight.sh").read_text()
    tail = pf[pf.index('if [ -n "$DENSE_TRAIN_NOTE$PAIRED_NOTE'):]
    script = f'set -e\nDENSE_TRAIN_NOTE="{dense_note}"; PAIRED_NOTE="{paired_note}"; DENSE_TIME_NOTE="{time_note}"\nGATE_GB=72\n{tail}'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


@pytest.mark.parametrize("dense,paired", [("dense failed", ""), ("", "paired failed"), ("dense failed", "paired failed"), ("", "")])
def test_preflight_exits_zero_whatever_recorded_gates_failed(dense, paired):
    """`[ -n "$PAIRED_NOTE" ] && echo` as the branch's last command returned 1 exactly when only the dense gate had
    failed -- the likeliest recorded failure, which the branch exists to survive (review 9bacef9 C-1)."""
    r = _preflight_tail(dense, paired)
    assert r.returncode == 0, (dense, paired, r.stdout, r.stderr)
    for note in (dense, paired):
        if note:
            assert note in r.stdout
    assert ("PASSED EXCEPT" in r.stdout) == bool(dense or paired)


def test_preflight_probes_are_recorded_not_fatal_and_the_paired_build_is_guarded():
    pf = (ROOT / "scripts/preflight.sh").read_text()
    assert "--mode dense" in pf and "step_time_dense.json" in pf, "the dense path's step time is still unmeasured (9bacef9 E)"
    pm = (ROOT / "scripts/profile_memory.py").read_text()
    assert "OOM building the second (paired) model" in pm, "an OOM building the paired model still escapes as a traceback"
    src = inspect.getsource(__import__("profile_memory"))
    build = src[src.index("if args.paired:"):src.index("pm.eval(); paired")]
    assert "try:" in build and "OutOfMemoryError" in build and "sys.exit(3)" in build


# --------------------------------------------------------------------------- C-2: the E9 inventory
def test_evalsuite_refuses_an_incomplete_grid_and_names_the_missing_e9_arm(tmp_path):
    """The E9 loop's `[ -f "$ck" ] || continue` dropped an untrained arm silently, and a glob over runs/e9_*/ cannot see
    a directory that was never created: three of six E9 arms vanished with exit 0 (review 9bacef9 C-2)."""
    from test_review_e982f83 import _sim_queue
    r, log, _ = _sim_queue(tmp_path, e9_missing=("e9_uniform_quota", "e9_hierarchical_K64"))
    assert r.returncode == 2 and not log, (r.returncode, r.stdout, r.stderr)
    skipped = (tmp_path / "sim/runs/eval/SKIPPED_ARMS.txt").read_text()
    assert "e9_uniform_quota (E9)" in skipped and "e9_hierarchical_K64 (E9)" in skipped and "e9_per_head" not in skipped
    assert "REFUSING TO START" in r.stderr and "e9_hierarchical_K64" in r.stderr
    # ALLOW_MISSING_ARMS=1: the queue runs what exists, names the gap at the top AND at the end, exits non-zero
    r, log, _ = _sim_queue(tmp_path / "b", e9_missing=("e9_uniform_quota",), extra_env={"ALLOW_MISSING_ARMS": "1"})
    tags = [ln.split("--tag ")[1].split()[0] for ln in log]
    assert "e9_e9_hierarchical_K64_E2" in tags and "e9_e9_uniform_quota_E2" not in tags
    assert r.returncode in (1, 2) and r.stderr.count("e9_uniform_quota (E9)") >= 2, r.stderr


def test_queues_reject_a_non_positive_ngpu(tmp_path):
    for script in ("run_evalsuite.sh", "run_s2.sh"):
        for bad in ("0", "abc", "-1"):
            r = subprocess.run(["bash", str(ROOT / "scripts" / script), bad], capture_output=True, text=True, cwd=tmp_path,
                               env=dict(os.environ, PY="/bin/false"))
            assert r.returncode == 1 and "positive integer" in r.stderr, (script, bad, r.stderr)


def test_run_s2_reads_the_detector_guarded():
    s2 = (ROOT / "scripts/run_s2.sh").read_text()
    assert "cannot read patched_layers from runs/detector.json" in s2
    es = (ROOT / "scripts/run_evalsuite.sh").read_text()
    assert "is unreadable" in es and "E9_ARMS=" in es and "e9_uniform_quota_control" in es


# --------------------------------------------------------------------------- C-3: collect.py per-seed labels
def test_collect_per_seed_lists_are_positional_by_seed(tmp_path):
    """a key first scored at seed k landed at index 0 and was then labelled seed 0 (review 9bacef9 C-3)."""
    import collect
    def table(tag, seed, strata, extra=None):
        (tmp_path / f"{tag}_table.json").write_text(json.dumps(dict(
            provenance=dict(tag=tag, seed=seed, arm="marsea", experiment="E5", git="abc", detector_sha256="d", stratify="m"),
            table=strata, **(extra or {}))))
    table("marsea_s0_E5", 0, {"all": {"acc": 0.50}})
    table("marsea_s1_E5", 1, {"all": {"acc": 0.51, "S": 0.7}})
    table("marsea_s2_E5", 2, {"all": {"acc": 0.52}, "late": {"acc": 0.99}}, extra={"exact_match_strict": {"1": 0.3}})
    sys.argv = ["collect.py", "--eval_dir", str(tmp_path)]
    collect.main()
    c = json.loads((tmp_path / "collected.json").read_text())["jobs"]["marsea_E5"]
    assert c["all"]["acc"]["per_seed"] == [0.5, 0.51, 0.52] and c["all"]["acc"]["seeds"] == [0, 1, 2]
    assert c["all"]["S"]["per_seed"] == [None, 0.7, None], c["all"]["S"]
    assert c["late"]["acc"]["per_seed"] == [None, None, 0.99], c["late"]["acc"]
    assert c["exact_match_strict:1"]["value"]["per_seed"] == [None, None, 0.3]      # E6's strict table is read (E)


def test_collect_e8_and_coverage_carry_a_positional_seed_dimension(tmp_path):
    import collect
    rows_by_seed = {0: [dict(n=8, seed=0, e8={"19": dict(rho=0.04, rho_col=[0.1], zero_mass_tol=2e-5, pass2_rebuilt_row_frac=0.2,
                                                        _pass2_rows=100, source="dense")},
                              attention=dict(n_q=100, cols=[dict(E_size=5)]))],
                    2: [dict(n=8, seed=2, e8={"19": dict(rho=0.06, rho_col=[0.3], zero_mass_tol=2e-5, pass2_rebuilt_row_frac=0.6,
                                                        _pass2_rows=300, source="dense")},
                              attention=dict(n_q=100, cols=[dict(E_size=15)]))]}
    for seed in (0, 1, 2):
        tag = f"marsea_s{seed}_E3"
        (tmp_path / f"{tag}_table.json").write_text(json.dumps(dict(
            provenance=dict(tag=tag, seed=seed, arm="marsea", experiment="E3", git="abc", detector_sha256="d", stratify="n"),
            table={"8": dict(row_precision=0.5)})))
        (tmp_path / f"{tag}.jsonl").write_text("\n".join(json.dumps(r) for r in rows_by_seed.get(seed, [])))
    sys.argv = ["collect.py", "--eval_dir", str(tmp_path)]
    collect.main()
    out = json.loads((tmp_path / "collected.json").read_text())
    e8 = out["e8"]["marsea_E3"]["19"]
    assert e8["rho"]["per_seed"] == [0.04, 0.06] and e8["rho"]["seeds"] == [0, 2], e8["rho"]
    assert abs(e8["pass2_rebuilt_row_frac"]["mean"] - 0.5) < 1e-12, "the count-weighted pass-2 fraction was overwritten by a seed mean"
    assert e8["pass2_rows"] == 400 and e8["zero_mass_tol"]["mean"] == 2e-5 and e8["rho_col"]["mean"] == 0.2
    assert e8["source"] == ["dense"]
    cov = out["coverage"]["marsea_E3"]["8"]
    assert cov["per_seed"] == [0.05, None, 0.15] and cov["seeds"] == [0, 1, 2] and cov["n_seeds"] == 2, cov


# --------------------------------------------------------------------------- D / E: the cap's own theta, the guard window
def _razor_geometry(n, rho, dev):
    from marsea.normalizer import MarSeaNormalizer
    gaps = [15.5 + 0.05 * i for i in range(41)]
    torch.manual_seed(0)
    D = 16
    norm = MarSeaNormalizer(D, 8).to(dev)
    K = torch.zeros(1, 1, n, D, device=dev); K[0, 0, 0, 0] = 1.0
    K[0, 0, 1:, 1:] = torch.randn(n - 1, D - 1, device=dev) * 1e-3
    Q = torch.zeros(1, 1, len(gaps), D, device=dev); Q[0, 0, :, 0] = torch.tensor(gaps, device=dev) * D ** 0.5
    vis = torch.ones(1, 1, len(gaps), n, dtype=torch.bool, device=dev)
    S = torch.matmul(Q, K.transpose(-1, -2)) * D ** -0.5
    norm.relation.calibrate_b0(K, Q, vis, rho)
    return norm, S, K, Q, vis


def test_the_caps_own_theta_is_recorded_and_positive_on_every_binding_row_on_all_three_paths():
    """`diag.theta` is Stage 2's dual, non-negative by construction; the property "theta > 0 on every capped row" was
    asserted nowhere (review 9bacef9 E).  Diagnostics.cap_theta now carries the cap's."""
    from marsea.normalizer import State, CAP_TOL
    from marsea.chunked import marsea_chunked_attention
    from marsea.causal import FrozenPrefixCache, decode_step, NEG_PAD
    from marsea.invariants import invariant_report, invariant_report_sparse
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = 2048
    norm, S, K, Q, vis = _razor_geometry(n, 0.05, dev)
    with torch.no_grad():
        A, d = norm.normalize(S, vis, K, Q, State())
        assert d.cap_binds.any() and d.cap_theta is not None
        assert bool((d.cap_theta[d.cap_binds] > 0).all()) and bool((d.cap_theta[~d.cap_binds] == 0).all())
        ex = (d.Atil.double() - d.A_sm.double()).sum(-1)
        assert bool((~d.cap_binds & (ex > CAP_TOL)).sum() == 0) or bool((d.cap_theta[~d.cap_binds & (ex > CAP_TOL)] <= 0).all())
        rep = invariant_report(A, d, vis)
        assert rep["INV-5c"][0] and rep["INV-3a"][0] and "INV-5c (no cap_theta recorded)" not in rep.get("_not_checked", [])
        _, dc = marsea_chunked_attention(norm, Q, K, K, vis, State(), chunk=1024)
        assert dc.cap_binds.any() and bool((dc.cap_theta[dc.cap_binds] > 0).all()) and bool((dc.cap_theta[~dc.cap_binds] == 0).all())
        reps = invariant_report_sparse(dc, vis)
        assert reps["INV-5c"][0] and reps["INV-5-plumbing"][0] and reps["INV-3-mass"][0], reps
        # a stored sm_mass that no longer matches rowsum - excess used to trip nothing (E)
        dc.extra = dict(dc.extra); dc.extra["sm_mass"] = dc.extra["sm_mass"] * 1.1
        assert not invariant_report_sparse(dc, vis)["INV-3-mass"][0]
        # decode: the same cap on row t, its theta in the info dict
        bound = 0
        for i in range(S.shape[-2]):
            cache = FrozenPrefixCache(K_ret=8)
            # a column quota already in the cache (earlier queries'): with an empty cache Atil == A_sm on E and nothing binds
            cache.tau_j = torch.ones(1, 1, n, device=dev); cache.cbar_j = torch.full((1, 1, n), 1e-7, device=dev); cache.nu = torch.ones(1, 1, n, device=dev)
            cache.rel_scores = torch.full((1, 1, n, 8), NEG_PAD, device=dev); cache.rel_valid = torch.zeros(1, 1, n, 8, dtype=torch.bool, device=dev)
            cache.rel_qidx = torch.full((1, 1, n, 8), -1, dtype=torch.long, device=dev); cache.n_seen = n
            Ebig = torch.zeros(1, 1, 1, n, dtype=torch.bool, device=dev); Ebig[..., 1:200] = True
            _, info = decode_step(norm, cache, S[:, :, i:i + 1], vis[:, :, i:i + 1], K, Q[:, :, i:i + 1],
                                  logits_row=torch.where(Ebig, 5.0, -1e4))
            assert "cap_theta" in info and "sm_mass" in info
            if info["cap_binds"].any():
                bound += 1
                assert float(info["cap_theta"][info["cap_binds"]].min()) > 0 and bool((info["a1"] <= info["Atil"]).all())
        assert bound > 0, "decode's cap never bound: the property was not exercised"


def test_inv3a_trips_a_stage1_excess_the_old_tolerance_swallowed():
    """INV-3 carried the fp32-era tol_sum (1.13e-4 at 16K) on fp64 sums; a +3e-4 corruption of A did not trip it.  INV-3a
    on Stage 1's output now trips at ~1e-5, and INV-3 at +3e-4 (review 9bacef9 E)."""
    from marsea.invariants import invariant_report, cap_tolerance
    from marsea.normalizer import Diagnostics, CAP_TOL
    n_k = 2048
    torch.manual_seed(0)
    A_sm = torch.softmax(torch.randn(1, 1, 3, n_k), -1)
    def diag(A, a1=None):
        return Diagnostics(E=torch.zeros(1, 1, 3, n_k, dtype=torch.bool), Atil=A_sm.clone(), A_sm=A_sm.clone(),
                           a1=(a1 if a1 is not None else A.clone()), cap_binds=torch.zeros(1, 1, 3, dtype=torch.bool),
                           cap_theta=torch.zeros(1, 1, 3), cbar_j=torch.zeros(1, 1, n_k), cbar_i=torch.zeros(1, 1, 3),
                           tau_j=torch.ones(1, 1, n_k), tau_i=torch.ones(1, 1, 3), nu=torch.ones(1, 1, n_k))
    vis = torch.ones(1, 1, 3, n_k, dtype=torch.bool)
    rep = invariant_report(A_sm.clone(), diag(A_sm), vis)
    assert rep["INV-3"][0] and rep["INV-3a"][0]
    a1_bad = A_sm.clone(); a1_bad[0, 0, 0] += 2e-5 / n_k                    # Stage 1 handed out 2e-5 more than the softmax
    assert not invariant_report(A_sm.clone(), diag(A_sm, a1_bad), vis)["INV-3a"][0]
    assert 1e-5 < CAP_TOL + cap_tolerance(n_k) < 2e-5
    A_bad = A_sm.clone(); A_bad[0, 0, 1] += 3e-4 / n_k                       # the reviewer's corruption
    assert not invariant_report(A_bad, diag(A_bad), vis)["INV-3"][0]
    assert "@torch.no_grad()" in inspect.getsource(invariant_report) and hasattr(invariant_report, "__wrapped__"), \
        "invariant_report must run under @torch.no_grad (it held a whole-row fp64 copy with graph)"


def test_theta_guard_window_is_measured_and_within_inv3_on_this_device():
    """The guard's window (the fp32 prefix scan's error) had never been measured; on the dev GPU it is 1.1e-6 at 8K --
    above CAP_TOL -- and a strictly sequential fp32 scan could reach n_k eps / 4 with every earlier gate still passing
    (review 9bacef9 D).  check_unit_cap.py bisects it and gates it against INV-3's tolerance."""
    import check_unit_cap
    from marsea.invariants import cap_tolerance
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gaps = [15.5 + 0.05 * i for i in range(41)]
    gw = check_unit_cap.measure_guard_window(2048, gaps, dev)
    assert gw["unresolved_rows"] == 0 and 0 < gw["window"] <= cap_tolerance(2048), gw
    gt = check_unit_cap.gate_mechanism(2048, gaps, dev, 0.5, gw["window"])
    assert gt["passed"], gt
    assert gt["decode_live"]["ok"] and gt["chunked_cap_theta_positive_on_binding"] and gt["min_cap_theta_on_binding"] > 0
    assert gt["binds_above_window_excess"] >= 4e-6 and gt["dense_chunked_sm_mass_diff"] > gt["dense_chunked_excess_diff"]


def test_unit_cap_record_names_the_target_and_the_guard():
    from marsea.train import unit_cap_record
    rec = unit_cap_record()
    assert rec["theta_guard"] is True and "theta > 0" in rec["rule"] and "sum_j A_sm_ij" in rec["target"]
    from marsea.normalizer import row_masses
    src = inspect.getsource(row_masses)
    assert "sum(-1, dtype=torch.float64)" in src and "* E[..., J]" not in src, "row_masses should difference two fp64 sums (B-1)"
