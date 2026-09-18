"""Regression tests for the e982f83 review: the evaluation queue's concurrency and output names (B-1, B-2), the 16K
job sizes (B-3), the three B-1 residuals (C-1..C-4), the unit cap's slack (D), and the smaller findings (E, F, I)."""
import json, os, pathlib, shutil, subprocess, sys
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


# --------------------------------------------------------------------------- B-1 / B-2 / H: the evaluation queue
FAKE_PY = r'''#!/usr/bin/env bash
if [ "$1" = "-" ] || [ "$1" = "-c" ]; then exec "$REAL_PY" "$@"; fi
exec 8>"$SIM/gpu$CUDA_VISIBLE_DEVICES.lock"
flock -n 8 || echo "overlap on gpu $CUDA_VISIBLE_DEVICES" >> "$SIM/overlap.log"
L="$SIM/conc"
( flock 9; n=$(cat $L.cur 2>/dev/null || echo 0); n=$((n+1)); echo $n > $L.cur
  m=$(cat $L.max 2>/dev/null || echo 0); [ $n -gt $m ] && echo $n > $L.max
  echo "$CUDA_VISIBLE_DEVICES $*" >> $L.log ) 9>$L.lock
case "$*" in *E3pad*) sleep 0.02;; *) sleep 0.15;; esac
( flock 9; n=$(cat $L.cur); echo $((n-1)) > $L.cur ) 9>$L.lock
case "$*" in *"--tag B3_s1_E4 "*) exit 1;; esac
exit 0
'''


def _sim_queue(tmp_path, licence=True, ngpu=8, data_check=False, e9_missing=(), extra_env=None):
    if shutil.which("flock") is None:
        pytest.skip("flock not available")
    sim = tmp_path / "sim"
    (sim / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts/run_evalsuite.sh", sim / "scripts/")
    runs = sim / "runs"; runs.mkdir()
    det = {"patched_layers": [1], "l_star": 1, "h_star": 0}
    if licence:
        det.update(s0_licence={}, column_measurable={})
    (runs / "detector.json").write_text(json.dumps(det))
    for arm in ("marsea", "B0", "B2", "B3"):
        for s in (0, 1, 2):
            (runs / f"{arm}_seed{s}").mkdir(); (runs / f"{arm}_seed{s}/final.pt").touch()
    for arm in ("B1", "B4"):
        (runs / f"{arm}_seed0").mkdir(); (runs / f"{arm}_seed0/final.pt").touch()
    # every E9 arm run_s2.sh trains: the queue now inventories them and refuses an incomplete grid (review 9bacef9 C-2)
    for e9 in ("e9_tau_i_pinned", "e9_key_only_relation", "e9_per_head", "e9_uniform_quota", "e9_uniform_quota_control",
               "e9_hierarchical_K64"):
        if e9 in e9_missing:
            continue
        (runs / e9 / "marsea_seed0").mkdir(parents=True); (runs / e9 / "marsea_seed0/final.pt").touch()
    fake = sim / "fakepy"; fake.write_text(FAKE_PY); fake.chmod(0o755)
    env = dict(os.environ, PY=str(fake), SIM=str(sim), REAL_PY=sys.executable, EVAL_DATA_CHECK="1" if data_check else "0",
               PYTHONPATH=str(ROOT), **(extra_env or {}))
    r = subprocess.run(["bash", str(sim / "scripts/run_evalsuite.sh"), str(ngpu)], cwd=sim, env=env,
                       capture_output=True, text=True, timeout=600)
    log = (sim / "conc.log").read_text().splitlines() if (sim / "conc.log").exists() else []
    mx = int((sim / "conc.max").read_text()) if (sim / "conc.max").exists() else 0
    return r, log, mx


def test_evalsuite_never_runs_more_jobs_than_gpus_and_every_job_has_its_own_tag(tmp_path):
    """The barrier was tested once per ARM while each arm launched seven jobs; 7 is coprime with 8, so 56 evaluations
    ran at once on 8 GPUs (B-1).  Every seed and every E9 arm wrote {arm}_{kind}_{experiment} (B-2)."""
    r, log, mx = _sim_queue(tmp_path)
    assert r.returncode == 1 and "B3_s1_E4" in r.stderr, "an injected failure was not reported"
    assert 1 <= mx <= 8, f"{mx} concurrent evaluations on 8 GPUs"
    ov = tmp_path / "sim" / "overlap.log"
    assert not ov.exists(), ov.read_text()                          # never two jobs on one GPU
    tags = [ln.split("--tag ")[1].split()[0] for ln in log]
    assert len(tags) == len(set(tags)), "two jobs share an output tag"
    assert all(0 <= int(ln.split()[0]) < 8 for ln in log), "a job was placed on a GPU that does not exist"
    e9 = [t for t in tags if t.startswith("e9_")]
    assert sorted(e9) == ["e9_e9_hierarchical_K64_E2", "e9_e9_key_only_relation_E2", "e9_e9_per_head_E2", "e9_e9_tau_i_pinned_E2",
                          "e9_e9_uniform_quota_E2", "e9_e9_uniform_quota_control_E2", "e9_reference_E2"], e9   # e16a843 F-3
    assert all("--mode chunked" in ln for ln in log if "--tag e9_" in ln)
    depth = sorted(t for t in tags if t.endswith("_E3depth"))
    assert depth == ["B0_s0_E3depth", "marsea_s0_E3depth"], depth                   # H: a control, not a sweep
    for ln in log:
        if "_E3 " in ln or "_E3depth " in ln:
            assert "--n 400" in ln, ln                                              # B-3 / H
        if "_E3pad " in ln:
            assert "--n 80" in ln, ln
        if "--tag B0_s" in ln and "_E5" in ln:
            assert "--paired_marsea_ckpt" in ln, "E5's dense arms are not on the paired relation (G)"
        assert "--head_block 2" in ln or "run_e6" in ln


def test_evalsuite_refuses_to_start_without_an_s0_licence(tmp_path):
    r, log, _ = _sim_queue(tmp_path, licence=False)
    assert r.returncode != 0 and not log and "S0 licence" in r.stdout


def test_evalsuite_refuses_to_start_without_its_evaluation_data(tmp_path):
    """14 E5musique jobs used to fail one at a time hours in, for a manual download nothing checked (e16a843 F-2)."""
    r, log, _ = _sim_queue(tmp_path, data_check=True)
    assert r.returncode != 0 and not log, r.stdout
    assert "MuSiQue dev file" in r.stdout and "no RULER set matches" in r.stdout


def test_balanced_take_keeps_every_config():
    from run_eval import balanced_take
    per = [[(f, i) for i in range(200)] for f in "abcde"]
    got = balanced_take(per, 400)
    assert len(got) == 400 and {f for f, _ in got} == set("abcde")
    assert all(sum(1 for f, _ in got if f == c) == 80 for c in "abcde")
    assert len(balanced_take(per, None)) == 1000
    assert len(balanced_take([[1] * 10], 80)) == 10


def test_eval_outputs_are_tagged_and_carry_provenance():
    src = (ROOT / "scripts/run_eval.py").read_text()
    assert 'out / f"{args.arm}_{args.kind}_{args.experiment}"' not in src
    for key in ("tag=tag", "seed=seed", "ckpt=args.ckpt", "git=prov[\"git\"]", "provenance=prov", "detector_sha256"):
        assert key in src, key
    assert "os.replace" in src, "outputs are not written atomically"


def test_training_sources_fail_closed(tmp_path):
    from marsea.data.mix import build_training_sources
    with pytest.raises(FileNotFoundError):
        build_training_sources([], str(tmp_path / "missing.jsonl"), 0, 0)
    s2 = (ROOT / "scripts/run_s2.sh").read_text()
    assert "REFUSING TO START" in s2 and "s0_licence" in s2 and "preflight_train_dense.json" in s2
    assert "CHUNK=" not in s2


def test_preflight_gates_the_dense_arms_and_regenerates_on_doubt():
    pf = (ROOT / "scripts/preflight.sh").read_text()
    assert '|| echo "   dense training did not fit' not in pf
    dense_train = [ln for ln in pf.split("\n$PY ") if "preflight_train_dense.json" in ln][0]
    assert "--lengths 2048 4096 8192" in dense_train and "--gate_GB $GATE_GB" in dense_train
    assert '[ "$NEED_DET" != "0" ]' in pf and "|| echo 1" in pf
    assert "--sites 2" in pf and "preflight_eval_dense.json" in pf and "check_unit_cap.py" in pf


# --------------------------------------------------------------------------- C-2 / C-3: kept heads and the paired relation
def test_kept_head_raises_for_a_head_the_layer_did_not_keep():
    from marsea.evaluate import _kept_head
    from marsea.normalizer import Diagnostics
    d = Diagnostics(extra={"head_sub": [7, 3]})
    assert _kept_head(d, 7) == 0 and _kept_head(d, 3) == 1
    with pytest.raises(KeyError):
        _kept_head(d, 5)
    d1 = Diagnostics(extra={"head_sub": 4})
    assert _kept_head(d1, 4) == 0
    with pytest.raises(KeyError):
        _kept_head(d1, 5)
    assert _kept_head(Diagnostics(), 5) == 5


def test_paired_relation_is_keyed_by_layer_and_head(monkeypatch):
    """keyed by LAYER, two sites in one layer overwrote each other and value columns were scored against the
    coreference head's relation (C-2)."""
    import marsea.evaluate as ev
    from marsea.normalizer import Diagnostics
    E = torch.zeros(1, 2, 6, 6, dtype=torch.bool)
    E[0, 0, 5, 1] = True                 # head 3
    E[0, 1, 5, 2] = True                 # head 7
    def fake_tfp(model, ctx, p, g, l_star, device="cuda", h_star=None, fields=None, sites=None):
        assert sorted(sites) == [(19, 3), (19, 7)]
        return 0.0, 1, {19: Diagnostics(E=E, extra={"head_sub": [3, 7]})}
    monkeypatch.setattr(ev, "teacher_forced_pass", fake_tfp)
    class Ex: prompt_ids, gold_ids = [0], [0]
    pr = ev.PairedRelation(None, None, [Ex()], 19, 3, [(19, 3), (19, 7)], device="cpu")
    got = pr.get(0)
    assert set(got) == {(19, 3), (19, 7)}
    assert got[(19, 3)][5, 1] and not got[(19, 3)][5, 2]
    assert got[(19, 7)][5, 2] and not got[(19, 7)][5, 1]
    assert ev._paired_at(got, 19, 7) is got[(19, 7)]
    with pytest.raises(KeyError):
        ev._paired_at(got, 19, 5)


# --------------------------------------------------------------------------- C-1 / C-4: head blocks and the decode cache
def _chunk_case(T=48, Hkv=2, g=4, seed=3, dtype=torch.float64):
    from marsea.normalizer import MarSeaNormalizer
    D, R = 16, 8
    torch.manual_seed(seed)
    H = Hkv * g
    Q = torch.randn(1, H, T, D, dtype=dtype) * 1.5
    K = torch.randn(1, Hkv, T, D, dtype=dtype) * 1.5
    V = torch.randn(1, Hkv, T, D, dtype=dtype)
    i = torch.arange(T)[:, None]; j = torch.arange(T)[None, :]
    vis = (j <= i)[None, None].clone()
    norm = MarSeaNormalizer(D, R).to(dtype)
    with torch.no_grad():
        norm.relation.calibrate_b0(K, Q, vis, 0.3)
    return norm, Q, K, V, vis


FIELDS = ("E", "A_sm", "Atil", "a1", "A", "p", "cbar_j", "tau_j", "nu", "psi_j", "kstar", "cbar_i", "tau_i", "theta",
          "Rtil", "supp_rel", "cap_binds", "c")


@pytest.mark.parametrize("hb", [2, 4, 3])
@pytest.mark.parametrize("heads", [[3, 5], [7, 3], [0, 1], 6])
def test_head_blocked_merge_keeps_every_requested_head_in_order(hb, heads):
    """two kept heads in DIFFERENT head blocks: the merge overwrote block by block and stamped the whole list, so
    A.shape[1] was 1, head 3's slot held head 5's data and head 5 raised IndexError (review e982f83 C-1)."""
    from marsea.chunked import marsea_chunked_attention
    from marsea.normalizer import State
    from marsea.evaluate import _kept_head
    norm, Q, K, V, vis = _chunk_case()
    hl = [heads] if isinstance(heads, int) else heads
    with torch.no_grad():
        norm.head_block = None
        _, ref = marsea_chunked_attention(norm, Q, K, V, vis, State(), chunk=16, dense_outputs=True)
        norm.head_block = hb
        _, got = marsea_chunked_attention(norm, Q, K, V, vis, State(), chunk=16, dense_outputs=True, dense_head=heads)
    assert got.extra["head_sub"] == heads
    assert got.A.shape[1] == len(hl)
    for h in hl:
        k = _kept_head(got, h)
        for name in FIELDS:
            a, b = getattr(ref, name), getattr(got, name)
            if a is None:
                continue
            assert b is not None, name
            if a.dtype == torch.bool or not a.is_floating_point():
                assert torch.equal(a[:, h], b[:, k]), f"{name} at head {h} (hb={hb})"
            else:
                assert (a[:, h] - b[:, k]).abs().max() <= 1e-12, f"{name} at head {h} (hb={hb})"
    # the merged sparse store is FULL-H and globally indexed (it re-globalised kept-space indices: {0, 4} for {3, 7})
    assert torch.equal(torch.unique(got.extra["sparse"]["bhi"][:, 1]), torch.unique(ref.extra["sparse"]["bhi"][:, 1]))
    assert got.extra["sparse"]["j"].numel() == ref.extra["sparse"]["j"].numel()


@pytest.mark.parametrize("hb", [None, 2])
def test_decode_cache_is_full_when_the_prefill_also_kept_measurement_heads(hb):
    """narrowing extra["sparse"] (and the per-key state) to the kept heads left 7 of 8 heads' decode relation empty
    -- [137, 0, 0, ...] against [145, 130, 155, ...] (review e982f83 C-4)."""
    from marsea.chunked import marsea_chunked_attention, narrow_sparse
    from marsea.causal import FrozenPrefixCache
    from marsea.normalizer import State
    norm, Q, K, V, vis = _chunk_case(T=64)
    B, H, T = 1, Q.shape[1], Q.shape[2]
    norm.head_block = hb
    with torch.no_grad():
        _, d_plain = marsea_chunked_attention(norm, Q, K, V, vis, State(), chunk=16)
        _, d_kept = marsea_chunked_attention(norm, Q, K, V, vis, State(), chunk=16, dense_outputs=True, dense_head=[3, 6])
    caches = []
    for d in (d_plain, d_kept):
        c = FrozenPrefixCache(K_ret=8); c.init_from_sparse(d, B, H, T, T, Q.device, dtype=torch.float64); caches.append(c)
    a, b = caches
    assert a.rel_valid.sum((0, 2, 3)).tolist() == b.rel_valid.sum((0, 2, 3)).tolist()
    assert all(x > 0 for x in b.rel_valid.sum((0, 2, 3)).tolist()), "some head has no decode relation"
    assert (a.eviction_events, a.near_truncation_events) == (b.eviction_events, b.near_truncation_events)
    for nm in ("tau_j", "cbar_j", "nu"):
        assert torch.equal(getattr(a, nm), getattr(b, nm)), nm
    # the collected copy is narrowed AFTER seeding, into head_sub's order
    ns = narrow_sparse(d_kept.extra["sparse"], [6, 3], H)
    full = d_kept.extra["sparse"]
    assert int((ns["bhi"][:, 1] == 0).sum()) == int((full["bhi"][:, 1] == 6).sum())
    assert int((ns["bhi"][:, 1] == 1).sum()) == int((full["bhi"][:, 1] == 3).sum())
    # seeding from narrowed state is refused rather than silently wrong
    with pytest.raises(ValueError):
        bad = type(d_kept)(**{k: v for k, v in d_kept.__dict__.items() if k != "extra"})
        bad.extra = {"sparse": full}
        FrozenPrefixCache(K_ret=8).init_from_sparse(bad, B, H, T, T, Q.device, dtype=torch.float64)


# --------------------------------------------------------------------------- D: the unit cap's slack, both directions
def _entropy_sweep_case(n_k, n_rows_per=2, seed=0):
    """queries scaled per row so the score scale -- and so the row entropy -- sweeps the range where the softmax
    kernel's row-sum error peaks (~2-3 nats); keys with Gaussian and heavy-tailed rows."""
    D = 16
    g = torch.Generator().manual_seed(seed)
    scales = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
    Q = torch.randn(1, 2, len(scales) * n_rows_per, D, generator=g)
    Q = Q / Q.norm(dim=-1, keepdim=True) * torch.tensor(scales).repeat_interleave(n_rows_per)[None, None, :, None] * D ** 0.5
    K = torch.randn(1, 1, n_k, D, generator=g)
    K[..., n_k // 2:, :] = K[..., n_k // 2:, :] / torch.rand(1, 1, n_k - n_k // 2, 1, generator=g).clamp_min(0.05).sqrt() * 0.3
    V = torch.randn(1, 1, n_k, D, generator=g)
    vis = torch.ones(1, 1, Q.shape[2], n_k, dtype=torch.bool)
    return Q, K, V, vis


@pytest.mark.parametrize("n_k", [8192, 16384])
def test_the_cap_never_binds_on_forced_empty_rows_on_any_path(n_k):
    """(i) of D-6: under force_empty no row may bind the unit cap, on the dense, chunked AND decode paths, across the
    entropy sweep -- which also pins the three call sites to the same fp32 slack (fp64 tests pass for any slack)."""
    from marsea.normalizer import MarSeaNormalizer, State
    from marsea.chunked import marsea_chunked_attention
    from marsea.causal import FrozenPrefixCache, decode_step, NEG_PAD
    from marsea.relation import repeat_kv
    torch.manual_seed(0)
    norm = MarSeaNormalizer(16, 8)
    Q, K, V, vis = _entropy_sweep_case(n_k)
    B, H, n_q, D = Q.shape
    with torch.no_grad():
        S = torch.matmul(Q, repeat_kv(K, H).transpose(-1, -2)) * D ** -0.5
        A, d = norm.normalize(S, vis, K, Q, State(), logits_override=torch.full((1, 1, 1, 1), -1e4))
        assert int(d.cap_binds.sum()) == 0, f"dense: the cap bound on {int(d.cap_binds.sum())} forced-empty rows"
        assert torch.equal(A, d.A_sm)
        _, dc = marsea_chunked_attention(norm, Q, K, V, vis, State(), chunk=1024, force_empty=True)
        assert int(dc.cap_binds.sum()) == 0, f"chunked: the cap bound on {int(dc.cap_binds.sum())} forced-empty rows"
        assert dc.extra["pass2_rebuilt_row_frac"] == 0.0
        assert torch.equal(dc.extra["excess"], torch.zeros_like(dc.extra["excess"])), "an empty relation has a nonzero excess"
        for i in range(n_q):
            cache = FrozenPrefixCache(K_ret=8)
            cache.tau_j = torch.ones(B, H, n_k); cache.cbar_j = torch.zeros(B, H, n_k); cache.nu = torch.ones(B, H, n_k)
            cache.rel_scores = torch.full((B, H, n_k, 8), NEG_PAD); cache.rel_valid = torch.zeros(B, H, n_k, 8, dtype=torch.bool)
            cache.rel_qidx = torch.full((B, H, n_k, 8), -1, dtype=torch.long); cache.n_seen = n_k
            A_row, info = decode_step(norm, cache, S[:, :, i:i + 1], vis[:, :, i:i + 1], K, Q[:, :, i:i + 1],
                                      logits_row=torch.full((B, H, 1, n_k), -1e4))
            assert int(info["cap_binds"].sum()) == 0, f"decode: the cap bound on forced-empty row {i}"


def test_the_cap_binds_on_a_genuine_excess_and_not_below_cap_tol():
    """(ii) of D-6, the converse the suite lacked: nothing tested that the binding test was SHARP enough.  A relation that
    adds 2e-6 to a row must bind the cap; one that adds 5e-7 (under CAP_TOL) is left alone -- on a razor-edge row whose
    fp32 softmax total is 1e-4 off, where every slack constant either missed the first or fired on rows with no relation."""
    from marsea.primitives import proj_le_masked
    from marsea.normalizer import relation_excess, cap_binds_from_excess, CAP_TOL
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_k = 8192
    S = torch.zeros(3, n_k, device=dev); S[:, 0] = 16.65                         # the razor edge
    A = torch.softmax(S, -1)[None, None]                                         # [1,1,3,n_k]
    vis = torch.ones_like(A, dtype=torch.bool)
    E = torch.zeros_like(vis); E[..., :, 1:9] = True                             # eight relation entries per row
    for excess, must_bind in ((2e-6, True), (1e-4, True), (0.1, True), (5e-7, False), (0.0, False)):
        Atil = A.clone(); Atil[E] = A[E] + excess / 8                           # spread over the relation
        ex = relation_excess(Atil, A, E)
        assert abs(float(ex.max()) - excess) < 4e-7, (excess, float(ex.max()))  # the elementwise-rounding bound
        binds = cap_binds_from_excess(ex)
        assert bool(binds.all()) == must_bind, f"excess {excess:.0e}: cap binds = {binds.flatten().tolist()}"
        a1, _ = proj_le_masked(Atil, torch.ones(1, 1, 3, device=dev), vis, binds=binds)
        if must_bind:
            assert float((a1.sum(-1, dtype=torch.float64) - 1).abs().max()) < 2e-6
        else:
            assert torch.equal(a1, Atil)
    assert CAP_TOL == 1e-6


# --------------------------------------------------------------------------- E / F / I: invariants, E8, counters
def test_sparse_inv3_is_not_a_tautology():
    """the kept half of the sparse INV-3 read `rowsum` exactly where cap_binds = (rowsum > 1 + slack) was False, so it
    held by construction and a corrupted A left it green (review e982f83 F-1)."""
    from marsea.chunked import marsea_chunked_attention
    from marsea.invariants import invariant_report_sparse
    from marsea.normalizer import State
    norm, Q, K, V, vis = _chunk_case(T=64, Hkv=1, g=2)
    with torch.no_grad():
        _, d = marsea_chunked_attention(norm, Q, K, V, vis, State(), chunk=16)
    rep = invariant_report_sparse(d, vis)
    assert rep["INV-3"][0], rep["INV-3"]
    sp = d.extra["sparse"]
    rid = (sp["bhi"][:, 1] * Q.shape[2] + sp["bhi"][:, 2])
    first_sub = next(k for k in range(rid.numel()) if not bool(d.cap_binds.reshape(-1)[rid[k]]))
    bad = dict(sp); bad["A"] = sp["A"].clone(); bad["A"][first_sub] += 2.0
    d.extra = dict(d.extra); d.extra["sparse"] = bad
    rep2 = invariant_report_sparse(d, vis)
    assert not rep2["INV-3"][0] and rep2["INV-3"][1] > 1.0, rep2["INV-3"]


def test_merge_e8_weights_pass2_by_rows_and_keeps_list_types():
    from marsea.fidelity import merge_e8_summaries
    a = dict(_nnz=10, _n_vis=100, _pass2_rows=400, pass2_rebuilt_row_frac=0.25, hist_Erow=None, rho_head=[0.1, 0.1],
             frac_rows_zero_mass=None, zero_mass_tol=1e-5)
    b = dict(_nnz=30, _n_vis=100, _pass2_rows=200, pass2_rebuilt_row_frac=0.40, hist_Erow=[{"0": 1.0}],
             rho_head=[0.3], frac_rows_zero_mass=[0.0], zero_mass_tol=2e-5)
    m = merge_e8_summaries([a, b])
    assert abs(m["pass2_rebuilt_row_frac"] - (0.25 * 400 + 0.40 * 200) / 600) < 1e-12
    assert m["hist_Erow"] is None and m["frac_rows_zero_mass"] is None      # never the bool True
    assert m["rho_head"] == [0.1, 0.1, 0.3] and abs(m["rho"] - 0.2) < 1e-12 and m["zero_mass_tol"] == 2e-5


def test_zero_mass_rows_mean_the_same_thing_on_both_paths():
    """dense used `A.sum(-1) == 0`, sparse `<= _zero_mass_tol` (review e982f83 E): one definition now, both keys."""
    from marsea.chunked import marsea_chunked_attention
    from marsea.fidelity import e8_summary_from_diag, e8_summary_from_sparse
    from marsea.normalizer import State
    from marsea.relation import repeat_kv
    norm, Q, K, V, vis = _chunk_case(T=64, Hkv=1, g=2)
    with torch.no_grad():
        S = (torch.matmul(Q, repeat_kv(K, 2).transpose(-1, -2)) * 16 ** -0.5).masked_fill(~vis, float("-inf"))
        _, dd = norm.normalize(S, vis, K, Q, State())
        _, dc = marsea_chunked_attention(norm, Q, K, V, vis, State(), chunk=16)
    ed, es = e8_summary_from_diag(dd, vis), e8_summary_from_sparse(dc, vis)
    for key in ("frac_rows_zero_mass", "frac_rows_zero_mass_ambiguous", "zero_mass_tol"):
        assert ed.get(key) is not None and es.get(key) is not None, key
    assert ed["zero_mass_tol"] == es["zero_mass_tol"]
    assert ed["frac_rows_zero_mass"] == es["frac_rows_zero_mass"]


def test_detector_names_the_top_head_and_flags_usability():
    import inspect
    from marsea import detector
    src = inspect.getsource(detector)
    assert "\n    src = above or ranking" not in src and "l_star, h_star = ranking[0][1], ranking[0][2]" in src


def test_collect_pools_seeds_and_refuses_mismatched_provenance(tmp_path):
    """nothing aggregated across seeds, and nothing read rec["e8"] or E3's |E_.j| / n (review e982f83 G)."""
    import collect
    assert collect.job_of("marsea_s1_E3pad") == "marsea_E3pad" and collect.job_of("e9_per_head_E2") == "e9_per_head_E2"
    rows = [dict(n=8, e8={"19": dict(rho=0.04, hist_Ecol=[{"1": 0.9}, {"1": 0.7}], truncation_flag=None,
                                     regime_q10_50_90=[0.15, 0.5, 0.9])},
                 attention=dict(n_q=100, cols=[dict(E_size=5), dict(E_size=15)]))]
    for seed, prec in ((0, 0.5), (1, 0.7)):
        tag = f"marsea_s{seed}_E3"
        (tmp_path / f"{tag}_table.json").write_text(json.dumps(dict(
            provenance=dict(tag=tag, seed=seed, arm="marsea", experiment="E3", git="abc", detector_sha256="d", stratify="n"),
            table={"8": dict(row_precision=prec, row_recall_before=0.4, col_recall_after=0.9, support_EM=0.5, n=80,
                             n_columns=400, interval_hit_rate_by_kind={"coref": 0.7, "value": 0.6},
                             site_by_kind={"coref": [[14, 5]]}, column_measurable=True)})))
        (tmp_path / f"{tag}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    sys.argv = ["collect.py", "--eval_dir", str(tmp_path)]
    collect.main()
    out = json.loads((tmp_path / "collected.json").read_text())
    st = out["jobs"]["marsea_E3"]["8"]["row_precision"]
    assert abs(st["mean"] - 0.6) < 1e-12 and st["n_seeds"] == 2 and abs(st["std"] - 0.1414213562) < 1e-6
    assert abs(out["coverage"]["marsea_E3"]["8"]["mean"] - 0.1) < 1e-12
    assert abs(out["e8"]["marsea_E3"]["19"]["hist_Ecol"]["1"] - 0.8) < 1e-12
    # G-1: the *_before readings, support_EM, the sample sizes and the per-kind splits survive
    st8 = out["jobs"]["marsea_E3"]["8"]
    for k in ("row_recall_before", "col_recall_after", "support_EM", "n", "n_columns", "interval_hit_rate_by_kind[coref]"):
        assert st8[k]["n_seeds"] == 2, k
    assert st8["_per_seed_raw"]["site_by_kind"] == [{"coref": [[14, 5]]}] * 2
    # G-3: a pooled [q10, q50, q90] triple is read, not dropped; G-4: E8 scalars carry a seed dimension
    e8 = out["e8"]["marsea_E3"]["19"]
    assert e8["regime_median_of_q10_50_90"] == [0.15, 0.5, 0.9]
    assert e8["rho"]["n_seeds"] == 2 and abs(e8["rho"]["mean"] - 0.04) < 1e-12
    assert "n_q" in out["captions"]["coverage"]
    bad = json.loads((tmp_path / "marsea_s1_E3_table.json").read_text()); bad["provenance"]["git"] = "other"
    (tmp_path / "marsea_s1_E3_table.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit):
        collect.main()
    # G-2: the conflict is recorded and the file is still written, unpooled for that job
    out = json.loads((tmp_path / "collected.json").read_text())
    assert "marsea_E3" in out["provenance_conflicts"]
    assert out["jobs"]["marsea_E3"]["8"]["row_precision"]["mean"] is None
    assert out["jobs"]["marsea_E3"]["8"]["row_precision"]["per_seed"] == [0.5, 0.7]


# --------------------------------------------------------------------------- a REAL forward, CPU, tiny random Qwen2
def _tiny_patched(mode, head_block):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from marsea.backbone import patch_model, MarSeaContext
    from marsea.baselines import make_normalizer
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=8,
                      num_key_value_heads=2, max_position_embeddings=512, attn_implementation="eager")
    model = Qwen2ForCausalLM(cfg).double().eval()
    ctx = MarSeaContext(mode=mode)
    ctx.chunk = 32
    torch.manual_seed(1)
    patch_model(model, [1, 2], lambda l: make_normalizer("marsea", 8, head_block=head_block).double(), ctx)
    with torch.no_grad():
        for l in (1, 2):
            model.model.layers[l].self_attn.normalizer.relation.b0.fill_(-0.5)
    model.marsea_ctx = ctx
    return model, ctx


def test_two_sites_in_one_layer_through_a_real_forward_on_both_paths():
    """The flagship B-1 test is tokenizer-gated and skips on CPU, so nothing drove a real two-heads-in-one-layer forward
    (review e982f83 §0).  A tiny random Qwen2 does: sites (2, 1) and (2, 6) sit in different head blocks at hb = 2, which
    is C-1's configuration, and every kept tensor must equal the unblocked dense reference."""
    from marsea.evaluate import teacher_forced_pass, _kept_head
    ids = torch.randint(0, 97, (1, 96)).tolist()[0]
    sites = [(2, 1), (2, 6)]
    ref_model, ref_ctx = _tiny_patched("dense", None)
    _, _, ref = teacher_forced_pass(ref_model, ref_ctx, ids[:-4], ids[-4:], 2, device="cpu", h_star=1, sites=[(2, 1)] )
    ref_full = {}
    for h in (1, 6):
        _, _, dg = teacher_forced_pass(ref_model, ref_ctx, ids[:-4], ids[-4:], 2, device="cpu", h_star=h, sites=[(2, h)])
        ref_full[h] = dg[2]
    for mode, hb in (("dense", None), ("chunked", None), ("chunked", 2), ("dense", 2)):
        model, ctx = _tiny_patched(mode, hb)
        _, _, dg = teacher_forced_pass(model, ctx, ids[:-4], ids[-4:], 2, device="cpu", h_star=1, sites=sites)
        d = dg[2]
        assert d.extra["head_sub"] == [1, 6] and d.A.shape[1] == 2, (mode, hb, d.extra.get("head_sub"), d.A.shape)
        for h in (1, 6):
            k, r = _kept_head(d, h), ref_full[h]
            for name in ("E", "A", "Atil", "a1", "p", "tau_j", "cbar_i", "kstar"):
                a, b = getattr(r, name)[:, 0], getattr(d, name)[:, k]
                if a.dtype == torch.bool or not a.is_floating_point():
                    assert torch.equal(a, b), (mode, hb, h, name)
                else:
                    assert (a - b).abs().max() <= 1e-9, (mode, hb, h, name, float((a - b).abs().max()))
    # C-3, AFTER the loop: inside it, e982f83 failed here on the first (dense) iteration and never reached the chunked,
    # head-blocked one this test exists for (review e16a843 D)
    with pytest.raises(KeyError):
        _kept_head(d, 3)


# --------------------------------------------------------------------------- e16a843 B/C: the razor-edge row
def _razor(n, gaps=None):
    gaps = gaps or [15.5 + 0.05 * i for i in range(41)]
    S = torch.zeros(len(gaps), n)
    S[:, 0] = torch.tensor(gaps)
    return S


@pytest.mark.parametrize("n_k", [8192, 16384])
def test_the_cap_never_binds_on_a_razor_edge_softmax_row_on_this_device(n_k):
    """The razor edge -- a spike ~16.6 nats over a flat tail just under half an ulp of it -- is where torch.softmax's
    lane-wise CPU kernel errs by n_k eps / 2L on the row sum (6e-5 at 8K here; the review's B-1).  Three rounds of slack
    constants were each right on one kernel.  The cap now decides on the relation's excess, so this passes on EVERY
    device: through the real normaliser, dense and chunked, with the relation forced empty and with a real relation
    calibrated to rho_0."""
    from marsea.normalizer import MarSeaNormalizer, State, row_softmax
    from marsea.chunked import marsea_chunked_attention
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    S = _razor(n_k).to(dev)
    A = row_softmax(S, torch.ones_like(S, dtype=torch.bool))
    worst = float((A.sum(-1, dtype=torch.float64) - 1).abs().max())
    print(f"[razor] {dev} torch.softmax row-sum error at n_k = {n_k}: {worst:.2e}")
    torch.manual_seed(0)
    D = 16
    norm = MarSeaNormalizer(D, 8).to(dev)
    # queries and keys that REPRODUCE the razor rows' scores: K[0] along e_1 with the spike, the rest orthogonal-ish
    n_q = S.shape[0]
    K = torch.zeros(1, 1, n_k, D, device=dev); K[0, 0, 0, 0] = 1.0
    K[0, 0, 1:, 1:] = torch.randn(n_k - 1, D - 1, device=dev) * 1e-3
    Q = torch.zeros(1, 1, n_q, D, device=dev); Q[0, 0, :, 0] = S[:, 0] * D ** 0.5
    vis = torch.ones(1, 1, n_q, n_k, dtype=torch.bool, device=dev)
    with torch.no_grad():
        Sqk = torch.matmul(Q, K.transpose(-1, -2)) * D ** -0.5
        assert torch.allclose(Sqk[0, 0, :, 0], S[:, 0]) and Sqk[0, 0, :, 1:].abs().max() < 1e-2
        # (a) the relation forced empty: the cap must be the identity bitwise, whatever the kernel's row sums
        A_out, d = norm.normalize(Sqk, vis, K, Q, State(), logits_override=torch.full((1, 1, 1, 1), -1e4, device=dev))
        assert int(d.cap_binds.sum()) == 0 and torch.equal(A_out, d.A_sm), "the cap bound on a razor row with no relation"
        _, dc = marsea_chunked_attention(norm, Q, K, K, vis, State(), chunk=1024, force_empty=True)
        assert int(dc.cap_binds.sum()) == 0 and dc.extra["pass2_rebuilt_row_frac"] == 0.0
        # (b) a live relation at rho_0: binding decided by the excess -- rows with a positive excess bind, others are the
        # identity; and the dense and chunked paths take the SAME decision on the same rows
        norm.relation.calibrate_b0(K, Q, vis, 0.05)
        A_out, d = norm.normalize(Sqk, vis, K, Q, State())
        ex = (d.Atil.double() - d.A_sm.double()).sum(-1)
        assert bool((d.cap_binds <= (ex > 1e-6)).all()), "a row bound whose relation added no mass"
        sub = ~d.cap_binds
        assert torch.equal(d.a1[sub], d.Atil[sub])
        # review 7814665 C: the cap never ADDS mass -- a1 <= Atil, the CAP's theta > 0 on every binding row (d.theta is
        # Stage 2's dual, non-negative by construction: asserting it said nothing -- review 9bacef9 E), no Stage-1 zero
        # lifted -- and a binding row lands on its own softmax mass, not on the literal 1 (whose fp32 realisation the
        # kernel may put below it)
        assert bool((d.a1 <= d.Atil).all()) and bool((d.cap_theta[d.cap_binds] > 0).all()) and bool((d.cap_theta[~d.cap_binds] == 0).all())
        assert int(((d.Atil == 0) & (d.a1 > 0)).sum()) == 0
        if d.cap_binds.any():
            tot = d.a1.sum(-1, dtype=torch.float64) - d.A_sm.sum(-1, dtype=torch.float64)
            assert float(tot[d.cap_binds].abs().max()) < 1e-5 * (n_k / 64) ** 0.5
        _, dc = marsea_chunked_attention(norm, Q, K, K, vis, State(), chunk=1024)
        near = (ex - 1e-6).abs() < 1e-5                       # the two softmax formulas differ by ~1e-6..6e-6 in the excess
        assert torch.equal(dc.cap_binds[~near], d.cap_binds[~near]), "dense and chunked disagree on a razor row off the boundary"


# --------------------------------------------------------------------------- 7814665 C: the cap never adds mass
def test_the_unit_cap_never_adds_mass_or_lifts_a_zero():
    """With the binding set decided by the relation, `proj_le(Atil, 1)` on a row whose fp32 total sits BELOW one (a
    kernel deficit larger than the relation's excess) solved the equality problem upward: theta < 0, Stage-1 zeros
    resurrected in a1, and a1's off-relation entries are A (review 7814665 C).  The cap now targets the row's own fp64
    softmax mass and the projection refuses to bind when its own arithmetic finds nothing above the target."""
    from marsea.normalizer import row_masses, unit_cap
    from marsea.primitives import proj_le_masked
    torch.manual_seed(0)
    A = torch.softmax(torch.randn(1, 1, 3, 4096) * 2, -1); A[..., :8] = 0
    A = A * torch.tensor([1 - 3e-5, 1 + 3e-5, 1.0]).view(1, 1, 3, 1) / A.sum(-1, keepdim=True)   # kernel deficit / surplus / exact
    vis = torch.ones_like(A, dtype=torch.bool); E = torch.zeros_like(vis); E[..., 8:16] = True
    Atil = A.clone(); Atil[E] += 2e-6 / 8                                     # the relation adds 2e-6 to every row
    ex, sm = row_masses(Atil, A, E)
    assert torch.allclose(ex, torch.full_like(ex, 2e-6), atol=1e-12) and torch.allclose(sm, A.sum(-1, dtype=torch.float64))
    a1, th, binds = unit_cap(Atil, vis, ex, sm)
    assert bool(binds.all()), binds
    assert bool((th > 0).all()) and bool((a1 <= Atil).all()), "the cap scaled a row up"
    assert int(((Atil == 0) & (a1 > 0)).sum()) == 0, "a Stage-1 zero was lifted"
    assert torch.allclose(a1.sum(-1, dtype=torch.float64), sm, atol=2e-7), "a binding row did not land on its softmax mass"
    # the old form, for the record: target 1 with a forced bind on the deficit row scales it UP
    a_old, th_old = proj_le_masked(Atil[..., :1, :], torch.ones(1, 1, 1), vis[..., :1, :], binds=torch.ones(1, 1, 1, dtype=torch.bool))
    assert torch.equal(a_old, Atil[..., :1, :]) or float(th_old) >= 0, "the guard did not stop an upward projection"
    # and a row the projection's own arithmetic finds at the target is left alone rather than scaled up
    v = A[..., :1, :] * (1 - 1e-4)
    a_g, th_g = proj_le_masked(v, v.sum(-1) + 1e-4, vis[..., :1, :], binds=torch.ones(1, 1, 1, dtype=torch.bool))
    assert torch.equal(a_g, v) and float(th_g) == 0.0
