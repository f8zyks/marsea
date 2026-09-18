import sys, os, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The queue and preflight scripts read their run configuration from the environment (L, CHUNK, HEAD_BLOCK, STEPS, ...),
# and the tests that run those scripts pin the DEFAULT launch lines and fixtures (a dense memory verdict at L = 8192,
# `--head_block 2` in every eval job).  On pod 1 the operator had L=4096 CHUNK=2048 HEAD_BLOCK=0 exported for the
# queue, preflight step 2 ran the suite in that shell, and four script tests failed on the operator's own knobs.  The
# suite must not depend on the shell that runs it: the knobs are removed from this process's environment BEFORE marsea
# is imported (the scripts are launched as subprocesses with dict(os.environ, ...); MARSEA_TOL_CAP is parsed when
# marsea.invariants is imported, and the acceptance tests test the default numerics -- an override belongs to a run).
RUN_KNOBS = ("L", "CHUNK", "HEAD_BLOCK", "STEPS", "MODE", "UNIFORM_L", "SKIP_DENSE_E9", "EXTENDED_E9",
             "ALLOW_PHASE_A_LAYER_CHANGE", "EVAL_EVERY", "N16K", "N16K_PAD", "E3DEPTH_ARMS", "E3DEPTH_SEEDS",
             "ALLOW_MISSING_ARMS", "SKIP_PAIRED_GATE", "GATE_GB", "REQUIRE_DATA", "EXPECT_NGPU", "MIN_GPU_MIB",
             "NEED_RULER", "DATA_RULER", "DATA_MUSIQUE", "HOTPOT_N", "EVAL", "PREFLIGHT_FORCE_DETECTOR", "EVAL_HEAD_BLOCK", "PHASE_A_ONLY", "JOBS_PER_GPU", "MARSEA_TOL_CAP")
STRIPPED = {k: os.environ.pop(k) for k in RUN_KNOBS if k in os.environ}


def pytest_report_header(config):
    if STRIPPED:
        return "run-configuration variables removed from the test environment: " + " ".join(
            f"{k}={v}" for k, v in sorted(STRIPPED.items()))

import numpy as np
import torch
import pytest

torch.set_default_dtype(torch.float32)


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)


def rand_case(rng, use_finfo_min=False, force_E=None, n_q=None, n_k=None, D=16, B=None, H=None, Hkv=None,
              rel_density=None, causal=True):
    """Random [B,H,n_q,n_k] case: offset-causal vis plus random extra masking (every real row keeps
    >= 1 visible key), per-batch padding, a random relation with empty rows/columns, random tau."""
    B = B or int(rng.integers(1, 3)); Hkv = Hkv or int(rng.integers(1, 3)); g = int(rng.integers(1, 3)); H = H or Hkv * g
    if H % Hkv: H = Hkv * g
    n_q = n_q or int(rng.integers(1, 41)); n_k = n_k or int(rng.integers(1, 11))
    i = torch.arange(n_q)[:, None]; j = torch.arange(n_k)[None, :]
    off = n_k - n_q
    vis = (j <= i + off) if causal else torch.ones(n_q, n_k, dtype=torch.bool)
    vis = vis & (torch.rand(n_q, n_k) > 0.15)
    for ii in range(n_q):
        if not vis[ii].any():
            vis[ii, max(0, min(n_k - 1, ii + off))] = True
    vis = vis[None, None].expand(B, 1, n_q, n_k).clone()
    if rng.random() < 0.3:                          # per-batch padding: drop a key entirely in one batch element
        b = int(rng.integers(0, B)); jj = int(rng.integers(0, n_k))
        vis[b, :, :, jj] = False
        for ii in range(n_q):
            if not vis[b, 0, ii].any():
                vis[b, :, ii, jj] = True
    scale = float(rng.uniform(0.3, 4.0))
    S = torch.randn(B, H, n_q, n_k) * scale
    neg = torch.finfo(torch.float32).min if use_finfo_min else float("-inf")
    S = S.masked_fill(~vis.expand_as(S), neg)
    K_kv = torch.randn(B, Hkv, n_k, D)
    Q = torch.randn(B, H, n_q, D)
    if force_E is None:
        dens = float(rng.uniform(0.0, 1.0)) if rel_density is None else rel_density
        E = (torch.rand(B, H, n_q, n_k) < dens)
        if rng.random() < 0.3:
            E[..., int(rng.integers(0, n_k))] = False
        if rng.random() < 0.3:
            E[..., int(rng.integers(0, n_q)), :] = False
        E = E & vis
    else:
        E = force_E.expand(B, H, n_q, n_k) & vis
    logits = torch.where(E, torch.rand(B, H, n_q, n_k) * 2 + 0.01, -(torch.rand(B, H, n_q, n_k) * 2 + 0.01))
    tau_j = torch.as_tensor(rng.uniform(0.05, 6.0, size=(B, H, n_k)), dtype=torch.float32)
    tau_i = torch.as_tensor(rng.uniform(0.05, 6.0, size=(B, H, n_q)), dtype=torch.float32)
    return dict(S=S, vis=vis, K_kv=K_kv, Q=Q, logits=logits, tau_j=tau_j, tau_i=tau_i, E=E, D=D)


def planted_column(rng, nq, m, delta_lo=0.2, spread=1.0):
    """verify_all.py::targets_scores -- a random column with a planted target set T (|T| = m) and a
    positive margin.  Returns (s [nq] np.float64, T set)."""
    T = set(rng.choice(nq, size=m, replace=False).tolist())
    s = np.empty(nq)
    hi = spread * rng.random(m)
    d = delta_lo + rng.random()
    lo = -d - spread * rng.random(nq - m)
    ti = sorted(T); ni = [i for i in range(nq) if i not in T]
    s[ti] = hi; s[ni] = lo
    return s, T


def interval(s, T):
    """Thm. 2's SHARP endpoints on the index set s is drawn over: (lo, hi, delta, W)."""
    T = sorted(T); m = len(T)
    smin = min(s[i] for i in T)
    W = float(sum(s[i] - smin for i in T))
    out = [s[i] for i in range(len(s)) if i not in T]
    delta = smin - (max(out) if out else -np.inf)
    lo = 1.0 / (W + m * delta) if np.isfinite(delta) else 0.0
    hi = (1.0 / W) if W > 0 else np.inf
    return lo, hi, delta, W
