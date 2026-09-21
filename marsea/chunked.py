"""
Sec. 6.6: the exact chunked implementation (required for E3 at 16K, optional below 8K).

The fan-out program is column-local given the row normaliser; the fan-in program is row-local given Atil.
Off the relation every Atil entry IS an A_sm entry, reproducible from S and the row log-sum-exp.

  pass 0  lse_i = m_i + r_i, m_i = max_j S_ij, r_i = log sum_j exp(S_ij - m_i)   (chunked over keys, O(T) memory)
  pass 1  per key-chunk J: S[:,J] -> A_sm[:,J] = exp((S - m) - r); logits, E, g; cbar_J; column_stats; tau_J;
          p on the relation; Atil[:,J] (chunk-local dense) -> O_tilde += Atil[:,J] @ V[J];
          store SPARSE (b,h,i,j, Atil_ij, g_ij) for (i,j) in E; rowsum_i += sum_J Atil[i,J]; rowmax_i
  pass 2  rows with rowsum_i <= 1: a1 = Atil (identity); fan-in step 2 on the row's sparse entries only.
          rows with rowsum_i > 1: rebuild the dense row from S[i,:], lse_i and the sparse entries; proj_le
          exactly; fan-in step 2 with the quota from the dense a1.  The fraction of such rows is logged.
  output  O = O_tilde + sum_{(i,j) in E} (A_ij - Atil_ij) V_j, with cap-binding rows replaced by a1-based rows.

Peak memory: O(n_q * chunk) for the chunk-local dense tensors + O(|E|) for the sparse relation + the
cap-binding rows' dense rebuild.  Per-chunk gradient checkpointing (non-reentrant) keeps the backward within
the same budget; the ST gradient to the logits is DENSE on visible pairs and is recomputed per chunk.
T11 asserts chunked == dense to 1e-9 (fp64) / relative 1e-5 (fp32) on every tensor at T <= 4K.
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.utils.checkpoint as cp

from .primitives import sorted_prefix_stat, entmax_masked, proj_le_masked, sparsemax_masked_with_stats, _fp
from .relation import repeat_kv, straight_through
from .heads import column_stats, row_summary
from .normalizer import MarSeaNormalizer, State, Diagnostics, NU_EPS, CAP_TOL, cap_binds_from_excess, unit_cap, softmax_about_max


def _chunk_fanout(norm: MarSeaNormalizer, q: torch.Tensor, k_kv_J: torch.Tensor, v_kv_J: torch.Tensor,
                  vis_J: torch.Tensor, lse_m: torch.Tensor, lse_r: torch.Tensor, nu_prev_J: torch.Tensor, scaling: float,
                  force_empty: bool, j0: int, n_k_total: int, head_offset: int = 0):
    """One key chunk J.  Returns chunk-local outputs (all tensors, for checkpoint).

    torch.utils.checkpoint restores the AMBIENT autocast state when it recomputes, so the guard is repeated here: without
    it the backward's recompute would run in bf16 while the forward ran in fp32 (H1)."""
    with torch.autocast(device_type=q.device.type, enabled=False):
        return _chunk_fanout_inner(norm, q, k_kv_J, v_kv_J, vis_J, lse_m, lse_r, nu_prev_J, scaling, force_empty, j0, n_k_total,
                                   head_offset)


_softmax_about_max = softmax_about_max          # one formula for the dense, chunked and decode paths


def _chunk_fanout_inner(norm, q, k_kv_J, v_kv_J, vis_J, lse_m, lse_r, nu_prev_J, scaling, force_empty, j0, n_k_total, head_offset=0):
    B, H, n_q, d = q.shape
    g_kv = H // k_kv_J.shape[1]
    k_rep = repeat_kv(k_kv_J, g_kv)
    v_rep = repeat_kv(v_kv_J, g_kv)
    # scores in the BACKBONE's dtype first (bf16 under autocast), then fp32: exactly what the dense path receives
    S = (torch.matmul(q, k_rep.transpose(-1, -2)) * scaling).to(_fp(q).dtype)   # [B,H,n_q,|J|] fp32
    vis = vis_J.expand(B, H, n_q, S.shape[-1])
    S32 = S.masked_fill(~vis, float("-inf"))
    A_sm = torch.where(vis, _softmax_about_max(S32, lse_m, lse_r), torch.zeros_like(S))
    if force_empty:
        logits = torch.full_like(S, -1e4)
    else:
        logits = norm.relation(k_kv_J, q, key_offset=j0, n_k_total=n_k_total, head_offset=head_offset)  # D-31, chunk-aware
    E = (logits > 0) & vis
    g = straight_through(E, logits, vis, getattr(norm, "st_temperature", 1.0))
    # the ST gate's OFF-relation part (forward 0, backward sigmoid'): sites (3) and (4) of Sec. 4.3 read it on every
    # visible pair (cbar_i = sum_j g a1 ; A = a1 + g(tau u - a1) with u = 0 off E), so it is accumulated densely here
    st_off = (g - E.to(g.dtype)) * (~E).to(g.dtype)                              # == soft - soft.detach() off E
    cbar_J = (g * A_sm).sum(-2)                                                 # [B,H,|J|]
    stats = column_stats(S32, vis)
    if norm.tau_j_global:
        tau_J = (norm.tau_min + torch.nn.functional.softplus(norm.tau_j_raw)).expand(B, H, S.shape[-1])
    else:
        tau_J = norm.tauK(_fp(k_rep), stats, nu_prev_J, head_offset)
    zcol = (S32.masked_fill(~E, 0.0) * tau_J.unsqueeze(-2)).transpose(-1, -2)
    ET = E.transpose(-1, -2)
    if abs(norm.alpha - 2.0) < 1e-12:
        # one sort, not two.  The dense path got this in the 2026-09-12 round; the chunked path kept re-running
        # sorted_prefix_stat on the identical input -- once per key chunk -- and threw its k* away (the diagnostic
        # k* is the support count below).
        pT, _, psi_J = sparsemax_masked_with_stats(zcol, ET)
    else:
        pT = entmax_masked(zcol, ET, norm.alpha)
        with torch.no_grad():
            _, psi_J, _, _, _ = sorted_prefix_stat(zcol, ET)
    p = pT.transpose(-1, -2)
    Atil = A_sm + g * (cbar_J.unsqueeze(-2) * p - A_sm)                        # chunk-local dense
    p2 = (p * p).sum(-2)
    has_rel = E.sum(-2) > 0
    nu_J = torch.where(has_rel, 1.0 / torch.where(has_rel, p2.clamp_min(NU_EPS), torch.ones_like(p2)), torch.ones_like(p2))
    # value 0 in the forward; carries the off-relation gradient of sites (3)/(4) for rows the unit cap leaves alone
    # (a1 = Atil = A_sm off E there; cap-binding rows are corrected densely in pass 2)
    off_a1 = st_off * A_sm
    # the dense path casts A to the backbone dtype (bf16) before the value-mix and accumulates in fp32; rounding the
    # weights to q.dtype here (a no-op in fp32/fp64) makes the chunked value-mix use IDENTICAL weights
    wdt = q.dtype
    O_J = torch.matmul((Atil - off_a1).to(wdt).to(Atil.dtype), _fp(v_rep))     # [B,H,n_q,d]   site (4): A = a1 - g a1 off E
    cbar_off_J = off_a1.sum(-1)                                                 # site (3)
    rowsum_J = Atil.sum(-1)
    smsum_J = A_sm.sum(-1, dtype=torch.float64)                                 # the row's softmax mass, fp64 (row_masses)
    rowmax_J = Atil.masked_fill(~vis, 0.0).amax(-1)
    # sparse relation entries
    idx = E.nonzero(as_tuple=False)                                             # [nnz, 4] (b,h,i,jl)
    b_i, h_i, i_i, jl = idx.unbind(-1)
    Atil_s = Atil[b_i, h_i, i_i, jl]
    g_s = g[b_i, h_i, i_i, jl]
    Asm_s = A_sm[b_i, h_i, i_i, jl]
    S_s = S32[b_i, h_i, i_i, jl]                                                # the score itself: FrozenPrefixCache
    j_s = jl + j0
    return (O_J, rowsum_J, rowmax_J, cbar_J, tau_J, nu_J, psi_J, idx[:, :3], j_s, Atil_s, g_s, Asm_s,
            (p > 0).sum(-2).to(torch.long), cbar_off_J, S_s, smsum_J)


def marsea_chunked_attention(norm: MarSeaNormalizer, q: torch.Tensor, k_kv: torch.Tensor, v_kv: torch.Tensor,
                             vis: torch.Tensor, state: Optional[State] = None, chunk: int = 1024,
                             scaling: Optional[float] = None, force_empty: bool = False,
                             dense_outputs: bool = False, use_checkpoint: Optional[bool] = None,
                             dense_head=None, row_block: int = 1024, head_offset: int = 0,
                             want_e8: bool = False):
    """q [B,H,n_q,d] post-RoPE; k_kv, v_kv [B,H_kv,n_k,d]; vis [B,1,n_q,n_k] bool.
    Returns (O [B,H,n_q,d] fp32, Diagnostics).  With dense_outputs=True the diagnostics also carry dense
    A_sm, Atil, a1, A, p (T11 only; T <= 4K)."""
    # H1: the programs run in fp32 (or fp64), and bf16 loses the exact zeros the mechanism IS.  The dense normaliser
    # disables autocast around its body; this path is called straight from the patched layer, so it must do the same --
    # otherwise every matmul/einsum below is autocast-eligible and silently runs in bf16 under training or evaluation.
    with torch.autocast(device_type=q.device.type, enabled=False):
        hb = getattr(norm, "head_block", None)
        if hb and q.shape[1] > hb:
            # the head-axis analogue of the key chunking below: every program is per Q-head (spec Sec. 6.2), so the
            # heads split exactly.  Without this the chunked path -- the one S2 runs -- could not head-block at all.
            return _chunked_by_head_block(norm, q, k_kv, v_kv, vis, state, chunk, scaling, force_empty, dense_outputs,
                                          use_checkpoint, dense_head, row_block, hb, head_offset, want_e8)
        return _marsea_chunked_attention(norm, q, k_kv, v_kv, vis, state, chunk, scaling, force_empty, dense_outputs,
                                         use_checkpoint, dense_head, row_block, head_offset, want_e8)


def _cat_compatible(vals) -> bool:
    """every tensor has the same rank, dtype and shape outside the head axis (dim 1)."""
    r = vals[0]
    return all(v.dim() == r.dim() and v.dtype == r.dtype and v.dim() >= 2
               and v.shape[0] == r.shape[0] and v.shape[2:] == r.shape[2:] for v in vals)


def narrow_sparse(sp: dict, heads, H: int) -> dict:
    """the sparse store restricted to `heads` (an int or a list of GLOBAL head indices), re-indexed into kept-head space
    in that order -- the head space extra["head_sub"] names.  Used on COLLECTED diagnostics only, after the decode
    cache has been seeded from the full store: at 16K the full store is several GB per patched layer."""
    hl = [heads] if isinstance(heads, int) else list(heads)
    dev = sp["bhi"].device
    pos = torch.full((H,), -1, dtype=torch.long, device=dev)
    pos[torch.as_tensor(hl, dtype=torch.long, device=dev)] = torch.arange(len(hl), device=dev)
    keep = pos[sp["bhi"][:, 1]] >= 0
    out = {k: v[keep] for k, v in sp.items() if k != "bhi"}
    bhi = sp["bhi"][keep].clone(); bhi[:, 1] = pos[bhi[:, 1]]
    out["bhi"] = bhi
    return out


def _chunked_by_head_block(norm, q, k_kv, v_kv, vis, state, chunk, scaling, force_empty, dense_outputs,
                           use_checkpoint, dense_head, row_block, hb, head_offset=0, want_e8=False):
    from .relation import repeat_kv as _rk
    from .normalizer import State as _State, Diagnostics as _Diag, broadcast_vis as _bv
    B, H = q.shape[0], q.shape[1]
    g_kv = H // k_kv.shape[1]
    k_rep = _rk(k_kv, g_kv); v_rep = _rk(v_kv, g_kv)          # per Q-head: the blocks then run with g = 1
    vis_b = _bv(vis, (B, H, q.shape[2], k_kv.shape[2]))        # a shape, not a tensor: expand is a view, not 805 MB
    outs, diags, nus = [], [], []
    for h0 in range(0, H, hb):
        sl = slice(h0, min(H, h0 + hb))
        st = _State(nu_prev=(state.nu_prev[:, sl] if (state is not None and state.nu_prev is not None) else None))
        # dense_head is an int or a LIST of global head indices (two measurement sites can share a layer, review
        # 30372ae B-1); each block keeps the ones that fall inside it, in BLOCK-local indices
        if dense_head is None:
            dh = None
        elif isinstance(dense_head, int):
            dh = dense_head - h0 if h0 <= dense_head < h0 + hb else None
        else:
            inside = [x - h0 for x in dense_head if h0 <= x < h0 + hb]
            dh = inside or None
        # dense_outputs with NO dense_head means "every head" (B5's per-layer trace): each block then rebuilds its own
        # heads and the merge concatenates them.  This used to read `dense_outputs and dh is not None`, which is False
        # in every block when dense_head is None -- so E/A/Atil came back None and B5 raised KeyError under
        # --mode chunked --head_block k (review d5bd980).
        want_dense_b = dense_outputs and (dense_head is None or dh is not None)
        O_h, d_h = _marsea_chunked_attention(norm, q[:, sl], k_rep[:, sl], v_rep[:, sl], vis_b[:, sl], st, chunk,
                                             scaling, force_empty, want_dense_b,
                                             use_checkpoint, dh, row_block, head_offset + h0, want_e8)
        outs.append(O_h); diags.append(d_h); nus.append(st.nu_next)
    if state is not None and all(n is not None for n in nus):
        state.nu_next = torch.cat(nus, dim=1)
    merged = _Diag()
    # when a block narrowed its per-head tensors to the kept head(s), the blocks no longer agree on the head axis and
    # concatenating them produces a tensor that is neither (dim 1 IS the cat axis, so no shape check can catch it):
    # the head_sub loop below sets those fields instead.
    narrowed = any(d.extra.get("head_sub") is not None for d in diags)
    for name in (() if narrowed else ("cbar_j", "cbar_i", "tau_j", "tau_i", "psi_j", "kstar", "nu", "theta", "Rtil",
                                      "supp_rel", "cap_binds", "cap_theta", "E", "A_sm", "Atil", "a1", "A", "p", "c")):
        vals = [getattr(d, name) for d in diags]
        # concatenate along the head axis only when every block agrees on every OTHER axis.  The old guard
        # (len({shapes}) <= len(vals)) is vacuously true for any list, and the real test was a bare
        # `except RuntimeError: pass` that turned a shape mismatch into a silent None -- which is precisely what B-1
        # and the E8 loss fed on.  The legitimate mismatch is a dense rebuild that exists for ONE block only.
        if all(torch.is_tensor(v) for v in vals) and _cat_compatible(vals):
            merged.__setattr__(name, torch.cat(vals, dim=1))
    if narrowed:
        # Each narrowed block holds ITS kept heads; together they are the kept set.  This used to SET each field from
        # every narrowed block in turn -- the last block won while head_sub was stamped with the whole list, so with
        # two sites in different blocks A.shape[1] was 1, the value side was read from the coreference head's tensors
        # and the coreference column raised IndexError (review e982f83 C-1).  Concatenate along dim 1 in block order
        # (= ascending global head), then permute into dense_head's order, which is what head_sub promises.
        hl = [dense_head] if isinstance(dense_head, int) else list(dense_head)
        nd = [(bi, d) for bi, d in enumerate(diags) if d.extra.get("head_sub") is not None]
        got = [bi * hb + x for bi, d in nd
               for x in ([d.extra["head_sub"]] if isinstance(d.extra["head_sub"], int) else d.extra["head_sub"])]
        assert sorted(got) == sorted(hl), f"head blocks kept {got}, dense_head asked for {hl}"
        perm = torch.as_tensor([got.index(h) for h in hl], dtype=torch.long)
        def _cat_perm(vals):
            t = torch.cat(vals, dim=1)
            return t.index_select(1, perm.to(t.device))
        for name in ("E", "A_sm", "Atil", "a1", "A", "p", "cbar_j", "tau_j", "nu", "psi_j", "kstar", "cbar_i",
                     "tau_i", "theta", "Rtil", "supp_rel", "cap_binds", "cap_theta", "c"):
            vals = [getattr(d, name, None) for _, d in nd]
            if all(torch.is_tensor(v) for v in vals) and _cat_compatible(vals):
                setattr(merged, name, _cat_perm(vals))
        for nm in ("A_rowsum", "E_row_sizes", "rowsum", "excess", "sm_mass"):   # follow the attributes into head_sub space
            vals = [d.extra.get(nm) for _, d in nd]
            if all(torch.is_tensor(v) for v in vals) and _cat_compatible(vals):
                merged.extra[nm] = _cat_perm(vals)
        merged.extra["head_sub"] = dense_head
        # the decode cache is seeded from FULL-H per-key state: blocks that narrowed carry it in decode_seed, the
        # others still hold it in their attributes (C-4)
        seed = {}
        for name in ("tau_j", "cbar_j", "nu", "kstar"):
            vals = [(d.extra.get("decode_seed") or {}).get(name, getattr(d, name, None)) for d in diags]
            if all(torch.is_tensor(v) for v in vals):
                seed[name] = torch.cat(vals, dim=1)
        merged.extra["decode_seed"] = seed
    merged.extra["head_block"] = hb
    # count-weighted, not max: with hb = 2 and H = 12 the logged number was the MAXIMUM of six per-block means, and
    # this is the quantity the memory claim is calibrated on and the one read against the training procedure's
    # "<= 10 % is healthy" threshold (review d5bd980).
    _w = [d.extra.get("pass2_rows", 0) for d in diags]
    merged.extra["pass2_rebuilt_row_frac"] = (sum(d.extra.get("pass2_rebuilt_row_frac", 0.0) * w for d, w in zip(diags, _w))
                                              / max(1, sum(_w)))
    merged.extra["pass2_rows"] = sum(_w)
    merged.extra["nnz"] = sum(d.extra.get("nnz", 0) for d in diags)
    if all("sparse" in d.extra for d in diags):
        # merge the per-block sparse relations back into one, with the head index shifted out of block-local space:
        # E8 is computed from this (the chunked path never materialises E), so losing it loses E8 entirely.  The
        # blocks' stores are never narrowed (C-4), so every block's head index is block-local and `+ bi * hb` is right
        # for all of them -- with narrowed stores it re-globalised kept-space indices ({0, 4} for heads {3, 7}).
        parts = []
        for bi, d in enumerate(diags):
            sp = dict(d.extra["sparse"]); bhi = sp["bhi"].clone(); bhi[:, 1] = bhi[:, 1] + bi * hb
            sp["bhi"] = bhi; parts.append(sp)
        merged.extra["sparse"] = {k: torch.cat([p[k] for p in parts], 0) for k in parts[0]}
    for name in ("E_row_sizes", "A_rowsum", "rowsum", "excess", "sm_mass"):
        if not narrowed and all(torch.is_tensor(d.extra.get(name)) for d in diags):
            merged.extra[name] = torch.cat([d.extra[name] for d in diags], dim=1)
    if want_e8:
        from .fidelity import merge_e8_summaries
        e8 = merge_e8_summaries([d.extra.get("e8") for d in diags])
        if e8:
            merged.extra["e8"] = e8
    return torch.cat(outs, dim=1), merged


def _marsea_chunked_attention(norm, q, k_kv, v_kv, vis, state=None, chunk=1024, scaling=None, force_empty=False,
                              dense_outputs=False, use_checkpoint=None, dense_head=None, row_block=1024, head_offset=0,
                              want_e8=False):
    if getattr(norm, "cap_mode", "row") != "row":
        raise NotImplementedError("cap_mode = relation runs on the dense path only so far (--mode dense)")
    B, H, n_q, d = q.shape
    n_k = k_kv.shape[-2]
    dev = q.device
    dt = torch.float64 if q.dtype == torch.float64 else torch.float32
    scaling = scaling if scaling is not None else d ** -0.5
    vis = vis.bool()
    if vis.dim() == 4 and vis.shape[1] == 1:
        vis_b = vis
    else:
        vis_b = vis.expand(B, H, n_q, n_k)[:, :1]
    use_ckpt = torch.is_grad_enabled() if use_checkpoint is None else use_checkpoint
    g_kv = H // k_kv.shape[1]
    # ---------------- pass 0: row lse (running)
    m_run = torch.full((B, H, n_q), float("-inf"), dtype=dt, device=dev)
    s_run = torch.zeros((B, H, n_q), dtype=dt, device=dev)
    with torch.no_grad():
        for j0 in range(0, n_k, chunk):
            J = slice(j0, min(n_k, j0 + chunk))
            k_rep = repeat_kv(k_kv[:, :, J], g_kv)
            S = (torch.matmul(q, k_rep.transpose(-1, -2)) * scaling).to(dt)
            S = S.masked_fill(~vis_b[..., J].expand_as(S), float("-inf"))
            m_new = torch.maximum(m_run, S.amax(-1))
            m_safe = torch.where(torch.isfinite(m_new), m_new, torch.zeros_like(m_new))
            s_run = s_run * torch.exp(torch.where(torch.isfinite(m_run), m_run, m_safe) - m_safe) + torch.exp(S - m_safe.unsqueeze(-1)).sum(-1)
            m_run = m_new
        lse_m = torch.where(torch.isfinite(m_run), m_run, torch.zeros_like(m_run))          # the row max, exact
        lse_r = torch.where(torch.isfinite(m_run), torch.log(s_run.clamp_min(torch.finfo(dt).tiny)), torch.zeros_like(m_run))
    # the softmax must be differentiable in S: r is recomputed chunk-wise with grad on, about the (constant) row max,
    # so d r / d S is the softmax and d (S - m - r) / d S is the softmax Jacobian.
    if torch.is_grad_enabled():
        # the row log-sum-exp has to be differentiable in S, so it is recomputed chunk by chunk with grad on.  Each
        # chunk's exp() saves its own [B, H, n_q, |J|] output for the backward, and with |J| = chunk that is a full
        # dense [B, H, T, T] retained across the whole pass -- the chunked path's most avoidable quadratic term
        # (3.2 GB at 8K, 12.9 GB at 16K).  Recompute each chunk in the backward instead (review 2026-09-12, fix 3).
        def _lse_chunk(q_, k_J, vis_J, m_det_):
            with torch.autocast(device_type=q_.device.type, enabled=False):
                k_rep = repeat_kv(k_J, g_kv)
                S = (torch.matmul(q_, k_rep.transpose(-1, -2)) * scaling).to(dt)
                visJ = vis_J.expand_as(S)
                # mask BEFORE the exp, exactly as pass 0 does.  torch.where alone is not enough: exp() still runs on
                # the off-vis scores, and where's zero gradient meets exp's backward (grad_out * output) as 0 * inf =
                # NaN whenever an off-vis score exceeds the row max by more than ~88 (review d5bd980 C).
                S = S.masked_fill(~visJ, float("-inf"))
                return torch.where(visJ, torch.exp(S - m_det_.unsqueeze(-1)), torch.zeros_like(S)).sum(-1)
        acc = torch.zeros((B, H, n_q), dtype=dt, device=dev)
        m_det = lse_m.detach()
        for j0 in range(0, n_k, chunk):
            J = slice(j0, min(n_k, j0 + chunk))
            acc = acc + cp.checkpoint(_lse_chunk, q, k_kv[:, :, J], vis_b[..., J], m_det, use_reentrant=False)
        real = vis_b[:, 0].any(-1).unsqueeze(1).expand(B, H, n_q)
        # finfo(dt).tiny, not 1e-300: in fp32 that literal rounds to 0.0, so the clamp is inert, and a fully
        # invisible row's acc == 0 then gives log's backward 0.0 / 0.0 = NaN straight into q.grad and k.grad.
        # Unreachable at B = 1 without padding (every row is real) and invisible to the fp64 tests, which is why it
        # survived two rounds; it goes live the moment a batch is padded (review d5bd980 C).
        lse_r = torch.where(real, torch.log(acc.clamp_min(torch.finfo(dt).tiny)), torch.zeros_like(acc))
    nu_prev = _fp(state.nu_prev) if (state is not None and state.nu_prev is not None) else torch.ones(B, H, n_k, dtype=dt, device=dev)
    # ---------------- pass 1: key chunks
    assert norm.gate == "st", "the chunked path implements the straight-through gate only"
    # F-3: the uniform quota (E9's predicted-budget ablation) is a reduction over ALL key chunks -- mu / n_active over
    # the whole (b, h) slice -- which the single-pass chunked structure cannot compute.  Running it here would report
    # the INHERITED quota under the uniform label, so refuse: that arm runs dense (8K).
    assert norm.quota_mode == "inherited", ("the chunked path implements the inherited quota only; run the "
                                            "quota_mode='uniform' E9 arm dense (--mode dense, L = 8192)")
    # K_ret deploys the hierarchical top-K of prop:tournament (normalizer.py's flat-solve branch).  The chunked
    # fan-out solves entmax directly on each key chunk and has no root stage, so running the K_ret arm here would
    # measure the EXACT flat solve the ablation exists to compare against -- silently, and under the K_ret label.
    assert norm.K_ret is None, ("the chunked path implements the exact flat solve only; run the K_ret E9 arm dense "
                                "(--mode dense, L = 8192)")
    O = torch.zeros((B, H, n_q, d), dtype=dt, device=dev)
    cbar_off = torch.zeros((B, H, n_q), dtype=dt, device=dev)
    rowsum = torch.zeros((B, H, n_q), dtype=dt, device=dev)
    sm_mass = torch.zeros((B, H, n_q), dtype=torch.float64, device=dev)
    rowmax = torch.zeros((B, H, n_q), dtype=dt, device=dev)
    cbar_j, tau_j, nu, psi_j, kstar_c = [], [], [], [], []
    sp_bhi, sp_j, sp_Atil, sp_g, sp_Asm, sp_S = [], [], [], [], [], []
    for j0 in range(0, n_k, chunk):
        J = slice(j0, min(n_k, j0 + chunk))
        args = (norm, q, k_kv[:, :, J], v_kv[:, :, J], vis_b[..., J], lse_m, lse_r, nu_prev[..., J], scaling, force_empty, j0, n_k,
                head_offset)
        if use_ckpt:
            out = cp.checkpoint(_chunk_fanout, *args, use_reentrant=False)
        else:
            out = _chunk_fanout(*args)
        (O_J, rowsum_J, rowmax_J, cbar_J, tau_J, nu_J, psi_J, bhi, j_s, Atil_s, g_s, Asm_s, kc_J, cbar_off_J, S_s, smsum_J) = out
        O = O + O_J
        cbar_off = cbar_off + cbar_off_J
        rowsum = rowsum + rowsum_J
        sm_mass = sm_mass + smsum_J
        rowmax = torch.maximum(rowmax, rowmax_J)
        cbar_j.append(cbar_J); tau_j.append(tau_J); nu.append(nu_J); psi_j.append(psi_J); kstar_c.append(kc_J)
        sp_bhi.append(bhi); sp_j.append(j_s); sp_Atil.append(Atil_s); sp_g.append(g_s); sp_Asm.append(Asm_s); sp_S.append(S_s)
    cbar_j = torch.cat(cbar_j, -1); tau_j = torch.cat(tau_j, -1); nu = torch.cat(nu, -1)
    psi_j = torch.cat(psi_j, -1); kstar_c = torch.cat(kstar_c, -1)
    if state is not None:
        state.nu_next = nu
    bhi = torch.cat(sp_bhi, 0); jj = torch.cat(sp_j, 0)
    Atil_s = torch.cat(sp_Atil, 0); g_s = torch.cat(sp_g, 0); Asm_s = torch.cat(sp_Asm, 0); S_sp = torch.cat(sp_S, 0)
    nnz = jj.numel()
    # ---------------- pass 2: rows
    row_id = (bhi[:, 0] * H + bhi[:, 1]) * n_q + bhi[:, 2]                       # flat row id per sparse entry
    order = torch.argsort(row_id, stable=True)
    row_id, jj, Atil_s, g_s, Asm_s, S_sp = row_id[order], jj[order], Atil_s[order], g_s[order], Asm_s[order], S_sp[order]
    bhi = bhi[order]
    n_rows = B * H * n_q
    counts = torch.bincount(row_id, minlength=n_rows)                            # |E_i.|
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(nnz, device=dev) - starts[row_id]                        # position within the row
    L_max = int(counts.max()) if nnz else 0
    E_rows = counts > 0
    # rows the relation pushed over one unit (the cap binds): rebuild them densely.  Decided from the relation's own
    # entries -- sum_E (Atil - A_sm), fp64-accumulated, the dense path's relation_excess on the sparse store -- and never
    # from `rowsum`, whose fp32 total carries the softmax kernel's row-sum error (review e16a843 B).  Exactly 0 on rows
    # with no relation entry, so force_empty never binds.
    with torch.no_grad():
        excess = torch.zeros(n_rows, dtype=torch.float64, device=dev).index_add(0, row_id, Atil_s.double() - Asm_s.double())
    over = cap_binds_from_excess(excess)                                          # cast BEFORE subtracting: exact (D-1)
    binds_eff = over.clone()                                                     # minus the rows the projection left alone
    cap_theta = torch.zeros(n_rows, dtype=dt, device=dev)                        # the cap's own dual (0 where it did not bind)
    sm_flat = sm_mass.reshape(-1)
    # the fraction of ROWS pass 2 rebuilds densely -- a memory/throughput diagnostic over all B*H*n_q rows (pad rows
    # included, because they cost the same to rebuild).  E8's `frac_rows_over_unit` is a different quantity: per head,
    # over REAL rows.  They used to share a name.
    frac_over = float(over.float().mean())
    a1_s = Atil_s                                                                # identity on sub-unit rows
    O_flat = O.reshape(n_rows, d)
    if bool(over.any()):
        # PASS 2's densification.  The rows a column relation pushed over one unit are rebuilt densely and run through
        # the exact unit cap.  This used to gather k_rep_full[b_r, h_r] and v_rep_full[b_r, h_r] -- [n_over, n_k, d]
        # each, i.e. d = 128 times the [n_over, n_k] score tensor beside them -- so at 8K the block cost 8 GB at
        # frac_over = 1 % and 82 GB at the 10 % the training procedure Sec. 11 calls HEALTHY, and 4x that at 16K where
        # E3 lives.  Grouping the rows by (b, h) turns both gathers into plain matmuls against [n_k, d], and the rows of
        # each group are processed in blocks so the remaining [nb, n_k] term is bounded too.
        ridx = torch.nonzero(over).flatten()
        b_r = ridx // (H * n_q); h_r = (ridx // n_q) % H; i_r = ridx % n_q
        k_rep_full = repeat_kv(k_kv, g_kv)                                       # [B,H,n_k,d]; one copy, hoisted
        v_rep_full = repeat_kv(v_kv, g_kv)
        if not force_empty:
            g_kv_ = H // k_kv.shape[1]
            Kf = k_kv.to(norm.relation.U.weight.dtype if not norm.relation.per_head else norm.relation.U.dtype)
            if norm.relation.per_head:
                hs_g = slice(head_offset, head_offset + H)                   # global head space (head blocking)
                u_all = torch.einsum("bhjd,hdr->bhjr", repeat_kv(Kf, g_kv_), norm.relation.U[hs_g])
            else:
                u_all = repeat_kv(norm.relation.U(Kf), g_kv_)                    # [B,H,n_k,r]
        from .relation import sink_pair_mask, SINK_LOGIT
        sink_rows = sink_pair_mask(n_q, n_k, dev)
        cbar_off_flat = cbar_off.reshape(-1)
        # the per-block results are COLLECTED and applied once.  Doing an out-of-place index_copy into O_flat
        # ([n_rows, d]) and a `where` over the whole [nnz] a1 inside the loop made every iteration cost the size of
        # the WHOLE problem rather than the size of the block, which turned the bounded rebuild back into a
        # quadratic loop (review d5bd980 H6).
        up_row, up_O, up_cb, up_sel, up_a1 = [], [], [], [], []
        bh = (b_r * H + h_r)
        # the entries of a row are contiguous: they were sorted by row_id above, so starts/counts address them
        # directly and no [n_rows] scratch or [nnz] gather is needed per iteration
        for bh_id in torch.unique(bh).tolist():
            b, h = bh_id // H, bh_id % H
            grp = torch.nonzero(bh == bh_id).flatten()
            for s0 in range(0, grp.numel(), row_block):
                gsel = grp[s0:s0 + row_block]
                rb = ridx[gsel]; ib = i_r[gsel]; nb = gsel.numel()
                cnt_b = counts[rb]
                tot_b = int(cnt_b.sum())
                if tot_b:
                    off_b_ = torch.cumsum(cnt_b, 0) - cnt_b
                    lrow = torch.repeat_interleave(torch.arange(nb, device=dev), cnt_b)
                    sel_idx = starts[rb][lrow] + (torch.arange(tot_b, device=dev) - off_b_[lrow])
                    lcol = jj[sel_idx]
                else:
                    lrow = sel_idx = lcol = torch.zeros(0, dtype=torch.long, device=dev)
                q_b = q[b, h, ib]                                                # [nb, d]
                S_b = (torch.matmul(q_b, k_rep_full[b, h].transpose(-1, -2)) * scaling).to(dt)    # [nb, n_k]; no gather
                vis_bk = vis_b[b, 0, ib]                                         # [nb, n_k]
                Asm_b = torch.where(vis_bk, _softmax_about_max(S_b.masked_fill(~vis_bk, float("-inf")), lse_m[b, h, ib], lse_r[b, h, ib]),
                                    torch.zeros_like(S_b))
                if force_empty:
                    lg_b = torch.full_like(S_b, -1e4)
                else:
                    if norm.relation.per_head:
                        v_b = torch.einsum("rd,dk->rk", q_b.to(u_all.dtype), norm.relation.V[head_offset + h])
                    else:
                        v_b = norm.relation.V(q_b.to(u_all.dtype))               # [nb, r]
                    if norm.relation.key_only:
                        lg_b = (u_all[b, h].sum(-1) / (norm.relation.r ** 0.5) + norm.relation.b0_of(head_offset + h)).to(dt).expand(nb, n_k)
                    else:
                        lg_b = (torch.matmul(v_b, u_all[b, h].transpose(-1, -2)) / (norm.relation.r ** 0.5)
                                + norm.relation.b0_of(head_offset + h)).to(dt)   # [nb, n_k]; no gather
                lg_b = lg_b.masked_fill(sink_rows[ib], SINK_LOGIT)               # D-31 on the rebuilt rows
                E_b = torch.zeros_like(vis_bk).index_put((lrow, lcol), torch.ones(lrow.numel(), dtype=torch.bool, device=dev))
                T_st = float(getattr(norm, "st_temperature", 1.0))                  # the SAME backward temperature as the fan-out's gate
                soft_b = torch.sigmoid(lg_b / T_st) * vis_bk.to(dt)
                st_off_b = (soft_b - soft_b.detach()) * (~E_b).to(dt)
                # off E: A_sm with its ST term (site 2); on E: the sparse Stage-1 values.  The unit cap then runs
                # EXACTLY as in the dense path (its gradient mixes all entries of the row).
                Atil_b = (Asm_b * (1.0 - st_off_b)).index_put((lrow, lcol), Atil_s[sel_idx])
                a1_b, th_b, bnd_b = unit_cap(Atil_b, vis_bk, excess[rb], sm_flat[rb])         # the dense path's cap, exactly
                binds_eff[rb] = bnd_b; cap_theta[rb] = th_b.detach()
                off_b = st_off_b * a1_b                                          # sites (3)/(4) off the relation
                O_b = torch.matmul((a1_b - off_b).to(q.dtype).to(dt), v_rep_full[b, h].to(dt))    # [nb, d]; no gather
                up_row.append(rb); up_O.append(O_b); up_cb.append(off_b.sum(-1))
                up_sel.append(sel_idx); up_a1.append(a1_b[lrow, lcol])
        rb_all = torch.cat(up_row); sel_all = torch.cat(up_sel)
        O_flat = O_flat.index_copy(0, rb_all, torch.cat(up_O))
        cbar_off_flat = cbar_off_flat.index_copy(0, rb_all, torch.cat(up_cb))
        a1_s = Atil_s.index_copy(0, sel_all, torch.cat(up_a1))
        cbar_off = cbar_off_flat.reshape(B, H, n_q)
    # fan-in quantities per row from the sparse entries
    cbar_i = torch.zeros(n_rows, dtype=dt, device=dev).index_add(0, row_id, g_s * a1_s) + cbar_off.reshape(-1)   # site (3), both parts
    Rtil = torch.zeros(n_rows, dtype=dt, device=dev).index_add(0, row_id, Atil_s)
    # entropy of the relation part (row_summary's 4th feature)
    R_safe = torch.where(Rtil > 0, Rtil, torch.ones_like(Rtil))
    qq = Atil_s / R_safe[row_id]
    ent = torch.zeros(n_rows, dtype=dt, device=dev).index_add(0, row_id, -(qq * torch.log(qq.clamp_min(1e-30))))
    ent = torch.where(Rtil > 0, ent, torch.zeros_like(ent))
    summary = torch.stack([Rtil, cbar_i, rowmax.reshape(-1), ent], -1).reshape(B, H, n_q, 4)
    if norm.tau_i_pinned:
        tau_i = torch.ones(B, H, n_q, dtype=dt, device=dev)
    else:
        if norm.tauQ.sized:
            raise NotImplementedError("size_aware_tau_i runs on the dense path only so far (--mode dense)")
        tau_i = norm.tauQ(_fp(q), summary, head_offset)
    tau_i_flat = tau_i.reshape(-1)
    # ragged proj_le on the relation rows: pad to [n_active_rows, L_max]
    act = torch.nonzero(E_rows).flatten()
    if nnz:
        row_pos = torch.full((n_rows,), -1, dtype=torch.long, device=dev)
        row_pos[act] = torch.arange(act.numel(), device=dev)
        rp = row_pos[row_id]
        V = torch.zeros((act.numel(), L_max), dtype=dt, device=dev).index_put((rp, rank), Atil_s)
        M = torch.zeros((act.numel(), L_max), dtype=torch.bool, device=dev).index_put((rp, rank), torch.ones(nnz, dtype=torch.bool, device=dev))
        quota = (cbar_i / tau_i_flat)[act]
        u_pad, theta_act = proj_le_masked(V, quota, M)
        u_s = u_pad[rp, rank]
        A_s = a1_s + g_s * (tau_i_flat[row_id] * u_s - a1_s)                     # ST site (4)
        theta = torch.zeros(n_rows, dtype=dt, device=dev).index_put((act,), theta_act)
        # effective weight on E: bf16(a1) + (bf16(A) - bf16(a1)) == bf16(A) exactly in fp32 -- the dense path's weight.
        # The correction is accumulated in BLOCKS of the sparse axis: `w.unsqueeze(-1) * V[b,h,j]` is [nnz, d], i.e.
        # d = 128 floats per relation entry, so at 8K and 5 % coverage the single tensor is ~10 GB -- the same shape
        # of mistake as pass 2's old [n_over, n_k, d] gathers, on the path S2 trains.  Each block is checkpointed, or
        # its gathered values would stay saved for the backward and the blocking would buy nothing.
        v_rep_all = repeat_kv(v_kv, g_kv).to(dt)                                 # [B,H,n_k,d]; one copy, hoisted
        def _corr_block(w_b, bi_b, hi_b, j_b):
            return w_b.unsqueeze(-1) * v_rep_all[bi_b, hi_b, j_b]
        w_s = A_s.to(q.dtype).to(dt) - a1_s.to(q.dtype).to(dt)
        # 64M elements = 256 MB of fp32 per block (the old 4M was 16 MB) with 16x fewer checkpointed Python iterations
        # (~600 per layer per forward at 8K, each with its own launch overhead -- review 30372ae E).  It is NOT the old
        # memory target: the memory profiles predating this constant must be re-measured (review e982f83 I).
        nnz_block = max(1, (1 << 26) // max(int(d), 1))
        for s0 in range(0, nnz, nnz_block):
            sl = slice(s0, min(nnz, s0 + nnz_block))
            args_c = (w_s[sl], bhi[sl, 0], bhi[sl, 1], jj[sl])
            corr = cp.checkpoint(_corr_block, *args_c, use_reentrant=False) if (use_ckpt and w_s.requires_grad) \
                else _corr_block(*args_c)
            O_flat = O_flat.index_add(0, row_id[sl], corr)
        supp_rel = torch.zeros(n_rows, dtype=torch.long, device=dev).index_add(0, row_id, (A_s > 0).long())
    else:
        u_s = A_s = Atil_s
        theta = torch.zeros(n_rows, dtype=dt, device=dev)
        supp_rel = torch.zeros(n_rows, dtype=torch.long, device=dev)
    # the row's TOTAL final mass, for E8's frac_rows_zero_mass (D-31: a real row whose only visible key re-shaped its
    # quota away ends at zero).  Off the relation A == a1, and a1's row sum is rowsum(Atil) where the cap did not bind
    # and 1 where it did (a projection onto {sum = 1}) -- not clamp(rowsum, 1): a non-binding row's fp32 total may sit
    # above one by the softmax kernel's own error, and that is its mass.  On the relation it is the sparse A.
    with torch.no_grad():
        a1_row = torch.where(binds_eff, sm_flat.to(dt), rowsum.reshape(-1))
        A_rowsum = (a1_row
                    - torch.zeros(n_rows, dtype=dt, device=dev).index_add(0, row_id, a1_s.detach())
                    + torch.zeros(n_rows, dtype=dt, device=dev).index_add(0, row_id, A_s.detach()))
    O = O_flat.reshape(B, H, n_q, d)
    diag = Diagnostics(cbar_j=cbar_j, cbar_i=cbar_i.reshape(B, H, n_q), tau_j=tau_j, tau_i=tau_i, psi_j=psi_j, kstar=kstar_c,
                       nu=nu, theta=theta.reshape(B, H, n_q), Rtil=Rtil.reshape(B, H, n_q), supp_rel=supp_rel.reshape(B, H, n_q),
                       cap_binds=binds_eff.reshape(B, H, n_q), cap_theta=cap_theta.reshape(B, H, n_q), c=None,
                       extra={"pass2_rebuilt_row_frac": frac_over, "pass2_rows": n_rows, "nnz": nnz,
                              "E_row_sizes": counts.reshape(B, H, n_q),
                              "E_col_sizes": None, "A_rowsum": A_rowsum.reshape(B, H, n_q),
                              "rowsum": rowsum.detach(), "excess": excess.reshape(B, H, n_q), "sm_mass": sm_mass})
    diag.extra["sparse"] = dict(bhi=bhi, j=jj, Atil=Atil_s, a1=a1_s, A=A_s, g=g_s, A_sm=Asm_s, S=S_sp)
    if want_e8:
        # BEFORE the dense rebuild below narrows every per-head tensor to the kept head: the summary reads cbar_i's
        # head dimension against the sparse store's head INDICES, so a narrowed diag makes it read a one-head tensor
        # against an all-head store (review 30372ae E).
        from .fidelity import e8_summary_from_sparse
        diag.extra["e8"] = e8_summary_from_sparse(diag, vis_b, getattr(norm, "K_ret", None))
    from .normalizer import debug_enabled
    if debug_enabled():
        # MARSEA_DEBUG=1 used to have NO effect here: check_invariants was called from the dense normaliser only, and
        # invariant_report's first statement reads d.Atil.dtype, so it raised on a chunked Diagnostics rather than
        # degrading.  The invariants now run on the sparse store, i.e. on the path S2 trains (review d5bd980 F).
        from .invariants import check_invariants
        check_invariants(None, diag, vis_b, where=f"chunked(head_offset={head_offset})")
    if dense_outputs:
        # The fidelity statistics need the dense column/row at (l*, h*) -- and ONLY there.  Rebuilding all H heads is
        # what the chunked path exists to avoid: at 16K one [1, H, T, T] fp32 tensor is 12.9 GB, so E3 (the load-bearing
        # experiment, D-27, run at 16K) would OOM on its own diagnostics.  With dense_head set, one head is rebuilt and
        # the readers index it as [0, 0] (Diagnostics.extra["head_sub"], as on the dense path).
        if dense_head is None:
            hsel = slice(None); Hd = H
            keep = torch.ones_like(jj, dtype=torch.bool)
        else:
            hl = [dense_head] if isinstance(dense_head, int) else list(dense_head)
            hsel = torch.as_tensor(hl, dtype=torch.long, device=dev); Hd = len(hl)
            keep = torch.isin(bhi[:, 1], hsel)
        pos = torch.full((H,), -1, dtype=torch.long, device=dev)
        if dense_head is not None:
            pos[hsel] = torch.arange(Hd, device=dev)
        b_i = bhi[keep, 0]
        h_i = bhi[keep, 1] if dense_head is None else pos[bhi[keep, 1]]
        i_i = bhi[keep, 2]
        j_i = jj[keep]
        E = torch.zeros((B, Hd, n_q, n_k), dtype=torch.bool, device=dev)
        E[b_i, h_i, i_i, j_i] = True
        diag.E = E
        # the SAME convention as _chunk_fanout: matmul in the inputs' dtype (bf16 under a bf16 backbone), then fp32 --
        # otherwise the rebuilt A_sm would not match the one the chunks produced (and a bf16 q against an fp32 k is a
        # dtype error, which the fp32-only tests never saw)
        k_rep_full = repeat_kv(k_kv, g_kv)[:, hsel]
        S_full = (torch.matmul(q[:, hsel], k_rep_full.transpose(-1, -2)) * scaling).to(dt)
        visf = vis_b.expand(B, H, n_q, n_k)[:, hsel] if vis_b.shape[1] > 1 else vis_b.expand(B, Hd, n_q, n_k)
        A_sm = torch.where(visf, _softmax_about_max(S_full.masked_fill(~visf, float("-inf")), lse_m[:, hsel], lse_r[:, hsel]),
                           torch.zeros_like(S_full))
        Atil = A_sm.clone(); Atil[b_i, h_i, i_i, j_i] = Atil_s[keep]
        a1, _, _ = unit_cap(Atil, visf, excess.reshape(B, H, n_q)[:, hsel], sm_mass[:, hsel])   # the SAME cap pass 2 ran
        A = a1.clone(); A[b_i, h_i, i_i, j_i] = A_s[keep]
        p = torch.zeros_like(A_sm)
        cb = cbar_j[:, hsel].unsqueeze(-2).expand_as(A_sm)[b_i, h_i, i_i, j_i]
        p[b_i, h_i, i_i, j_i] = torch.where(cb > 0, Atil_s[keep] / torch.where(cb > 0, cb, torch.ones_like(cb)), torch.zeros_like(cb))
        diag.A_sm, diag.Atil, diag.a1, diag.A, diag.p, diag.c = A_sm, Atil, a1, A, p, A_sm.sum(-2)
        if dense_head is not None:                       # every per-head diagnostic follows the same head space
            # the per-key state a FrozenPrefixCache is seeded from is kept at FULL H: init_from_sparse builds the
            # decode cache for every head, and narrowing it (with the sparse store) left 7 of 8 heads' relation empty
            # (review e982f83 C-4).  References, not copies.
            diag.extra["decode_seed"] = {nm: getattr(diag, nm) for nm in ("tau_j", "cbar_j", "nu", "kstar")
                                         if torch.is_tensor(getattr(diag, nm, None))}
            for name in ("cbar_j", "tau_j", "nu", "psi_j", "kstar", "cbar_i", "tau_i", "theta", "Rtil", "supp_rel", "cap_binds",
                         "cap_theta"):
                t = getattr(diag, name, None)
                if torch.is_tensor(t) and t.dim() >= 2 and t.shape[1] == H:
                    setattr(diag, name, t[:, hsel].contiguous())
            # the extras follow too, or a later reader indexes a one-head tensor against an all-head store
            # (review 30372ae E).  E8 was already computed above, on the full-H data.
            for name in ("A_rowsum", "E_row_sizes", "rowsum", "excess", "sm_mass"):
                t = diag.extra.get(name)
                if torch.is_tensor(t) and t.dim() >= 2 and t.shape[1] == H:
                    diag.extra[name] = t[:, hsel].contiguous()
            # extra["sparse"] is deliberately NOT narrowed here: the backbone seeds the decode cache from it first, and
            # narrows the COLLECTED copy afterwards (narrow_sparse), which is where the retained memory is (C-4)
            diag.extra["head_sub"] = dense_head
    else:
        diag.E = None
    return O, diag
