"""
The TRIGGERED (causal) teacher-forced form of the normaliser (owner's decision, 2026-09-20).

The full-sequence form solves each column programme ONCE over every row of the sequence.  Under teacher forcing that
couples an answer row to LATER answer rows -- whose queries encode the gold tokens -- through four channels (the column's
member set, the inherited quota cbar_j, the column sparsemax, TauK's column statistics and nu), and it lets the answer
rows change the prompt rows' programmes.  Decoding (spec Sec. 6.4) cannot do either: the prompt is prefilled alone, and
at step t a column is re-solved over the members that exist then.  So the function that was trained is not the function
that generates, and the training form leaks.

Here the teacher-forced pass computes exactly what prefill + frozen-prefix decoding computes:

  rows < n_p  (the prompt)   the ordinary programmes, over the prompt rows and keys only  == the generation prefill
  row t >= n_p (an answer row), triggered by its own relation E[t, .]:
      cbar_j^(t)  = cbar_j^(prefill) + sum_{n_p <= t' <= t} E[t', j] A_sm[t', j]
      members(j)  = top-K_ret scores of ( prefill members of j  U  answer rows t' <= t with E[t', j] )
      p_tj        = row t's share of sparsemax(tau * members(j));  tau = the SEALED tau_j (prefill; or the key's arrival),
                    or -- trigger_tau -- heads.TauKTrigger's prediction from the column as it stands at this trigger
      Atil[t, j]  = cbar_j^(t) p_tj   on E,   A_sm[t, j] off it;   then the row programme on row t, as everywhere
  Earlier rows are never revised (their outputs stand; the KV cache above stays valid) and no row sees a later one.

`causal.decode_step` is the reference: tests/test_triggered.py drives it row by row and requires the same A.  All answer
rows are computed at once -- a sequentially truncated top-K list equals the top-K of the union, so row t's list is a
masked top-K over (prefill list || answer rows <= t) -- in blocks of rows to bound the [rows, active columns, K + m]
candidate tensor.  Gradients flow through every site the dense form has (the four straight-through sites, tau_j,
tau_i, the members' scores and the prefill quota).
"""
from __future__ import annotations
from typing import Optional
import torch

from .primitives import sparsemax_masked, proj_le_masked, _fp
from .relation import repeat_kv, straight_through
from .heads import column_stats, row_summary, trigger_stats

NEG_PAD = -1e30
NU_EPS = 1e-12


def normalize_triggered(norm, S: torch.Tensor, vis: torch.Tensor, K_kv: torch.Tensor, Q: torch.Tensor, state, n_prefill: int,
                        K_ret: int = 64, head_offset: int = 0, row_block: int = 32, logits_override: Optional[torch.Tensor] = None,
                        **kw):
    """S [B,H,n,n] scaled scores (square: a full teacher-forced pass), rows/keys < n_prefill are the prompt.
    Returns (A [B,H,n,n], Diagnostics) like MarSeaNormalizer._normalize_one."""
    from .normalizer import Diagnostics, State, broadcast_vis, row_softmax, row_masses, apply_unit_cap
    assert not kw, f"the triggered form does not take {sorted(kw)}"
    assert norm.quota_mode == "inherited" and abs(norm.alpha - 2.0) < 1e-12 and norm.gate == "st", \
        "the triggered form implements the main arm (inherited quota, sparsemax, straight-through gate)"
    B, H, n_q, n_k = S.shape
    assert n_q == n_k, "the triggered form is a full teacher-forced pass (prefill and decode have their own paths)"
    n_p, m = int(n_prefill), n_q - int(n_prefill)
    assert 0 < n_p < n_q
    out_dtype = S.dtype
    visb = broadcast_vis(vis, S).expand(B, H, n_q, n_k)
    g_kv = H // K_kv.shape[1]
    # ---- the prompt: exactly the generation prefill
    nu_prev = state.nu_prev if (state is not None and state.nu_prev is not None) else None
    st_p = State(nu_prev=None if nu_prev is None else nu_prev[..., :n_p])
    full_lo = logits_override is not None and tuple(logits_override.shape[-2:]) == (n_k, n_k)    # a whole [n, n] override: sliced
    norm._n_prefill_dense = n_p                                          # the prompt block: rows < n_p are prompt rows
    A_p, d_p = norm._normalize_one(S[:, :, :n_p, :n_p], visb[:, :, :n_p, :n_p], K_kv[:, :, :n_p], Q[:, :, :n_p], st_p,
                                   logits_override=(logits_override[..., :n_p, :n_p] if full_lo else logits_override), head_offset=head_offset)
    norm._n_prefill_dense = None
    with torch.autocast(device_type=S.device.type, enabled=False):
        S32 = _fp(S).masked_fill(~visb, float("-inf"))
        dt, dev = S32.dtype, S.device
        S_a = S32[:, :, n_p:, :]; vis_a = visb[:, :, n_p:, :]                       # [B,H,m,n]
        A_sm_a = row_softmax(S_a, vis_a)
        if logits_override is not None:
            logits_a = _fp(logits_override[..., n_p:, :] if full_lo else logits_override).expand(B, H, m, n_k)
        else:
            logits_a = norm.relation(K_kv, Q[:, :, n_p:], key_offset=0, n_k_total=n_k, head_offset=head_offset)
        E_a, g_a = norm.select_E(logits_a, vis_a, head_offset, first_row=n_p, n_prefill=n_p)
        # ---- tau_j: sealed at prefill for the prompt keys; an answer key is sealed at its arrival, from its one visible
        # entry S[t, t] and nu_prev = 1 (a one-member column has p = 1), as decode_step does
        diag_s = torch.diagonal(S32[:, :, n_p:, n_p:], dim1=-2, dim2=-1)           # [B,H,m]
        if norm.tau_j_global:
            tau_new = (norm.tau_min + torch.nn.functional.softplus(norm.tau_j_raw)).expand(B, H, m)
        else:
            stats_new = column_stats(diag_s.unsqueeze(-2), torch.ones(B, H, 1, m, dtype=torch.bool, device=dev))
            tau_new = norm.tauK(repeat_kv(_fp(K_kv[:, :, n_p:]), g_kv), stats_new, torch.ones(B, H, m, dtype=dt, device=dev), head_offset)
        tau_all = torch.cat([d_p.tau_j, tau_new], -1)                               # [B,H,n]
        # ---- the inherited quota, grown row by row (ST site 1)
        zeros_m = torch.zeros(B, H, m, dtype=dt, device=dev)
        cbar_run = torch.cat([d_p.cbar_j, zeros_m], -1).unsqueeze(-2) + torch.cumsum(g_a * A_sm_a, dim=-2)   # [B,H,m,n]
        # ---- each column's prefill list: its top-K_ret relation scores (what FrozenPrefixCache.init_from_prefill keeps)
        S_p = S32[:, :, :n_p, :n_p]
        k0 = min(K_ret, n_p)
        top0 = torch.topk(S_p.masked_fill(~d_p.E, NEG_PAD).transpose(-1, -2), k0, dim=-1)       # over the queries of each key
        v0 = torch.gather(d_p.E.transpose(-1, -2), -1, top0.indices)
        L_s = torch.full((B, H, n_k, K_ret), NEG_PAD, dtype=dt, device=dev); L_v = torch.zeros((B, H, n_k, K_ret), dtype=torch.bool, device=dev)
        L_s = torch.cat([torch.cat([top0.values.masked_fill(~v0, NEG_PAD), L_s[:, :, :n_p, k0:]], -1), L_s[:, :, n_p:]], -2)
        L_v = torch.cat([torch.cat([v0, L_v[:, :, :n_p, k0:]], -1), L_v[:, :, n_p:]], -2)
        S_aT = S_a.transpose(-1, -2); E_aT = E_a.transpose(-1, -2)                  # [B,H,n,m]: the answer rows of each key
        trig_tau = getattr(norm, "tauK_trig", None)
        learned_share = getattr(norm, "tail_share_head", None) is not None
        col_form = getattr(norm, "active_direction", None) == "column"
        if col_form:
            trig_tau = None                                                          # tau_col is a preset
        if trig_tau is not None or norm.tauQ.sized or learned_share or col_form:     # the TRUE size of each relation set, row by row
            cnt_run = torch.cat([d_p.E.sum(-2), torch.zeros(B, H, m, dtype=torch.long, device=dev)], -1).unsqueeze(-2) + torch.cumsum(E_a.long(), dim=-2)
        if trig_tau is not None:
            k_all = repeat_kv(_fp(K_kv), g_kv)                                      # [B,H,n,d]
        p_rows = []; tau_rows = []
        for r0 in range(0, m, row_block):
            r1 = min(m, r0 + row_block); rb = r1 - r0
            E_blk = E_a[:, :, r0:r1]                                                # [B,H,rb,n]
            A_max = int(E_blk.sum(-1).max())
            if A_max == 0:
                p_rows.append(torch.zeros(B, H, rb, n_k, dtype=dt, device=dev)); tau_rows.append(torch.zeros(B, H, rb, n_k, dtype=dt, device=dev)); continue
            order = torch.argsort(E_blk.to(torch.int8), dim=-1, descending=True, stable=True)[..., :A_max]   # the active columns first
            act = torch.gather(E_blk, -1, order)                                    # [B,H,rb,A]
            flat = order.reshape(B, H, rb * A_max)
            gat = lambda X: torch.gather(X, 2, flat.unsqueeze(-1).expand(B, H, rb * A_max, X.shape[-1])).reshape(B, H, rb, A_max, X.shape[-1])
            t_rel = torch.arange(r0, r1, device=dev)                                # the row's index among the answer rows
            seen = (torch.arange(r1, device=dev)[None, :] <= t_rel[:, None]).view(1, 1, rb, 1, r1)   # answer rows t' <= t
            cand = torch.cat([gat(L_s), gat(S_aT[..., :r1])], -1)                   # [B,H,rb,A,K+r1]
            cval = torch.cat([gat(L_v), gat(E_aT[..., :r1]) & seen], -1) & act.unsqueeze(-1)
            kk = min(K_ret, cand.shape[-1])
            top = torch.topk(cand.masked_fill(~cval, NEG_PAD), kk, dim=-1)
            tval = torch.gather(cval, -1, top.indices)
            if col_form:
                tau_c = torch.full((B, H, rb, A_max, 1), norm.tau_col, dtype=dt, device=dev)
            elif trig_tau is None:
                tau_c = torch.gather(tau_all.unsqueeze(2).expand(B, H, rb, n_k), -1, order).unsqueeze(-1)   # sealed
            else:                                                                   # predicted NOW, from the column as it stands
                # the PADDED slots (a row with fewer active columns than the block's widest) must reach the MLP with finite
                # features: a padded slot can sit on a column the row cannot see, whose score is -inf, and then tau is NaN
                # there -- masked out of the forward, but the backward of tau * values is 0 * NaN (P7, step 751, 2026-09-21)
                s_tr = torch.gather(S_a[:, :, r0:r1], -1, order).masked_fill(~act, 0.0)
                st = trigger_stats(top.values, tval, torch.gather(cnt_run[:, :, r0:r1], -1, order), s_tr)
                k_act = torch.gather(k_all, 2, flat.unsqueeze(-1).expand(B, H, rb * A_max, k_all.shape[-1]))
                tau_c = trig_tau(k_act, st.reshape(B, H, rb * A_max, 6), head_offset).reshape(B, H, rb, A_max, 1)
            p_list = sparsemax_masked(tau_c * top.values.masked_fill(~tval, 0.0), tval)
            own = (top.indices == (K_ret + t_rel).view(1, 1, rb, 1, 1)) & tval       # row t's own entry (absent: evicted -> 0)
            p_new = (p_list * own.to(dt)).sum(-1) * act.to(dt)                       # [B,H,rb,A]
            p_rows.append(torch.zeros(B, H, rb, n_k, dtype=dt, device=dev).scatter(-1, order, p_new))
            tau_rows.append(torch.zeros(B, H, rb, n_k, dtype=dt, device=dev).scatter(-1, order, (tau_c.squeeze(-1) * act.to(dt)).detach()))
        p_a = torch.cat(p_rows, -2)                                                 # [B,H,m,n]; 0 off E
        if col_form:
            # the column form (owner, 2026-09-22): quota = R_j + lam_col T_j over the rows so far; a non-member row of a
            # column with a relation is emitted at (1 - lam_col) A_sm
            tail_run = torch.cat([d_p.c - d_p.cbar_j, zeros_m], -1).unsqueeze(-2) + torch.cumsum((1.0 - g_a) * A_sm_a, dim=-2)
            budget_run = cbar_run + norm.lam_col * tail_run                          # [B,H,m,n]
            active = cnt_run > 0
            scale = torch.where(active & ~E_a, 1.0 - norm.lam_col, 1.0)
            Atil_a = A_sm_a * scale + g_a * (budget_run * p_a - A_sm_a * scale)      # on E: the column's assignment
            # no cap after the column programme (owner, 2026-09-22): the unit normalisation is the softmax at its start;
            # a one-end row that wins several columns carries more than one unit
            A_a = a1_a = Atil_a
            cap_theta_a = torch.zeros(B, H, m, dtype=dt, device=dev); cap_binds_a = torch.zeros(B, H, m, dtype=torch.bool, device=dev)
            cbar_i_a = (A_a * E_a.to(dt)).sum(-1); Rtil_a = (Atil_a * E_a.to(dt)).sum(-1)
            tau_i_a = torch.ones(B, H, m, dtype=dt, device=dev); theta_a = torch.zeros(B, H, m, dtype=dt, device=dev); u_a = p_a
        else:
            Atil_a = A_sm_a + g_a * (cbar_run * p_a - A_sm_a)                           # ST site 2
        # ---- the row programme, row-local: identical to the dense form's
        if not col_form:
          excess, sm_mass = row_masses(Atil_a, A_sm_a, E_a)
          a1_a, cap_theta_a, cap_binds_a = apply_unit_cap(norm, Atil_a, A_sm_a, E_a, vis_a, excess, sm_mass, Q=Q[:, :, n_p:],
                                                          col_size=cnt_run if learned_share else None, p=p_a, head_offset=head_offset)
          cbar_i_a = (g_a * a1_a).sum(-1)                                             # ST site 3
          Rtil_a = (Atil_a * E_a.to(dt)).sum(-1)
          if norm.tau_i_pinned:
              tau_i_a = torch.ones(B, H, m, dtype=dt, device=dev)
          else:
              tau_i_a = norm.tauQ(_fp(Q[:, :, n_p:]), norm.tauQ.summary(Atil_a, E_a, vis_a, cbar_i_a, Rtil_a, col_size=cnt_run if norm.tauQ.sized else None, p=p_a), head_offset)
          u_a, theta_a = proj_le_masked(Atil_a.masked_fill(~E_a, 0.0), cbar_i_a / tau_i_a, E_a)
          A_a = a1_a + g_a * (tau_i_a.unsqueeze(-1) * u_a - a1_a)                     # ST site 4
        # ---- the hand-off to the next patched layer: the prefill's nu for the prompt keys, 1 for a key sealed at arrival
        nu_all = torch.cat([d_p.nu, torch.ones(B, H, m, dtype=dt, device=dev)], -1)
        if state is not None:
            state.nu_next = nu_all

        def rows(top_block, bottom):                                                # [.., n_p, n_p] over [.., m, n] -> [.., n, n]
            if top_block is None or bottom is None:
                return None
            pad = torch.zeros(*top_block.shape[:-1], m, dtype=top_block.dtype, device=dev)
            return torch.cat([torch.cat([top_block, pad], -1), bottom.to(top_block.dtype)], -2)

        cat1 = lambda a, b: None if a is None else torch.cat([a, b.to(a.dtype)], -1)
        A = rows(_fp(A_p), A_a)
        kstar_a = (p_a > 0).sum(-2)                                                 # answer rows' entries per key
        diag = Diagnostics(E=rows(d_p.E, E_a), logits=rows(d_p.logits, logits_a) if d_p.logits is not None and d_p.logits.shape == d_p.E.shape else None,
                           cbar_j=cbar_run[:, :, -1, :], cbar_i=cat1(d_p.cbar_i, cbar_i_a), tau_j=tau_all, tau_i=cat1(d_p.tau_i, tau_i_a),
                           psi_j=None, kstar=torch.cat([d_p.kstar, torch.zeros(B, H, m, dtype=d_p.kstar.dtype, device=dev)], -1) + kstar_a,
                           nu=nu_all, theta=cat1(d_p.theta, theta_a), Rtil=cat1(d_p.Rtil, Rtil_a),
                           supp_rel=cat1(d_p.supp_rel, ((A_a > 0) & E_a).sum(-1)), cap_binds=cat1(d_p.cap_binds, cap_binds_a),
                           cap_theta=cat1(d_p.cap_theta, cap_theta_a), p=rows(d_p.p, p_a), a1=rows(d_p.a1, a1_a),
                           Atil=rows(d_p.Atil, Atil_a), A_sm=rows(d_p.A_sm, A_sm_a), c=None, u=rows(d_p.u, u_a), A=A)
        diag.extra["triggered"] = dict(n_prefill=n_p, K_ret=K_ret)
        diag.extra["tau_trig_rows"] = torch.cat(tau_rows, -2)                       # [B,H,m,n]: the tau of each triggered solve (monitor)
    return A.to(out_dtype), diag
