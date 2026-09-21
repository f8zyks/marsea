"""
Sec. 7 (scaling and the causal form) and Sec. 6.4 (generation with the KV cache).

hierarchical_topk   prop:tournament -- per-block top-K by ORIGINAL score, ONE exact solve at the root;
                    exact whenever K_ret >= k*.  E8 must report k* against K_ret (truncation flag).
FrozenPrefixCache   decode-time state per patched layer (D-28): per key j, tau_j (sealed), cbar_j
                    (incremental), and the relation's score list bounded by K_ret.  At decode step t only
                    row t's entries are emitted; past rows are never revised (prop:prefix(iii): the
                    frozen-prefix variant over-admits, never under-admits).
seal-at-boundary    the E9 alternative emission policy: tau_j / relation list re-sealed when a block of
                    `block_size` tokens closes; frozen inside the open block.
Training uses the full-sequence teacher-forced form (no sealing); this module is decode + T7/T8.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import torch

from .primitives import sparsemax_masked, sorted_prefix_stat, proj_le_masked, _fp, NEG_PAD
from .normalizer import MarSeaNormalizer, State, Diagnostics, row_softmax, broadcast_vis, CAP_TOL, row_masses, unit_cap
from .heads import column_stats, row_summary
from .relation import repeat_kv, straight_through

K_RET_DEFAULT = 64
BLOCK_DEFAULT = 512


# --------------------------------------------------------------------------- prop:tournament
def hierarchical_topk(z: torch.Tensor, mask: torch.Tensor, K_ret: int = K_RET_DEFAULT,
                      block_size: int = BLOCK_DEFAULT):
    """sparsemax over the last dim of z (restricted to mask) computed as: per-block top-K_ret by ORIGINAL
    score, then ONE exact sparsemax at the root over the survivors.  Exact iff K_ret >= k* (Prop. tournament).
    z is ALREADY multiplied by tau_j (the tau scaling is monotone, so top-K by tau*z == top-K by z for
    tau > 0; the caller passes tau_j * S as elsewhere).
    returns p [..., L] (exact zeros; 0 for non-survivors), kstar [...], truncated [...] (k* >= 0.9 K_ret)."""
    z = _fp(z)
    mask = mask.bool()
    L = z.shape[-1]
    lead = z.shape[:-1]
    nb = (L + block_size - 1) // block_size
    Lp = nb * block_size
    K = min(K_ret, block_size)                                                # per-block retention
    zp = torch.full((*lead, Lp), NEG_PAD, dtype=z.dtype, device=z.device)
    zp[..., :L] = z.masked_fill(~mask, NEG_PAD)
    mp = torch.zeros((*lead, Lp), dtype=torch.bool, device=z.device)
    mp[..., :L] = mask
    zb = zp.reshape(*lead, nb, block_size)
    mb = mp.reshape(*lead, nb, block_size)
    top = torch.topk(zb, K, dim=-1, largest=True, sorted=True)                    # per block, by original score
    surv_idx = (top.indices + torch.arange(nb, device=z.device).view(*([1] * len(lead)), nb, 1) * block_size)
    surv_idx = surv_idx.reshape(*lead, nb * K)
    surv_val = top.values.reshape(*lead, nb * K)
    surv_ok = torch.gather(mb.reshape(*lead, Lp), -1, surv_idx)
    # the ROOT of the tournament retains K survivors (the global top-K by original score, since every block's
    # top-K contains any element of the global top-K) and runs ONE exact solve on them.  Under-retention
    # (K < k*) therefore yields the top-K prefix of the sorted order (Prop. tournament(ii) / harness Prop. 22).
    K_root = min(K_ret, surv_val.shape[-1])                                   # the root retains K_ret
    root = torch.topk(surv_val.masked_fill(~surv_ok, NEG_PAD), K_root, dim=-1, largest=True, sorted=True)
    root_idx = torch.gather(surv_idx, -1, root.indices)
    root_ok = torch.gather(surv_ok, -1, root.indices)
    root_val = root.values
    p_s = sparsemax_masked(root_val, root_ok)                                     # ONE exact solve at the root
    with torch.no_grad():
        kstar, _, _, _, _ = sorted_prefix_stat(root_val, root_ok)
    p = torch.zeros((*lead, Lp), dtype=z.dtype, device=z.device)
    p = p.scatter(-1, root_idx, p_s)
    p = p[..., :L]
    truncated = kstar >= 0.9 * K_ret
    return p, kstar, truncated


# --------------------------------------------------------------------------- decode-time cache
@dataclass
class FrozenPrefixCache:
    """One per patched layer.  Shapes: [B,H,n_k] per key; relation score lists [B,H,n_k,K_ret]."""
    K_ret: int = K_RET_DEFAULT
    block_size: int = BLOCK_DEFAULT
    policy: str = "frozen_prefix"                  # or "seal_at_boundary"
    tau_j: Optional[torch.Tensor] = None           # [B,H,n_k]   sealed
    cbar_j: Optional[torch.Tensor] = None          # [B,H,n_k]   incremental inherited quota
    rel_scores: Optional[torch.Tensor] = None      # [B,H,n_k,K_ret]  top-K_ret relation scores S[i,j], i in E_.j
    rel_valid: Optional[torch.Tensor] = None       # [B,H,n_k,K_ret]  bool
    rel_qidx: Optional[torch.Tensor] = None        # [B,H,n_k,K_ret]  long, the query index of each stored score
    nu: Optional[torch.Tensor] = None              # [B,H,n_k]   last nu (for the next layer's TauK on new keys)
    rel_count: Optional[torch.Tensor] = None       # [B,H,n_k]   the TRUE size of E_.j so far (the lists are truncated at K_ret)
    n_seen: int = 0                                # queries seen so far (prefix length)
    Q_hist: Optional[torch.Tensor] = None          # [B,H,T,d] post-RoPE queries (seal-at-boundary only)
    # Two different events, counted separately (review e982f83 E).  One integer used to add them, so E8's "k* near
    # K_ret" reading and D-28's cache-loss reading could not be told apart -- and fixing the dense/chunked parity
    # without splitting it only made both paths agree on a number that meant two things.
    eviction_events: int = 0                       # a relation entry LOST from a column's top-K_ret list (cache lossiness)
    near_truncation_events: int = 0                # a column whose support k* reached 0.9 K_ret (hierarchy exactness)
    extra: dict = field(default_factory=dict)

    @property
    def truncation_events(self) -> int:
        """the old conflated sum, kept for callers that print it; report the two counters, never this."""
        return self.eviction_events + self.near_truncation_events

    # ---- prefill from a dense pass
    @torch.no_grad()
    def init_from_prefill(self, S32: torch.Tensor, diag: Diagnostics, Q: Optional[torch.Tensor] = None):
        """S32 [B,H,n_q,n_k] fp32 scores (-inf off vis); diag from the dense normaliser on the prompt."""
        E = diag.E
        B, H, n_q, n_k = S32.shape
        K = self.K_ret
        self.tau_j = diag.tau_j.detach().clone()
        self.cbar_j = diag.cbar_j.detach().clone()
        self.nu = diag.nu.detach().clone()
        Srel = S32.masked_fill(~E, NEG_PAD).transpose(-1, -2)                  # [B,H,n_k,n_q]
        k = min(K, n_q)
        top = torch.topk(Srel, k, dim=-1, largest=True, sorted=True)
        valid = torch.gather(E.transpose(-1, -2), -1, top.indices)
        self.rel_scores = torch.full((B, H, n_k, K), NEG_PAD, dtype=S32.dtype, device=S32.device)
        self.rel_valid = torch.zeros((B, H, n_k, K), dtype=torch.bool, device=S32.device)
        self.rel_qidx = torch.full((B, H, n_k, K), -1, dtype=torch.long, device=S32.device)
        self.rel_scores[..., :k] = top.values.masked_fill(~valid, NEG_PAD)
        self.rel_valid[..., :k] = valid
        self.rel_qidx[..., :k] = top.indices.masked_fill(~valid, -1)
        self.rel_count = E.sum(-2)
        self.n_seen = n_q
        # a column whose relation exceeds K_ret loses entries to the list: the SAME event _insert_scores counts at
        # decode time, and the one the sparse prefill counts.  Without it the two prefills disagreed on a reported
        # D-28/E8 reading, so truncation_events was incomparable between --mode dense and --mode chunked
        # (review 30372ae B-2); the sparse side is the one consistent with decode bookkeeping, so this is the side
        # that changed.
        self.eviction_events += int((E.transpose(-1, -2).sum(-1) > K).sum())
        if diag.kstar is not None:
            self.near_truncation_events += int((diag.kstar >= 0.9 * K).sum())
        if self.policy == "seal_at_boundary" and Q is not None:
            self.Q_hist = Q.detach().float().clone()

    # ---- prefill from the CHUNKED pass (D-28 on the path E3 actually runs)
    @torch.no_grad()
    def init_from_sparse(self, diag: Diagnostics, B: int, H: int, n_q: int, n_k: int, device, dtype=torch.float32):
        """Same state as init_from_prefill, built from the chunked path's SPARSE relation store.

        The chunked path never materialises S or E -- that is what it exists to avoid -- so the dense prefill could
        not run there, and `backbone` returned before seeding the cache at all.  Every decode step then re-entered
        the chunked forward with n_q = 1 and recomputed tau_j LIVE from a one-row column_stats (every column at
        n_vis = 1: the degenerate branch, second max := max, gap 0, std clamped) instead of the value sealed at
        prefill, and cbar_j from row t alone instead of the accumulated column.  That is D-28 violated silently, in
        the generation mode of the load-bearing experiment (review d5bd980 B-3).

        The relation's per-column top-K_ret is taken by sorting the sparse entries by (column, -score): the dense
        form's torch.topk over a [B,H,n_k,n_q] tensor is exactly what cannot be built at 16K."""
        sp = diag.extra["sparse"]
        K = self.K_ret
        # the FULL-H per-key state: when the prefill also kept measurement heads, the diagnostics' own fields are
        # narrowed to them, and seeding from those left 7 of 8 heads without a relation (review e982f83 C-4)
        seed = diag.extra.get("decode_seed") or {nm: getattr(diag, nm) for nm in ("tau_j", "cbar_j", "nu", "kstar")}
        for nm in ("tau_j", "cbar_j", "nu"):
            t = seed.get(nm)
            if not (torch.is_tensor(t) and tuple(t.shape) == (B, H, n_k)):
                raise ValueError(f"init_from_sparse: {nm} has shape {None if t is None else tuple(t.shape)}, the cache "
                                 f"needs {(B, H, n_k)} -- seeding from narrowed diagnostics")
        if sp["bhi"].numel() and int(sp["bhi"][:, 1].max()) >= H:
            raise ValueError("init_from_sparse: the sparse store's head index exceeds H")
        self.tau_j = seed["tau_j"].detach().clone()
        self.cbar_j = seed["cbar_j"].detach().clone()
        self.nu = seed["nu"].detach().clone()
        self.rel_scores = torch.full((B, H, n_k, K), NEG_PAD, dtype=dtype, device=device)
        self.rel_valid = torch.zeros((B, H, n_k, K), dtype=torch.bool, device=device)
        self.rel_qidx = torch.full((B, H, n_k, K), -1, dtype=torch.long, device=device)
        self.rel_count = torch.zeros((B, H, n_k), dtype=torch.long, device=device)
        bhi, jj, S_s = sp["bhi"], sp["j"], sp["S"]
        if jj.numel():
            col = (bhi[:, 0] * H + bhi[:, 1]) * n_k + jj                       # flat column id per relation entry
            # rank within the column by DESCENDING score; a stable sort on the query index first makes ties resolve
            # by query order, as torch.topk(sorted=True) does on the dense path
            o1 = torch.argsort(bhi[:, 2], stable=True)
            col, S_o, qi = col[o1], S_s[o1].to(dtype), bhi[o1, 2]
            o2 = torch.argsort(S_o, descending=True, stable=True)
            col, S_o, qi = col[o2], S_o[o2], qi[o2]
            o3 = torch.argsort(col, stable=True)                                # stable: keeps the score order inside
            col, S_o, qi = col[o3], S_o[o3], qi[o3]
            counts = torch.bincount(col, minlength=B * H * n_k)
            starts = torch.cumsum(counts, 0) - counts
            rank = torch.arange(col.numel(), device=device) - starts[col]
            keep = rank < K
            c_k, r_k = col[keep], rank[keep]
            flat_s = self.rel_scores.reshape(-1, K); flat_v = self.rel_valid.reshape(-1, K); flat_q = self.rel_qidx.reshape(-1, K)
            flat_s[c_k, r_k] = S_o[keep]
            flat_v[c_k, r_k] = True
            flat_q[c_k, r_k] = qi[keep]
            self.rel_scores = flat_s.reshape(B, H, n_k, K)
            self.rel_valid = flat_v.reshape(B, H, n_k, K)
            self.rel_qidx = flat_q.reshape(B, H, n_k, K)
            self.eviction_events += int((counts > K).sum())
            self.rel_count = counts.reshape(B, H, n_k)
        self.n_seen = n_q
        if torch.is_tensor(seed.get("kstar")):
            self.near_truncation_events += int((seed["kstar"] >= 0.9 * K).sum())

    @torch.no_grad()
    def _append_new_key(self, tau_new: torch.Tensor, cbar_new: torch.Tensor, nu_new: torch.Tensor):
        """extend every per-key tensor by one key (the token just decoded).  tau_new [B,H,1]."""
        B, H, n_k, K = self.rel_scores.shape
        dev = self.rel_scores.device
        self.tau_j = torch.cat([self.tau_j, tau_new], -1)
        self.cbar_j = torch.cat([self.cbar_j, cbar_new], -1)
        self.nu = torch.cat([self.nu, nu_new], -1)
        if self.rel_count is not None:
            self.rel_count = torch.cat([self.rel_count, torch.zeros((B, H, 1), dtype=self.rel_count.dtype, device=dev)], -1)
        self.rel_scores = torch.cat([self.rel_scores, torch.full((B, H, 1, K), NEG_PAD, dtype=self.rel_scores.dtype, device=dev)], -2)
        self.rel_valid = torch.cat([self.rel_valid, torch.zeros((B, H, 1, K), dtype=torch.bool, device=dev)], -2)
        self.rel_qidx = torch.cat([self.rel_qidx, torch.full((B, H, 1, K), -1, dtype=torch.long, device=dev)], -2)

    @torch.no_grad()
    def _insert_scores(self, s_new: torch.Tensor, E_new: torch.Tensor, qidx: int):
        """insert S[t, j] into key j's top-K list for every j with E[t,j] (replace the minimum if full)."""
        # candidate list = [stored..., new]; keep the top-K by score among valid entries
        B, H, n_k, K = self.rel_scores.shape
        cand = torch.cat([self.rel_scores, s_new.unsqueeze(-1).masked_fill(~E_new.unsqueeze(-1), NEG_PAD)], -1)
        cval = torch.cat([self.rel_valid, E_new.unsqueeze(-1)], -1)
        cidx = torch.cat([self.rel_qidx, torch.full((B, H, n_k, 1), qidx, dtype=torch.long, device=cand.device)], -1)
        top = torch.topk(cand.masked_fill(~cval, NEG_PAD), K, dim=-1, largest=True, sorted=True)
        self.rel_scores = top.values
        self.rel_valid = torch.gather(cval, -1, top.indices)
        self.rel_qidx = torch.gather(cidx, -1, top.indices).masked_fill(~self.rel_valid, -1)
        self.rel_scores = self.rel_scores.masked_fill(~self.rel_valid, NEG_PAD)
        # an insertion that evicted a valid entry from a FULL list means the key's relation exceeds K_ret
        evicted = (cval.sum(-1) > K)
        self.eviction_events += int(evicted.sum())


@torch.no_grad()
def decode_step(norm: MarSeaNormalizer, cache: FrozenPrefixCache, S_row: torch.Tensor, vis_row: torch.Tensor,
                K_kv: torch.Tensor, q_t: torch.Tensor, nu_prev_new_key: Optional[torch.Tensor] = None,
                logits_row: Optional[torch.Tensor] = None):
    """One decode step (spec Sec. 6.4).  S_row [B,H,1,n_k] scaled scores of the new query t against ALL cached
    keys (the last key is the new token itself); vis_row [B,1,1,n_k]; K_kv [B,H_kv,n_k,d]; q_t [B,H,1,d].
    Returns (A_row [B,H,1,n_k] fp32, Diagnostics-lite dict).  Only row t is emitted; the cache is advanced."""
    vis = broadcast_vis(vis_row, S_row)
    B, H, _, n_k = S_row.shape
    g_kv = H // K_kv.shape[1]
    S32 = _fp(S_row).masked_fill(~vis, float("-inf"))
    A_sm = row_softmax(S32, vis)                                                 # [B,H,1,n_k]
    logits = norm.relation(K_kv, q_t, key_offset=0, n_k_total=n_k) if logits_row is None else _fp(logits_row)
    E_row = (logits > 0) & vis                                                   # pairwise: no past row changes
    t = cache.n_seen                                                             # index of the new query
    # ---- the new token's own key: seal tau_j now (its visible column is {t}); cbar_j = A_sm[t,t] if E[t,t]
    n_new = n_k - cache.tau_j.shape[-1]
    if n_new > 0:
        assert n_new == 1, "decode_step handles one new token at a time"
        col_new = S32[..., -1:]                                                  # [B,H,1,1]
        vis_new = vis[..., -1:]
        stats = column_stats(col_new, vis_new)                                   # n_vis = 1: guarded stats
        nu_prev = nu_prev_new_key if nu_prev_new_key is not None else torch.ones(B, H, 1, dtype=S32.dtype, device=S32.device)
        if norm.tau_j_global:
            tau_new = (norm.tau_min + torch.nn.functional.softplus(norm.tau_j_raw)).expand(B, H, 1)
        else:
            k_new = repeat_kv(_fp(K_kv[..., -1:, :]), g_kv)
            tau_new = norm.tauK(k_new, stats, nu_prev)
        cache._append_new_key(tau_new, torch.zeros(B, H, 1, dtype=S32.dtype, device=S32.device),
                              torch.ones(B, H, 1, dtype=S32.dtype, device=S32.device))
    # ---- incremental inherited quota and the per-key re-solve at FROZEN tau_j
    E_j = E_row[..., 0, :]                                                       # [B,H,n_k]
    s_j = S32[..., 0, :].masked_fill(~vis[..., 0, :], 0.0)
    cache.cbar_j = cache.cbar_j + E_j.to(S32.dtype) * A_sm[..., 0, :]
    cache._insert_scores(s_j, E_j, t)
    if cache.rel_count is not None:
        cache.rel_count = cache.rel_count + E_j.long()
    tau_use = cache.tau_j
    if getattr(norm, "tauK_trig", None) is not None:
        # the trigger-time temperature: predicted from the column as it stands now, for the columns row t triggers
        from .heads import trigger_stats
        st = trigger_stats(cache.rel_scores, cache.rel_valid, cache.rel_count, s_j)
        tau_use = torch.where(E_j, norm.tauK_trig(repeat_kv(_fp(K_kv), g_kv), st), cache.tau_j)
    z = tau_use.unsqueeze(-1) * cache.rel_scores.masked_fill(~cache.rel_valid, 0.0)
    p_list = sparsemax_masked(z, cache.rel_valid)                                # [B,H,n_k,K]
    is_new = (cache.rel_qidx == t) & cache.rel_valid
    p_new = (p_list * is_new.to(p_list.dtype)).sum(-1)                           # 0 if evicted / not in relation
    p2 = (p_list * p_list).sum(-1)
    has_rel = cache.rel_valid.any(-1)
    cache.nu = torch.where(has_rel, 1.0 / torch.where(has_rel, p2.clamp_min(1e-12), torch.ones_like(p2)), torch.ones_like(p2))
    Atil_row = torch.where(E_j, cache.cbar_j * p_new, A_sm[..., 0, :]).unsqueeze(-2)   # [B,H,1,n_k]; ONLY row t
    # ---- fan-in step 1 / 2 on row t (row-local; exactly Sec. 5)
    one = torch.ones(B, H, 1, dtype=S32.dtype, device=S32.device)
    excess, sm_mass = row_masses(Atil_row, A_sm, E_row)                          # the same cap as the dense path
    a1, cap_theta, cap_binds = unit_cap(Atil_row, vis, excess, sm_mass)
    Ef = E_row.to(S32.dtype)
    cbar_i = (a1 * Ef).sum(-1)
    Rtil = (Atil_row * Ef).sum(-1)
    if norm.tau_i_pinned:
        tau_i = one
    else:
        tau_i = norm.tauQ(_fp(q_t), norm.tauQ.summary(Atil_row, E_row, vis, cbar_i, Rtil, col_size=cache.rel_count, p=p_new.unsqueeze(-2)))
    u, theta = proj_le_masked(Atil_row.masked_fill(~E_row, 0.0), cbar_i / tau_i, E_row)
    A_row = torch.where(E_row, tau_i.unsqueeze(-1) * u, a1)
    cache.n_seen = t + 1
    if cache.policy == "seal_at_boundary" and cache.Q_hist is not None:
        cache.Q_hist = torch.cat([cache.Q_hist, q_t.detach().float()], -2)
        if cache.n_seen % cache.block_size == 0:
            reseal_at_boundary(norm, cache, K_kv, vis)
    info = dict(E=E_row, cbar_i=cbar_i, tau_i=tau_i, theta=theta, Rtil=Rtil, Atil=Atil_row, a1=a1, A_sm=A_sm,
                supp_rel=((A_row > 0) & E_row).sum(-1), cap_binds=cap_binds, cap_theta=cap_theta,
                excess=excess, sm_mass=sm_mass,
                kstar_new=(p_new > 0).sum(-1), eviction_events=cache.eviction_events,
                near_truncation_events=cache.near_truncation_events)
    from .normalizer import debug_enabled
    if debug_enabled():
        # the decode row is a single [B,H,1,n_k] slice, so the DENSE report runs on it directly and costs nothing.
        # MARSEA_DEBUG=1 was inert here: check_invariants was never called on this path at all (review 30372ae C).
        # INV-1 is skipped and recorded: cbar_j is the accumulated column, not a statement about row t alone.
        from .invariants import check_invariants
        dd = Diagnostics(E=E_row, Atil=Atil_row, a1=a1, A_sm=A_sm, A=A_row, cbar_i=cbar_i, tau_i=tau_i,
                         tau_j=cache.tau_j, nu=cache.nu, theta=theta, Rtil=Rtil, cap_binds=cap_binds, cap_theta=cap_theta)
        check_invariants(A_row, dd, vis, where=f"decode_step(t={t})")
    return A_row, info


@torch.no_grad()
def reseal_at_boundary(norm: MarSeaNormalizer, cache: FrozenPrefixCache, K_kv: torch.Tensor, vis_last: torch.Tensor):
    """E9 emission policy: at a block boundary re-read tau_j from the grown visible column (TauK on the
    prefix's column stats) and rebuild the relation lists; frozen again until the next boundary."""
    Q = cache.Q_hist                                                             # [B,H,T,d]
    B, H, T, d = Q.shape
    g_kv = H // K_kv.shape[1]
    k_rep = repeat_kv(_fp(K_kv), g_kv)                                           # [B,H,n_k,d]
    n_k = k_rep.shape[-2]
    S = torch.einsum("bhid,bhjd->bhij", Q, k_rep) * (d ** -0.5)
    i = torch.arange(T, device=Q.device)[:, None]; j = torch.arange(n_k, device=Q.device)[None, :]
    vis = (j <= i + n_k - T)[None, None].expand(B, H, T, n_k) & vis_last[..., :1, :].expand(B, H, T, n_k)
    S32 = S.masked_fill(~vis, float("-inf"))
    stats = column_stats(S32, vis)
    if norm.tau_j_global:
        tau = (norm.tau_min + torch.nn.functional.softplus(norm.tau_j_raw)).expand(B, H, n_k)
    else:
        tau = norm.tauK(k_rep, stats, cache.nu)
    cache.tau_j = tau
    logits = norm.relation(K_kv, Q, key_offset=0, n_k_total=n_k)
    E = (logits > 0) & vis
    A_sm = row_softmax(S32, vis)
    cache.cbar_j = (A_sm * E.to(A_sm.dtype)).sum(-2)
    K = cache.K_ret
    Srel = S32.masked_fill(~E, NEG_PAD).transpose(-1, -2)
    k = min(K, T)
    top = torch.topk(Srel, k, dim=-1, largest=True, sorted=True)
    valid = torch.gather(E.transpose(-1, -2), -1, top.indices)
    cache.rel_scores = torch.full((B, H, n_k, K), NEG_PAD, dtype=S32.dtype, device=S32.device)
    cache.rel_valid = torch.zeros((B, H, n_k, K), dtype=torch.bool, device=S32.device)
    cache.rel_qidx = torch.full((B, H, n_k, K), -1, dtype=torch.long, device=S32.device)
    cache.rel_scores[..., :k] = top.values.masked_fill(~valid, NEG_PAD)
    cache.rel_valid[..., :k] = valid
    cache.rel_qidx[..., :k] = top.indices.masked_fill(~valid, -1)


# --------------------------------------------------------------------------- T7 helper
@torch.no_grad()
def prefix_psi_trace(z_col: torch.Tensor, keep: torch.Tensor, tau: float):
    """psi_j^(t) and supports of sparsemax(tau * z[:t] on keep[:t]) as the prefix grows (Prop. prefix), for T7."""
    T = z_col.shape[-1]
    psis, supports = [], []
    for t in range(1, T + 1):
        m = keep[:t].clone()
        if not m.any():
            psis.append(None); supports.append(set()); continue
        zt = (tau * z_col[:t])[None]
        kstar, psi, _, _, _ = sorted_prefix_stat(zt, m[None])
        p = sparsemax_masked(zt, m[None])[0]
        psis.append(float(psi)); supports.append(set(torch.nonzero(p > 0).flatten().tolist()))
    return psis, supports
