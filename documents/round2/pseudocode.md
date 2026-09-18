# MarSea — implementation-ready pseudo-code (v4 read, 2026-09-05)

Sources are cited inline as `[spec §x]`, `[paper Lnnn / label]`, `[record §x]`, `[harness Lnnn]`,
`[ref::fn]` (= `reference/marsea_cold.py`). Where the four sources disagree the choice made here is
listed in `contradictions.md` under the number given as `[C-n]`.

Conventions used throughout:

```
B, H, H_kv, g = H // H_kv, n_q, n_k, d = head_dim, r = relation rank
S      [B,H,n_q,n_k]  fp32 scores after 1/sqrt(d) scaling, -inf off vis            [spec §2]
vis    [B,1,n_q,n_k]  bool; broadcast to [B,H,n_q,n_k] inside every normaliser       [spec §2, §6.1]
E, g   [B,H,n_q,n_k]  bool relation / straight-through gate float                     [spec §4]
*_j    [B,H,n_k]      per-key;   *_i [B,H,n_q] per-query                               [spec §2]
```
All programs run in fp32 (fp64 allowed for gradcheck) `[spec §6.3, H1]`. Every tensor listed in
§0 of `[spec §2]` is materialised in the dense reference path; the chunked path (§5.6) is the only
place they are not.

---

## 1. Primitives  `[spec §3; harness L14-33, L588-595; paper lem:prefix, eq:affine]`

### 1.1 `sorted_prefix_stat(z, mask)` — as `[ref::sorted_prefix_stat]`, unchanged

```python
def sorted_prefix_stat(z, mask):
    # z    [..., L] fp32/fp64 (values at ~mask ignored);  mask [..., L] bool
    # returns kstar [...] int (0 if no valid entry), psi [...], G [..., L] (sorted order, +inf at pads),
    #         u [..., L] sorted values, mvalid [..., L] validity in sorted order
    zp = z.masked_fill(~mask, -1e30)                       # pad, NEVER -inf  [spec §3 batched form, H5]
    u, order = sort(zp, dim=-1, descending=True, stable=True)   # stable: H8
    mvalid = gather(mask, -1, order)
    uv = where(mvalid, u, 0)
    cs = cumsum(uv, -1)                                    # cs[r] = u[0]+...+u[r] over VALID entries
    k  = cumsum(mvalid.float(), -1)                        # prefix COUNT of valid entries (1-based)
    G  = cs - k * u                                        # G[r] = sum_{t<=r}(u[t]-u[r])  [paper L687]
    G  = where(mvalid, G, +inf)
    cond  = (G < 1.0) & mvalid
    kstar = (k * cond).amax(-1)                            # max{k : G(k) < 1}  == 1 + k*u > cs  [harness L18]
    idx   = (kstar.long() - 1).clamp_min(0)[..., None]
    psi   = (gather(cs, -1, idx).squeeze(-1) - 1.0) / kstar.clamp_min(1.0)   # cs[kstar-1] (0-based)  [spec §3 v4]
    return kstar, psi, G, u, mvalid
```
Gradient: none needed (only consumed inside the custom Function below and in diagnostics).

### 1.2 `sparsemax_masked(z, mask)` — as `[ref::_SparsemaxMasked]`, unchanged

```python
class SparsemaxMasked(autograd.Function):
    forward(z, mask):  kstar, psi, ... = sorted_prefix_stat(z, mask)
                       p = clamp(z - psi[...,None], min=0); p = where(mask, p, 0)     # exact zeros, no eps (H4)
                       save supp = (p > 0)
    backward(grad_p):  s = supp.float(); n = s.sum(-1, keepdim) .clamp_min(1)
                       return s * (grad_p - (grad_p*s).sum(-1, keepdim)/n), None      # diag(1_S) - 1_S 1_S^T/|S|  [spec §3]
```
`entmax_masked(z, mask, alpha)`: `alpha == 2 -> sparsemax_masked`; else the bisection of
`[ref::entmax_bisect_masked]` unchanged (bounds `[(α-1)max z - 1, (α-1)max z]`, 50 iters, final
`p /= p.sum()`, exact zeros from the clamp, gradient = unrolled iteration) `[spec §3 v4; harness L22-33]`.
Used only by B4 at α = 1.5.

### 1.3 Projections  `[spec §3; harness L588-595]`

```python
def proj_eq_masked(v, s, mask):            # s [...] > 0 guaranteed by caller
    return s[...,None] * sparsemax_masked(v / s[...,None], mask)

def proj_le_masked(v, s, mask, tol=1e-9):  # argmin .5||a-v||^2 s.t. sum a <= s, a >= 0, on mask
    # as [ref::proj_le_masked] with ONE change: the zero-quota test is  s > 1e-9, not s > 0   [spec §3 v4, H6, H11]
    w      = clamp(v, min=0) * mask
    sw     = w.sum(-1)
    pos    = s > tol
    s_safe = where(pos, s, 1.0)                         # H11: safe divisor BEFORE the where
    pe     = proj_eq_masked(v, s_safe, mask)
    binds  = sw > s
    a      = where(binds[...,None], pe, w)
    a      = where(pos[...,None], a, 0)                 # proj_le(v, s<=1e-9) := 0
    # implied threshold theta (log only, no grad):  0 if slack, else max over supp(a) of (v - a)
    with no_grad: on = (a > 0) & mask
                  th = where(on, v - a, -inf).amax(-1); th = where(on.any(-1) & binds, th, 0)
    return a, th
```
`theta` is a *derived* dual, never a parameter or leaf `[spec H7; paper L431]`.

### 1.4 Masked-score discipline `[spec §3 v4, H10, H13]`
Every consumer of `S` other than the row-softmax first does `S.masked_fill(~vis, 0)` (or
`masked_fill(~E, 0)`) and then multiplies/gathers. Never `S * mask`.

---

## 2. Relation head  `[spec §4; paper L491, L1721-1731 (app:selective); ref::RelationHead]`

```python
class RelationHead(nn.Module):                       # ONE per patched layer, shared across heads  [spec §4.1]
    U = Linear(d, r, bias=False); V = Linear(d, r, bias=False); b0 = Parameter(0.)   # r = 16 default
    init: U.weight, V.weight ~ N(0, var = 1/d)  (std 1/sqrt(d))                        [spec §4.1 v4]

    def raw(K_kv, Q):                                # K_kv [B,H_kv,n_k,d] POST-RoPE key; Q [B,H,n_q,d]
        u = repeat_kv(U(K_kv.float()), g)            # [B,H,n_k,r]   (HF repeat_kv order: each KV head g times, consecutive)
        v = V(Q.float())                             # [B,H,n_q,r]
        return einsum('bhir,bhjr->bhij', v, u) / sqrt(r)          # [B,H,n_q,n_k]
    def forward(K_kv, Q):  return raw(K_kv, Q) + b0
    @no_grad
    def calibrate_b0(K_kv, Q, vis, rho0=0.05):       # [spec §4.2]: mean(E & vis)/mean(vis) == rho0 on this batch
        vals = raw(K_kv, Q)[vis.expand(...)]         # all visible pairs, all heads, whole batch
        b0.fill_(-quantile(vals, 1 - rho0))          # sorts once; use torch.quantile on a subsample if > 2^24 entries

E = (logits > 0) & vis                                # [B,H,n_q,n_k] bool  — the ONE relation object (INV-5)

def straight_through(E, logits, vis, T_st=1.0):       # [spec §4.3 v4; ref::straight_through] unchanged
    soft = sigmoid(logits / T_st) * vis.float()       # no gradient off vis
    return E.float() + (soft - soft.detach())         # forward == E exactly; backward d/dlogit = sigmoid'
```
- `b0` and `1/sqrt(r)` are the paper's `e_ij = 1[<u,v> > 0]` with a constant coordinate appended
  `[spec §4.1 v4]` — see `[C-7]`.
- Calibration happens **once, at the first step of Phase B, on the Phase-A weights**, per patched
  layer, on one warm-up batch, then `b0` trains `[spec §4.2 v4, §9]`.
- Gradient reaches `U, V, b0` only through `g` at the four interpolation sites of §4 below. The
  estimator is biased (membership is not differentiated) — accepted `[spec §4.3]`; the forward pass
  must stay hard `[spec §15]`. E9 alternative: hard-concrete gate with temperature annealed
  1.0 → 0.2, same four sites.

---

## 3. Exclusivity heads, field statistics, normalisation, init/calibration  `[spec §5.2; paper eq:tauhead L362-368, L376-387]`

### 3.1 Field statistics

```python
def column_stats(S, vis):        # as [ref::column_stats], unchanged (already v4-conformant)
    # S [B,H,n_q,n_k] (-inf off vis) -> [B,H,n_k,6] = (max, 2nd max, mean, BIASED std, max-2nd, (max-mean)/std)
    # over visible i.  Guards [spec §5.2 v4, H12]:  n_vis_j == 1 -> 2nd := max (gap 0), std = 0, ratio uses
    # std.clamp_min(1e-6);  n_vis_j == 0 -> all six := 0.   Computed on S.masked_fill(~vis, 0) and on
    # S.masked_fill(~vis, -inf) for the maxes (topk(2, dim=-2)); H11 safe-denominators.

def row_summary(Atil, E, vis, cbar_i, Rtil):   # as [ref::row_summary], unchanged  -> [B,H,n_q,4]
    # (Rtil_i, cbar_i, max over VISIBLE j of Atil_ij, Shannon entropy of Atil[i,E_i.]/Rtil_i; 0 when Rtil_i == 0)

f(x) = sign(x) * log1p(|x|)      # applied to every SCALAR feature entering a head            [spec §5.2 v4]
```

### 3.2 Heads

```python
class TauK(nn.Module):           # ONE per patched layer, shared across heads               [spec §5.2, §6.2]
    ln  = LayerNorm(d)           # [v4] on the key vector; post-RoPE keys have norms in the tens
    fc1 = Linear(d + 6 + 1, 64); fc2 = Linear(64, 1);  act = GELU;  tau_min = 0.05
    init: fc2.weight ~ N(0, 0.01^2);  fc2.bias = softplus^-1(1.0 - tau_min)  (default before calibration)
    zero_field_inputs: bool      # B3 sets True: stats and log nu zeroed AT THE INPUT, byte-identical otherwise
    def forward(k_rep, stats, nu_prev):              # k_rep [B,H,n_k,d] = repeat_kv(K_kv); stats [B,H,n_k,6]; nu_prev [B,H,n_k]
        x_stats = f(stats); x_nu = f(log(nu_prev.clamp_min(1e-30)))[...,None]
        if zero_field_inputs: x_stats, x_nu = 0, 0
        x = cat([ln(k_rep.float()), x_stats, x_nu], -1)
        return tau_min + softplus(fc2(gelu(fc1(x))).squeeze(-1))          # [B,H,n_k] > 0
    @no_grad
    def calibrate(S, vis):                           # [spec §5.2 v4]: "tau_j ~= 1/std" as a MEDIAN over columns
        std = column_stats(S, vis)[..., 3][vis.sum(-2) > 1]
        target = median(1 / std.clamp_min(1e-6))
        fc2.bias.fill_(softplus^-1(max(target - tau_min, 1e-3)))

class TauQ(nn.Module):
    ln = LayerNorm(d); fc1 = Linear(d + 4, 64); fc2 = Linear(64, 1); GELU; tau_min = 0.05
    init: fc2.weight ~ N(0, 0.01^2);  fc2.bias = softplus^-1(1.0 - tau_min)   # tau_i = 1 at init: step 2 == identity on a1 (see C-5)
    def forward(q, summary):  x = cat([ln(q.float()), f(summary)], -1); return tau_min + softplus(fc2(gelu(fc1(x))).squeeze(-1))
```
Both differ from `[ref::TauK/TauQ]` only by the `LayerNorm` on the vector input and `f(.)` on the
scalar features `[spec §5.2 v4; spec §16]`. `TauK.calibrate` is called once per patched layer at the
first Phase-B step on the same warm-up batch as `calibrate_b0` `[spec §5.2 v4]`. `nu_prev` for the first
patched layer is `1.0` (singleton support) `[spec §5.3; paper L1511-1513]`; otherwise the previous
patched layer's `nu` (NOT detached: `[spec §5 L315]` writes `state.nu_next = nu` and the paper L381-382 chooses `nu` over `|supp|` because it carries a gradient; the reference detaches — `[C-9]`).

---

## 4. `marsea_normalize` — dense reference  `[spec §5; paper eq:step1, eq:program, eq:rowstep1, eq:rowprog; harness L597-639 (executable reference); ref::MarSeaNormalizer.normalize]`

As `[ref::MarSeaNormalizer.normalize]` with the changes marked `# CHG`. Order of operations is
fixed by `[spec §15]` and must not be altered.

```python
def marsea_normalize(S, vis, K_kv, Q, state, *, logits_override=None, tau_j_override=None, tau_i_override=None):
    out_dtype = S.dtype; vis = broadcast(vis.bool(), S.shape)            # [B,H,n_q,n_k]
    S32 = S.float().masked_fill(~vis, -inf)
    with autocast(enabled=False):                                        # [spec §6.3]
        # ---- row softmax (the ONLY normalisation, ever)  [record §38.1, §41; paper L289-300]
        A_sm  = softmax(S32, -1); A_sm = where(vis.any(-1, keepdim), A_sm, 0)   # fully-masked rows -> 0  [spec §2 v4]
        c     = A_sm.sum(-2)                                             # [B,H,n_k]; INV-2 derived, never imposed
        # ---- relation
        logits = relation(K_kv, Q) if logits_override is None else logits_override
        E = (logits > 0) & vis;  g = straight_through(E, logits, vis)
        # ---- STEP 1 fan-out: inherited quota READS A_sm, never S              [eq:step1; record §38]
        cbar_j = (g * A_sm).sum(-2)                                      # [B,H,n_k]   ST site (1)
        # ---- STEP 2 fan-out: shape + exact zeros on the relation only         [eq:program]
        stats   = column_stats(S32, vis)
        nu_prev = state.nu_prev if state.nu_prev is not None else ones(B,H,n_k)
        tau_j   = TauK(repeat_kv(K_kv, g), stats, nu_prev) if tau_j_override is None else tau_j_override   # [B,H,n_k]
        zcol = (S32.masked_fill(~E, 0) * tau_j[...,None,:]).transpose(-1,-2)   # [B,H,n_k,n_q]; H13
        pT   = entmax_masked(zcol, E.transpose(-1,-2), alpha=2)          # p_ij = 0 off E_.j; p sums to 1 on non-empty E_.j
        p    = pT.transpose(-1,-2)
        with no_grad: kstar, psi_j, _, _, _ = sorted_prefix_stat(zcol, E.transpose(-1,-2))   # diagnostics
        Atil = A_sm + g * (cbar_j[...,None,:] * p - A_sm)                # ST site (2); forward == eq:program exactly
        kstar_c = (p > 0).sum(-2)
        nu = where(E.sum(-2) > 0, 1 / (p*p).sum(-2).clamp_min(1e-12), 1.0)   # CHG(v4): nu := 1 on empty column, in [1,|E_.j|]
        state.nu_next = nu                                               # CHG vs reference: no detach [C-9]
        # ---- STEP 1 fan-in: cap the WHOLE row at one unit; NOT a softmax          [eq:rowstep1; record §38 item 3]
        a1, _  = proj_le_masked(Atil, ones(B,H,n_q), vis)                # identity on rows summing <= 1
        cbar_i = (g * a1).sum(-1)                                        # ST site (3); POST-cap quota
        Rtil   = (E * Atil).sum(-1)
        cap_binds = Atil.sum(-1) > 1 + 1e-6                              # CHG(v4): tolerance
        # ---- STEP 2 fan-in: ceiling at the quota; TARGET is Atil (PRE-cap)      [eq:rowprog; INV-14; harness L625]
        tau_i = TauQ(Q, row_summary(Atil, E, vis, cbar_i, Rtil)) if tau_i_override is None else tau_i_override
        u, theta = proj_le_masked(Atil.masked_fill(~E, 0), cbar_i / tau_i, E)   # <=, never proj_eq; u = 0 off E; quota<=1e-9 -> u = 0
        A = a1 + g * (tau_i[...,None] * u - a1)                          # ST site (4)
    diag = Diagnostics(E, logits, cbar_j, cbar_i, tau_j, tau_i, psi_j, kstar=kstar_c, nu, theta, Rtil,
                       supp_rel=((A > 0) & E).sum(-1), cap_binds, p, a1, Atil, A_sm, c, u)
    return A.to(out_dtype), diag
```
Gradient notes: `tau_j` enters only through `zcol` (exact sparsemax Jacobian); `tau_i` through the
quota `cbar_i/tau_i` and the gain `tau_i * u` (exact through `proj_le`'s branch — the `where` is
piecewise); `logits` only through `g` at sites (1)–(4). Under `MARSEA_DEBUG=1` assert INV-1..14
`[spec §2]` with the stated tolerances after every call.

Direction check: `a / tau_i` inside the norm (quota `cbar_i/tau_i`, gain `tau_i`) — `[spec §5.1; record §26.2, §35.1]`.

---

## 5. Backbone patch  `[spec §1, §6; record §39.1]`

### 5.1 Load, detect, swap

```python
model = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=bfloat16, attn_implementation="sdpa")
assert config.num_attention_heads == 12 and config.num_key_value_heads == 2 and head_dim == 128   # Qwen2.5-1.5B [spec §1.3]
assert_signature(transformers.models.qwen2.modeling_qwen2.eager_attention_forward,
                 ["module","query","key","value","attention_mask","scaling","dropout"])            # pin version [spec §1.3]

for l in PATCHED_LAYERS:                              # from the detector (§5.5); provisional set for T0
    old = model.model.layers[l].self_attn
    new = MarSeaAttention(old, layer_idx=l, normalizer=arm.make_normalizer(d=head_dim))
    model.model.layers[l].self_attn = new             # projections are the SAME nn.Linear objects (no copy)  [C-22]
model.marsea_ctx = MarSeaContext()                    # per-forward state: nu handoff, diagnostics sink, padding mask
```
`MarSeaAttention` **wraps** the original module (holds `q_proj,k_proj,v_proj,o_proj` by reference,
same `num_key_value_groups`, `scaling`, `layer_idx`) so that peft's LoRA later targets them by their
unchanged names. Unpatched layers keep SDPA; never set `config._attn_implementation="eager"` `[spec §6.1]`.

### 5.2 Forward (per patched layer) — `Qwen2Attention.forward` with the attention call replaced

```python
def MarSeaAttention.forward(hidden_states, position_embeddings, attention_mask, past_key_value=None, cache_position=None, **kw):
    B, n_q, _ = hidden_states.shape
    q = q_proj(h).view(B, n_q, H, d).transpose(1,2); k = k_proj(h).view(B, n_q, H_kv, d).transpose(1,2); v = ...
    cos, sin = position_embeddings; q, k = apply_rotary_pos_emb(q, k, cos, sin)          # keys are POST-RoPE from here
    if past_key_value is not None: k, v = past_key_value.update(k, v, layer_idx, {"sin":sin,"cos":cos,"cache_position":cache_position})
    n_k = k.shape[-2]
    vis = build_vis(model.marsea_ctx.pad_mask, n_q, n_k, cache_position)                 # §5.3  [B,1,n_q,n_k] bool
    if attention_mask is not None: assert_consistent(attention_mask, vis)                 # bool or additive 4-D; see C-6
    A, diag = marsea_eager_forward(self, q, k, v, vis, scaling, kw)                       # below
    ...
def marsea_eager_forward(module, q, k, v, vis, scaling, kw):
    k_rep = repeat_kv(k, g); v_rep = repeat_kv(v, g)
    S = (q @ k_rep.transpose(2,3)) * scaling                                              # bf16 under autocast
    S = S.masked_fill(~vis, -inf)                                                         # replaces the additive mask; identical softmax  [C-26]
    state = ctx.state_for(layer_idx)                                                      # nu_prev from previous patched layer
    A, diag = normalizer.normalize(S, vis, K_kv=k, Q=q, state=state)                      # <-- THE LINE  [spec §1.2]
    ctx.push(layer_idx, state.nu_next, diag if ctx.collect else None)
    A = dropout(A, p=attention_dropout, training)                                         # backbone's (0.0 for Qwen2.5)
    out = (A.to(q.dtype) @ v_rep).transpose(1,2).contiguous()                             # value-mix in bf16 [spec §6.3]
    return o_proj(out.reshape(B, n_q, -1)), A
```
Gradient checkpointing: wrap `marsea_eager_forward` in `torch.utils.checkpoint` on patched layers
(the normaliser keeps ~4 `[B,H,T,T]` fp32 tensors live) `[spec §6.5, §9]`.

### 5.3 `vis` from the padding mask (not from the 4-D additive mask)

```python
def build_vis(pad_mask, n_q, n_k, cache_position):       # pad_mask [B, n_k] bool (1 = real token)  — HF's 2-D attention_mask
    i = arange(n_q)[:,None]; j = arange(n_k)[None,:]; off = n_k - n_q          # KV offset [spec §2]
    causal = (j <= i + off)                                                    # [n_q,n_k]
    vis = causal[None,None] & pad_mask[:,None,None,:]                          # keys that are pad are invisible
    q_real = pad_mask[:, -n_q:] if cache is None else ones                     # right padding: pad QUERY rows fully invisible [spec §6.1 v4]
    vis = vis & q_real[:,None,:,None]
    assert (vis.any(-1) | ~q_real[:,None,:]).all()                             # every REAL row sees >= 1 key (no left-pad real rows)
    return vis
```
Rationale: transformers ≥ 4.53 hands SDPA layers a *boolean* mask or `None`; earlier versions hand
an additive fp mask with `finfo.min`; the spec's `mask > -1e30` derivation is version-fragile
`[C-6]`. Building `vis` from the 2-D mask + causal offset is exact for right padding and for
KV-cache decoding, and it makes the padding-length-invariance check `[paper L2672-2674]` trivial.

### 5.4 GQA, dtype, state
- Everything per Q-head; `U_phi`, `TauK` read `K_kv` (KV-head) and `repeat_kv` afterwards `[spec §6.2]`.
- `S` bf16 → fp32 inside; `A` cast back; value-mix bf16 `[spec §6.3]`.
- `MarSeaContext`: set by a forward pre-hook on the model (`pad_mask`, `collect` flag, clears the
  `nu` chain); patched layers are visited in index order so "previous patched layer" is the last
  pushed `nu` `[spec §5.3]`. Do not rely on `**kwargs` plumbing (filtered by typed kwargs in newer
  transformers) `[C-23]`.

### 5.5 Which layers (`PATCHED_LAYERS`, `(l*, h*)`)  `[spec §6.7; paper L881-884]`
```
detector: 200 RULER S-NIAH prompts (4K, type_needle_v=words, seed 0), teacher-forced on the UNPATCHED
  backbone with attn_implementation="eager" and output_attentions=True;
  score(l,h) = mean over prompts of  fraction of gold-value answer positions whose row-argmax (over the
  whole visible row) lands on the needle's copy of the token being emitted;  head retrieves if score > 0.1
PATCHED_LAYERS = the L_patch = 4 distinct layers of the top-scoring heads (rank heads by score, take layers in
  order of first appearance until 4);  (l*, h*) = the top head of the top patched layer.  Record scores per (l,h).
```
T0 is invariant to the choice; run T0 with a provisional set before the detector `[spec §14 step 4]`.

### 5.6 Exact chunked implementation  `[spec §6.6]` (required at 16K only; S2 runs at 8K dense)
```
pass 0  lse_i = logsumexp_j S_ij  (flash-style, chunked over keys, O(T) memory)
pass 1  for key-chunk J (1024): S[:,J] -> A_sm[:,J] = exp(S[:,J]-lse); logits[:,J], E[:,J], g[:,J];
        cbar_J = sum_i g A_sm;  column_stats on the chunk (needs only S[:,J]); tau_J; p on the relation;
        store SPARSE (i,j,Atil_ij, g_ij) for E entries;  rowsum_i += sum_J Atil[i,J]  (= 1 + sum_{E} (Atil - A_sm))
pass 2  rows with rowsum_i <= 1+1e-6: a1 = Atil (identity); fan-in step 2 on the row's sparse entries only.
        rows with rowsum_i > 1: rebuild the dense row from S[i,:], lse_i and its sparse entries; proj_le exactly;
        then fan-in step 2 on the sparse entries with quota from the dense a1.  Log the fraction of such rows.
output  O = (A_sm V) via the flash kernel  +  sparse corrections sum_{(i,j) in E} (A_ij - A_sm_ij) V_j
        + for cap-binding rows, replace the row's output by the dense a1-based product.
```
Backward: `torch.utils.checkpoint` per chunk. NOTE the ST gradient to `logits` is DENSE on visible pairs, not
only on relation entries: at site (2) an off-relation visible pair has `p_ij = 0`, so `dL/dlogit_ij = -A_sm_ij * sigmoid'(logit_ij) * dL/dAtil_ij`
(and likewise `-a1_ij * sigmoid'` at site (4)); the chunked backward must recompute `A_sm[:,J]` and `sigmoid'` per chunk for it.
**T11** compares every output tensor/diagnostic against §4 at T ≤ 4K, fp64 to 1e-9 then fp32 relative 1e-5.

### 5.7 Decoding (generation mode)  `[spec §6.4, §7; paper prop:prefix(iii); harness L307-326]`
Frozen-prefix (default): at prefill, run §4 once on the prompt; cache per patched layer and key `j`:
`tau_j`, `E[:,j]` restricted to relation entries, the relation score vector `S[E_.j, j]`, and `cbar_j`.
At decode step `t` (n_q = 1): compute `S[t,:]`, `A_sm[t,:]`, `E[t,:]` from the pairwise head (new query
vs cached keys); for each `j` with `E[t,j]`: `cbar_j += A_sm[t,j]`, append `S[t,j]` to the key's
relation vector, re-solve `p_.j` at the **frozen** `tau_j` and take only the new entry
`Atil[t,j] = cbar_j p_j[t]` (earlier entries are never revisited — frozen); off-relation `Atil[t,j] = A_sm[t,j]`;
then the fan-in program on row `t` exactly as §4. Hierarchical top-K (`K_ret = 64`, block 512, on
ORIGINAL scores) bounds the per-key re-solve `[spec §7]`. Seal-at-boundary (E9 alternative): `tau_j`,
`E[:,j]`, `cbar_j^(b)` recomputed only when a 512-block closes and held for the next block.
Correctness test: greedy decode with `E = ∅` token-identical to the unpatched model `[spec §6.4]`. See `[C-17]` and residual R-3.

---

## 6. Baselines as `Normalizer`s  `[spec §8; paper L926-935, L2676-2714]`

Interface: `A, diag = normalize(S, vis, K_kv, Q, state)`; all share §5's patch, layers, LoRA `[spec §8]`.

**B0 `SoftmaxNorm`** — as `[ref::SoftmaxNorm]`. `A = row_softmax(S)`; fully-masked rows → 0. T0 bit-level vs
the unpatched eager model (logits 1e-5 bf16, greedy tokens exact).

**B1 `SoftmaxOneNorm`** (Miller 2023) — as `[ref::SoftmaxOneNorm]`:
`A = exp(S32 - logsumexp(cat([S32_visible, 0], -1)))` on visible entries, 0 off `vis`; rows < 1, no zeros
`[spec §8 v4]`. The `+1` sits at score 0 *after* the `1/sqrt(d)` scaling.

**B2 `MESHNorm`** — rewrite of `[ref::MESHNorm]` (its v3 version differentiates through the descent and
uses `eps = 0.1·std`; both wrong for v4 `[spec §16]`). Pinned to Zhang et al. ICML 2023 §3 / App. A:

```python
class MESHNorm(nn.Module):
    h_a = Sequential(LayerNorm(d), Linear(d,64), GELU, Linear(64,1))    # reads the KV-head key
    h_b = Sequential(LayerNorm(d), Linear(d,64), GELU, Linear(64,1))    # reads the Q-head query
    eps = 1.0; lam = 1.0; T_mesh = 4; inner_iters = 5; outer_iters = 20   # eps in {0.5,1,2}, lam in {0.3,1,3} are E9

    def normalize(S, vis, K_kv, Q, state=None):
        vis = broadcast(vis, S.shape); S32 = S.float().masked_fill(~vis, -inf)
        key_vis = vis.any(-2)                       # [B,H,n_k]  "visible key" = seen by >= 1 real query
        row_vis = vis.any(-1)                       # [B,H,n_q]  real query rows
        m = row_vis.sum(-1).float()                 # [B,H]  n_real_q  (MarSea's INV-2 total)
        la = repeat_kv(h_a(K_kv.float()), g).squeeze(-1)          # [B,H,n_k]; a identical across a GQA group
        lb = h_b(Q.float()).squeeze(-1)                            # [B,H,n_q]
        log_a = log(m)[...,None] + log_softmax(la.masked_fill(~key_vis, -inf), -1)   # -inf on unseen keys
        log_b = log(m)[...,None] + log_softmax(lb.masked_fill(~row_vis, -inf), -1)   # -inf on pad rows
        C = (-S32).masked_fill(~vis, +inf)                         # cost; log-kernel -C/eps = -inf off vis

        # ---- MESH step (Zhang et al. Eq. 11): minimise plan entropy by normalised gradient descent on the cost.
        #      NOT differentiated through: straight-through to C.                                  [spec §8 v4]
        with no_grad():
            Cp = C + randn_like(C).masked_fill(~vis, 0) * 1e-6     # symmetry-breaking noise; +inf stays +inf
            for t in range(T_mesh):
                with enable_grad():
                    Cv = Cp.detach().requires_grad_(True)
                    P  = sinkhorn_log(Cv, log_a.detach(), log_b.detach(), eps, iters=inner_iters, end_on="row")
                    Hp = -(P * log(P.clamp_min(1e-30))).sum((-1,-2))            # plan entropy per (b,h)
                    grad, = autograd.grad(Hp.sum(), Cv)
                grad = grad.masked_fill(~vis, 0)
                gn   = grad.flatten(-2).norm(dim=-1).clamp_min(1e-12)[...,None,None]   # ||grad||_2 per (b,h) slice
                Cp   = Cp - lam * grad / gn                                          # DESCENT: sharpens
        C_st = C + (Cp - C).detach().masked_fill(~vis, 0)          # forward = Cp; backward passes straight to C (i.e. to S)

        A = sinkhorn_log(C_st, log_a, log_b, eps, iters=outer_iters, end_on="row")   # unrolled; gradients into S, h_a, h_b
        col_resid = (A.sum(-2) - log_a.exp()).abs().masked_fill(~key_vis, 0).amax(-1)   # LOG per layer  [spec §8 v4; paper L2692-2695]
        return A.to(S.dtype), Diagnostics(extra={row_sum=A.sum(-1), col_resid=col_resid, a=log_a.exp(), b=log_b.exp()})

def sinkhorn_log(C, log_a, log_b, eps, iters, end_on="row"):      # as [ref::sinkhorn_log] with two changes
    # CHG: eps is a python float; CHG: ENDS ON THE ROW SCALING so rows carry b_i exactly:
    negC = -C / eps  (-inf off vis; dead rows/cols get a finite stand-in and masked duals as in the reference)
    f = 0; gd = 0
    for _ in range(iters):
        gd = la - logsumexp(negC_safe + f[...,:,None], dim=-2);  gd = gd.masked_fill(~key_ok, -inf)    # column step
        f  = lb - logsumexp(negC_safe + gd[...,None,:], dim=-1); f  = f.masked_fill(~row_ok, -inf)     # row step LAST
    return exp((negC + f[...,:,None] + gd[...,None,:]).masked_fill(dead, -inf))
```
Under a causal mask the two-marginal problem is infeasible; rows sum to `b_i` (not 1), the column
residual is logged, and the paper states the limitation `[spec §8 v4; paper L2691-2695]`. `h_a, h_b`
are created fresh at the first Phase-B step `[spec §9]`. T9 must show its column precision pinned.

**B3 `KeyOnlyTauNorm`** — `MarSeaNormalizer` with `TauK(zero_field_inputs=True)`; nothing else differs
(same relation, same `TauQ`) `[spec §8; paper L2678]`. As `[ref::KeyOnlyTauNorm]`.

**B4 `RowEntmaxNorm(alpha)`** — as `[ref::RowEntmaxNorm]`: `A = entmax_masked(S.masked_fill(~vis,0), vis, alpha)`,
α ∈ {1.5, 2} fixed per run, no temperature `[spec §8]`.

**B5 `MatchedSparsityNorm`** — two-pass, **evaluation-time only** `[spec §8 v4; paper L2680, L2705-2714]`:
```
pass 1  run the trained MarSea arm over the evaluation set (same examples, order, seed); record per
        (example, layer, head, row):  E[i,:] (bool over keys) and n_i = supp_rel_i = |supp(A[i, E_i.])|
pass 2  load the trained B0 arm (softmax-trained LoRA); on each patched layer run, per row i:
        A_sm = row_softmax(S);  keep = top-n_i entries of A_sm within E_i. (ties: stable, lower index first);
        A[E_i.] = A_sm * keep * (sum_{E_i.} A_sm / sum_keep A_sm);  A[~E_i.] = A_sm;  n_i = 0 -> A[E_i.] = 0
```
As `[ref::MatchedSparsityNorm]` (unchanged) with the trace loaded from pass 1 keyed by example id; the
normaliser asserts the trace's shapes match. Row sums to 1 (n_i > 0) or < 1 (n_i = 0). E9 optional:
oracle top-`m_j` within the relation.

---

## 7. Training loop  `[spec §9; paper L881-890, L2670-2671]`

```python
for seed in {0,1,2}:                                            # data order + init
  set_seed(seed)
  # ---------- Phase A: 500 steps, normaliser = SoftmaxNorm for EVERY arm (== plain LoRA fine-tune)
  # Phase A is arm-independent at fixed seed: run it ONCE per seed and fork the checkpoint to all 7 arms  [C-30]
  model = load_and_patch(NAME, PATCHED_LAYERS, normalizer=SoftmaxNorm())
  model = peft.get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                target_modules=["q_proj","k_proj","v_proj","o_proj"]))   # ALL 28 layers
  freeze everything else (embeddings, norms, MLPs);  gradient_checkpointing_enable() on patched layers
  opt = AdamW([{"params": lora_params, "lr": 2e-4}], betas=(0.9,0.95), weight_decay=0.01)
  sched = warmup(3% of 3500 = 105 steps) + cosine to 10% of base LR at step 3500   # ONE schedule over A+B
  for step in 0..499: train_step(batch=16 sequences via grad-accum of 16 x (1 seq/GPU), clip 1.0)
  save ckpt_A[seed]
  # ---------- Phase B: 3000 steps, arm's normaliser live
  for arm in {B0,B1,B2,B3,B4,MarSea}:                            # B5 has no training (eval-only, on B0's weights)
    model = load ckpt_A[seed]; swap normaliser := arm.normalizer (fresh modules: U,V,b0,TauK,TauQ | h_a,h_b)
    if arm in {MarSea, B3}:                                      # calibration on ONE warm-up batch, Phase-A weights  [spec §4.2, §5.2]
        with no_grad, collect=True: forward(warmup_batch)        # hooks capture (S, vis, K_kv, Q) per patched layer
        for l in PATCHED_LAYERS: relation[l].calibrate_b0(...,rho0=0.05); tauK[l].calibrate(S_l, vis_l)
        log coverage rho_0 per layer (must be ~0.05)
    opt.add_param_group({"params": arm_new_params, "lr": 1e-3}); scheduler continues (no restart)   # [C-24]
    for step in 500..3499: train_step(...); every 250 steps: fast eval on the held-out RULER grid + E8 block
    save ckpt_B[arm][seed]

def train_step(batch):
    with autocast(bf16): logits = model(input_ids, attention_mask=pad_mask)    # right-padded to L; pad_mask set on ctx
    loss = CE(logits[:, :-1], labels[:, 1:]) with labels = -100 on prompt and pad  (answer tokens only)
    (loss / 16).backward();  every 16 micro-steps: clip_grad_norm_(1.0); opt.step(); sched.step(); opt.zero_grad()
    if MARSEA_DEBUG: assert every parameter grad finite (T15)
```
Data `[spec §9, §13.1]`: RULER NIAH (E2/E3/E4 training grids, generated with the backbone tokenizer,
seeds distinct from the eval seeds) + MuSiQue-Ans train + HotpotQA-distractor train, mixed 1:1:1 by
examples; one example per sequence, right-padded to `L ∈ {4K, 8K, 16K}` by stage, **no packing**.
Logging every step from Phase-B step 0: coverage `rho` (both margins), histograms of `|E_.j|`, `|E_i.|`,
`var_j tau_j`, `var_i tau_i`, `k*` histogram, fraction of cap-binding rows, hierarchical-truncation flag
`[spec §12.4]`. Aux losses: none (E9 coverage-floor penalty off unless pre-registered) `[spec §9, §15]`.

---

## 8. Evaluation loop  `[spec §12; paper def:fidelity L760-774, L2716-2733, L2771-2842]`

### 8.1 Two modes, never mixed `[spec §12.1]`
- **Teacher-forced** (attention-level): input = prompt + gold answer, one forward with `collect=True`,
  read `A` and `Diagnostics` at the rows/columns of §8.2. No KV cache.
- **Generation** (task-level): greedy, `max_new_tokens` 32 (RULER) / 16 (QA) / task default (IHEval);
  parse; score. Frozen-prefix decoding (§5.7).

### 8.2 Which (layer, head, row, column)  `[spec §12.2, §6.7]`
```
(l*, h*)      top retrieval head among PATCHED_LAYERS (detector scores)
row i         query position i at (l*,h*).  "Answer positions" under teacher forcing = the positions whose
              next-token target is a gold token:  for RULER, the position emitting the FIRST token of each gold
              value (one row per value -> m rows);  for QA, every position emitting a gold answer token.   [C-29]
column j      key position j at (l*,h*)
passage/span  support of row i restricted to a span = union over the span's token keys:  selected iff
              exists j in span with E[i,j] & A[i,j] > 0  (ON the relation);  also log the dense version
              exists j in span with A[i,j] > 0 (all visible) for the dense arms' m/n identity.       [C-10]
Report per (l,h) for every patched (l,h) too; never pool heads.
```

### 8.3 Column ground truth `T_j` and the S0 routing check `[spec §12.3, §12.3a; record §36.3, §40]`
```
C-a  MV-NIAH value-key column:   j = first token of needle value v;  T_j = { row emitting the first token of v }   (m_j = 1)
C-b  question-key column:        j = last token of the queried key phrase IN THE QUESTION;  T_j = the m answer rows  [R-1]
C-c  QA supporting-passage col.: j = first token of gold passage;  T_j = answer rows
S0 (B0, unpatched):  usable iff  A_sm[T_j, j] / max_j' A_sm[T_j, j']  > 0.5  on >= 80% of examples;
      replication pair: two keys at (l*,h*) with planted |T_j| differing (M > m_j).  Gate for S1+.
m_j = 0 columns: excluded from every column average (count reported).                  [spec §12.3a]
```

### 8.4 Column quantities (per column with `T_j`, on the relation)  `[spec §12.4; paper thm:recovery, ass:margin]`
```
Erel   = E[:, j] at (l*,h*);  Trel = T_j ∩ Erel;  relation_recall = |Trel| / m_j   (logged separately, never charged to tau)
m      = |Trel|  (skip column if m == 0)
delta  = min_{i in Trel} S[i,j] - max_{i in Erel \ Trel} S[i,j]      (+inf if Erel \ Trel is empty)   [R-5]
W      = sum_{i in Trel} (S[i,j] - min_{Trel} S[.,j])
lo     = 1 / (W + m*delta)   (0 if delta = +inf);   hi = 1/W if W > 0 else +inf  (flag hi_inf; H2: exclude from upper-endpoint stats)
hit    = lo <= tau_j < hi        # SHARP lower endpoint, never 1/(m*delta)   [spec §12.4; paper L2785-2787]
dist   = signed distance to the nearer endpoint when not hit
S1_j   = supp(Atil[:, j]) ∩ Erel = supp(p[:,j]);  S2_j = supp(A[:, j]) ∩ Erel
P_j^(s) = |S_j^(s) ∩ Trel| / |S_j^(s)|  (1 on empty support);  R_j^(s) = |S_j^(s) ∩ Trel| / m;  rho_j = sum_{Trel} p
also: k*, nu_j, |Erel|, tau_j, psi_j, delta>0 flag, relative width m*delta/W
```
Dense arms (B0, B1, B2): no relation; score on the full visible column (`P_j = m_j/n_vis_j`, identity)
**and** on MarSea's recorded relation from the paired run (matched domain, `m_j/|E_.j|`). T9 checks both `[C-10]`.

### 8.5 Row quantities (per answer row `i`, on the relation)
```
Erow = E[i,:];  K_i = gold source keys (RULER: tokens of the gold needles; QA: gold passage tokens; passage-level via §8.2 union)
S_i^(1) = supp(Atil[i, Erow]);  S_i^(2) = supp(A[i, Erow]);  P_i, R_i as above;  relation_recall_i = |K_i ∩ Erow|/|K_i|
theta_i, tau_i, Rtil_i, cbar_i, tau_i*Rtil_i/cbar_i (regime statistic; see C-5), cap_binds_i, |Erow|, supp_rel_i
distractor three-way split over keys j in Erow \ K_i:                                        [spec §12.3a]
   excluded_by_stage1 : Atil[i,j] == 0 exactly     (also log argmax_{i' in E_.j} S[i',j] != i as the m_j=1 explanation)
   rejected_by_row    : Atil[i,j] > 0 and A[i,j] == 0
   surviving          : A[i,j] > 0
   (off-relation distractors are a 4th share: dense, reported for completeness)
```

### 8.6 Per (layer, head, step) and coverage/degeneracy diagnostics `[spec §4.4, §12.4; paper L2809-2827]`
coverage `rho_col = mean_j[|E_.j|>0]`, `rho_row = mean_i[|E_i.|>0]`, `rho = mean(E&vis)/mean(vis)`; full histograms
of `|E_.j|` and `|E_i.|` (not means); fraction of columns with `|E_.j| == 1` (degeneracy: theorem says nothing there);
`var_j tau_j`, `var_i tau_i`; `k*` histogram stratified by `|E_.j|`; fraction `k* >= 0.9 K_ret` (truncation flag);
fraction of rows over one unit after Stage 1; row-trigger rate ≥ column-induced rate (a violation is a bug).
A run whose `rho → 0` is reported as B0; a run whose `|E_.j|` mass sits at 1 has left the theorem `[spec §4.4]`.

### 8.7 Task metrics (generation mode) `[spec §13.1; paper L2741-2766]`
RULER: exact-set accuracy (split generated text on ", ", strip, compare as a set to `outputs`) beside
RULER's own recall score. MuSiQue: answer F1 + support F1. HotpotQA: answer EM/F1, supporting-fact
EM/F1, joint EM/P/R/F1. IHEval: task accuracy by conflicting-tier count, reported as a slope. E3 points
twice (real distractors vs inert padding) with the depth profile `{0.1,...,0.9}` at one `(m,n)`.

---

## 9. Data pipeline  `[spec §13.1; RULER niah.py (fetched, see contradictions C-1..C-3)]`

### 9.1 RULER generation
```
python scripts/data/synthetic/niah.py --tokenizer_type hf --tokenizer_path Qwen/Qwen2.5-1.5B --template "<niah template>"
   --max_seq_length L --tokens_to_generate 128 --num_samples N --random_seed s
   --type_haystack essay --type_needle_k words --type_needle_v numbers --num_needle_k K --num_needle_v V --num_needle_q 1
S-NIAH (E4):   K=1,  V=1                                  (niah_single_2)
MV/E2 m-sweep: K=32, V=m in {1,2,4,8,16}                  (32 keys each carrying m values; 1 queried -> m gold, 31m distractor values)   [C-1]
MK/E3 n-sweep: K=n in {8,16,32,64,128}, V=4               (m = 4 fixed)
inert padding: K=1, V=4 at the SAME L  (essay fills the length; distractors gone)
depth control: PATCH niah.py — stock RULER samples each needle's depth at random (random.sample(DEPTHS,...), L157);
               add --gold_depth d in {0.1,0.3,0.5,0.7,0.9} placing the queried key's V needles at depth d
               and the distractor needles at random depths.                                             [C-2]
jsonl fields: index, input (answer_prefix STRIPPED), outputs (list), length, answer_prefix, token_position_answer.
prompt = input + answer_prefix ; gold = " " + ", ".join(outputs) (RULER's own template; no chat template) [C-3]
```
Gold positions (teacher forcing): locate each needle sentence `"for {key} is: {value}."` by exact string
search in `input` (value-only search can collide inside the essay), map char offsets → token offsets with
the tokenizer's `offset_mapping`; the answer rows are the positions in the appended gold string whose
next token is the first token of each value. Train/eval seeds disjoint; eval grid held out.

### 9.2 MuSiQue / HotpotQA / IHEval
```
MuSiQue  musique_ans_v1.0_{train,dev}.jsonl: question, answer, paragraphs[{title, paragraph_text, is_supporting}],
         question_decomposition[{question, answer}] (bridge strings).  m = #is_supporting (2-4).
         E3-style n-sweep: sample n - m distractor paragraphs from OTHER questions' non-supporting paragraphs (seeded).
HotpotQA HF "hotpot_qa"/"distractor": question, answer, context{title, sentences}, supporting_facts{title, sent_id}. 10 paras, 2 gold.
IHEval   github ytyz1307zzh/IHEval, multi-turn rule-following split; tier count from metadata.
Prompt   "{context}\n\nQuestion: {question}\nAnswer:"  ->  loss/decode on " {answer}";   context = passages
         "Title: {t}\n{text}" joined by "\n\n"; gold passages in hop order, distractors at seeded fixed slots.
Padding  right-pad to L with pad_token (Qwen: <|endoftext|>); labels -100 on prompt+pad; pad_mask -> ctx.
Gold spans (T_j / K_i): character spans of gold passages / gold sentences mapped by offset_mapping.
```

---

## 10. Acceptance tests T0–T15 → harness blocks  `[spec §11; verify_all.py]`

| test | what | harness block (line) | tensor-side procedure |
|---|---|---|---|
| T0 | B0 == unpatched eager; MarSea `E=∅` likewise | — | 2K prompts, `attn_implementation="eager"` twin; logits `allclose(1e-5)` bf16; greedy 64 tokens identical; `logits_override=-inf` for MarSea |
| T1 | lem:prefix | L52-70 | `[ref::test_cold]` T1 block; also G monotone on padded batches |
| T2 | thm:recovery both endpoints | L92-123, L376-399 (scope) | planted columns (`targets_scores`) through `marsea_normalize` with `tau_j_override`; check `supp(p)==T` iff `lo<=tau<hi` |
| T3 | prop:twostep INV-1..4, 12 | L597-639 | `[ref::test_cold]` INV loop (400 random cases incl. finfo.min, offset-causal vis, per-batch padding) |
| T4 | prop:scopemono | L663-688 | shrink `E` by `logits_override`; recompute delta/interval from `diag`; containment |
| T5 | cor:capacity | L690-707 | `|E_.j|` 4..256 at `m_j=4`, oracle tau: `kstar==4`, `P_j==1` |
| T6 | prop:stagecomp INV-6/7 | L631, L636 (Prop. A(v)) — NOT L278-305 (hand floor) | on the real eq:rowprog path: `A[Atil==0]==0`, `supp(A[:,j]) ⊆ supp(Atil[:,j])` |
| T7 | prop:prefix INV-9 | L307-326, L437-454 | growing `vis` prefix with `tau_j_override` FROZEN; live head expected to fail (log, don't assert) |
| T8 | prop:tournament | L163-188, L205-221 | `hierarchical_topk == flat` whenever `K_ret >= k*` |
| T9 | fullsupportceiling | L223-236, L484-492 | B0/B1/B2 outputs: `P_j == m_j/|domain|` exactly, both domains (§8.4), zero seed variance |
| T10 | fig:prcurve L-shape | L526-577 | sweep `tau_j_override` on planted columns; no interior point |
| T11 | chunked == dense | — | random `E`, tau; T ≤ 4K; fp64 1e-9 then fp32 rel 1e-5 on every tensor of `Diagnostics` |
| T12 | INV-13 | L648-661 | `tau_j_override=1e-9`: `Atil[E_.j,j] == cbar_j/|E_.j|` (1e-6) and `!= A_sm` |
| T13 | gradients | — | `gradcheck(SparsemaxMasked)` fp64; FD of loss wrt `tau_j, tau_i` (override tensors) and wrt `logits` through the ST gate on a 4x6 toy (`[ref::test_cold]` L176-223) |
| T14 | INV-14 detector | L625 (target `At`, quota `ci`) | on cap-binding rows compare `A[i,E]` to `tau_i*proj_le(Atil[i,E], cbar_i/tau_i)` (equal) and to the `a1`-target version (must differ on some row) |
| T15 | NaN sweep | — | causal `n_q=n_k`, left-padded batch (pad rows fully invisible), `|E_.j|=0` columns, `cbar_i=0` rows: loss and every grad finite |
| +T16 | padding-length invariance `[paper L2672-2674]` | — | same example right-padded to two lengths: real-row `A`, `diag`, logits identical (bf16 tolerance) |

The 12.3a distractor split and `m_j=1` behaviour map to harness L709-744; run the split on the tensor path
too (assert `excluded_by_stage1` matches the argmax rule at `tau_j >= 1/delta_j`).

---

## 11. Build order and gates `[spec §14]`
1 primitives + T1/T2/T13 → 2 dense normaliser at `E=vis` + T3/T6/T9/T10/T12 → 3 relation + T4/T5 + coverage
histogram → 4 backbone swap + T0 (provisional layers) + B0 greedy twin + T15/T16 → 5 chunked + T11 + memory
profile 8K/16K → 6 detector, freeze `(l*,h*)`/`PATCHED_LAYERS` → 7 causal sealing + hierarchical top-K + T7/T8 +
frozen-prefix decode → 8 B0, B1, B4, then B3, B2 (verify vs MESH App. A), B5 last → 9 instrumentation +
RULER generation (+ the depth patch) + S0 → 10 Phase A/B at 4K one seed, E8, then the S2 grid at 8K.
Decisions reserved to Luke: `[spec §15]`.
