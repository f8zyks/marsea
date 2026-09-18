# MarSea — implementation contract (v4.12, 2026-09-15)

**Audience: Claude Code, generating executable code from this document plus `marsea_iclr.tex`
(Secs. 2, 3, 5; Apps. E, G, H) and `verify_all.py`.**

## 0. How to use this document

The paper is **normative** on *what* is computed and *why*. This document is normative on
*shapes, order of operations, edge cases, the backbone, the training recipe and the experiment
harness*. Where they disagree the paper wins — except at points marked ⚠️ **ERRATUM**, which
record places the paper's prose lags the mechanism and state what to implement.

Read, in this order: paper §2.1 (`eq:step1`, `eq:program`), §2.2 (`eq:rowstep1`, `eq:rowprog`),
§2.3–2.5 (relation, scaling, causal), §3 (`thm:separation`, `cor:capacity`, `thm:recovery`,
`prop:twostep`, `prop:scopemono`, `def:fidelity`), §5 (E1–E9), App. E (selective exclusivity),
App. G (`verify_all.py`'s protocol — its blocks are the acceptance tests of §11), App. H (protocol).

**What changed in v3 (sixth pass).** (1) §1 names the backbone and the patch point. (2) §4 fixes
the relation's initialisation and gradient path, which v2 left to a comment. (3) §6 gives the
backbone integration: module swap, GQA, dtype, memory arithmetic, and an exact chunked
implementation with its equality test. (4) §9 is the training recipe, matched across arms.
(5) §12 gives the column ground-truth constructions (audit-5 C2) and the routing check that
licenses them. (6) ⚠️ **ERRATUM retired**: the paper previously said `tau_j -> 0` "returns
standard attention". It does not — on a non-empty relation it is the *uniform* re-shaping
(`cbar_j/|E_.j|` on every relation entry). Only `E = ∅` is standard attention. Both drafts now say
so; the harness checks it (`verify_all.py`, "App. E(e)" rows); the initialisation in §4.2 follows.

**What changed in v4 (cold-read pass).** A second engineer implemented v3 from the text alone
(`reference/marsea_cold.py`, §16) and logged 31 points where the text let a literal reader write
wrong or NaN-producing code. Every one is resolved below and marked **[v4]**. The ones that would
have broken a real run: `proj_le` at a zero quota (the *majority* row under random scores) NaNs
the backward in a batched `where`; `column_stats` divides by a zero std on the last key of every
causal block; `nu` on an empty column was `1/eps`; B2 (MESH) was not specified enough to write and
its "rows sum to `b_i`" claim is false under a causal mask; `psi_j` monotonicity holds only at sealed `tau_j`;
fan-in step 2's target (`Atil`, pre-cap) vs its quota (`a1`, post-cap) was easy to mis-read and no
invariant catches the mis-reading; §13 named no data sources, formats or grid values.

**What changed in v4.1 (round 2: pseudo-code from ALL sources).** A second engineer, this time with
the paper, the master record, the harness and the reference implementation, wrote the full
pseudo-code (`round2/pseudocode.md`, in the package) and a 30-item cross-document contradiction
list (`round2/contradictions.md`). Resolved here, marked **[v4.1]**: `tau_i = 1` IS the identity
on step 1's row (audit-5's "not an identity" was wrong; paper fixed); RULER's generator gives
every key `num_needle_v` values, so the E2/E3 grids are restated with feasible token budgets;
needle depth needs a one-flag generator patch; the prompt is `input + answer_prefix`; `vis` is
built inside the patched layer from the 2-D pad mask (SDPA loading hands the layer no additive
mask); decode-time fan-out is a per-key re-solve at frozen `tau_j`; B5 is teacher-forced only;
`nu` is NOT detached; the dense arms' precision is reported on both domains; the `T_j`
construction has a pre-registered DEFAULT pending Luke's confirmation. The engineer's verdict:
sufficient without questions for everything except the five items in §17.

**What changed in v4.12 — the paper caught up (record §60).** The eight cap patches of §59.4 are
**applied** to `claude/marsea_iclr.tex`, together with the two edits (D-31, D-9a) the working copy
in `~/Dev/marsea` was missing; `verify_all.py`'s `Prop. A(iii)` row is re-targeted to the row's own
softmax mass and prints what the paper claims; the project and the repo now hold the same bytes;
and the master record is one file again, §1–§60. Nothing in the mechanism changed, so **v4.11's
contract below stands unaltered** — only §17.1 item 1 moves from open to done.

**⚠️⚠️ What changed in v4.11 — THE UNIT CAP, SOLVED IN GENERAL (record §59).** Marked
**[v4.11]** below, and it supersedes v4.10 item 1 outright.
1. **`cap_slack` is DELETED, together with its ceiling, its formula and `bind_tol`.** Three rounds
   of constants were each right on exactly one kernel, because the fp32 row sum of a softmax is off
   one by `n_k·eps/(2L)` with `L` the reduction's lane count — ~1e-6 on CUDA, 6e-5 at 16 lanes,
   1.2e-4 at 8, up to 1e-3 scalar. **No constant is correct on an unknown CPU.**
2. ⭐⭐ **The cap now binds on the RELATION'S MASS EXCESS and projects onto the row's own softmax
   mass.** `row_masses(Atil, A_sm, E) -> (excess, sm_mass)` is one key-chunked masked fp64 pass;
   `unit_cap(Atil, vis, excess, sm_mass) -> (a1, theta, binds)` binds iff `excess > CAP_TOL = 1e-6`
   and projects a binding row onto `sm_mass`, and `proj_le_masked` refuses to bind when its own
   arithmetic finds `theta <= 0`. Since `Atil == A_sm` **bitwise** off `E`, the excess equals
   `sum_j Atil - sum_j A_sm` exactly and **reads no row total**: the decision is kernel-free on any
   device, and an empty relation gives excess exactly `0.0` so T0 and sanity 5(a) hold bitwise.
3. **INV-3 and INV-5 are restated against `sm_mass`, and INV-5b is new.** The paper's
   `prop:twostep`(iii) becomes `sum_j A_ij <= sum_j A_sm_ij` — see §17.1 and
   `claude/marsea_paper_edits_cap.md`.
4. **`softmax_about_max` on the chunked path** (`exp((S-m)-r)`, flat at ~1e-6 in `n_k`); the dense
   and decode paths keep `torch.softmax` for T0 bit-compatibility, so the two differ by ~1e-6
   relative and the appendix must say which produced the chunked numbers.

**What changed in v4.10 (three review rounds; record §54–§58).** Marked **[v4.10]** below.
1. ⚠️ **SUPERSEDED BY v4.11 ITEM 1** — kept as the index of a dead end. The unit cap's binding
   slack was made size- and dtype-dependent. `CAP_TOL = 1e-6` is a bulk-mode
   failure at long context — 44 % of fp32 rows exceed `1 + 1e-6` at 16K, because torch's fused
   softmax carries a systematic positive bias ~80× that of `exp(S−max)/sum`. `cap_slack(n_k, dtype)`
   now supplies `bind_tol` at every site that binds or checks the unit cap (§3, §5, §7, §2's
   invariants). **This changes the forward pass, not a test tolerance**, and it changes INV-3 from
   `≤ 1` to `≤ 1 + δ(n_k)` — a paper-level change (§17.1).
2. **Two sites may share a layer** (D-9a: the coreference head and the detector's `(l*, h*)`). Every
   per-head slice, every merge across head blocks, and every diagnostic reader must carry
   `(layer, head)` pairs, never a layer-keyed map (§12.2, H15, T18).
3. **Dense and sparse paths must agree on every reported quantity**, not only on the mechanism's
   outputs. Every defect that survived the review rounds was an *instrumentation* defect of this
   shape (H16, T19).
4. §19 records the evaluation-queue contract, which the review found could not produce the paper's
   tables.

**Still the single most important thing**: the fan-out capacity is **inherited, never
predicted**. Step 1 hands the relation the mass an unmodified layer would have given it; step 2
re-shapes that mass and produces the exact zeros. The claim is a bound on **degree**, not mass.
Code that predicts `c_j` from a head is implementing an earlier version of this paper.

Target: PyTorch ≥ 2.3, `transformers` ≥ 4.48 (the `AttentionInterface` refactor; verified
against `main` on 2026-09-03), `peft` for LoRA. fp32 inside the programs (§10). Everything below
is per attention head; batch and head dimensions vectorise trivially except where noted.

---

## 1. The backbone — one decision, stated once

**All seven arms (MarSea and B0–B5) run on the same borrowed pretrained decoder, with only the
normalisation step of a subset of attention layers replaced.** Nothing in the attention core —
projections, RoPE, GQA, residuals, MLPs — is ours. "Borrowed" means: weights loaded from the
Hub, architecture code from `transformers`, our code entering at exactly one line.

### 1.1 Decision

| role | model | why |
|---|---|---|
| **primary** | `Qwen/Qwen2.5-1.5B` (base, not Instruct) | Apache 2.0; 28 layers; 12 Q-heads / 2 KV-heads (GQA 6:1), head dim 128, hidden 1536; 32K native context (config `max_position_embeddings` 131072, `rope_theta` 1e6, no sliding window); bf16 checkpoint; fits a 16K-context LoRA run on one 80 GB GPU with ≤ 4 patched layers (§6.5) |
| **cross-family check** | `meta-llama/Llama-3.2-3B` (base) | Llama 3.2 Community License (gated repo — request access first); 28 layers; 24 Q / 8 KV (3:1), head dim 128, hidden 3072; 128K context via llama3 RoPE scaling; same `eager_attention_forward` contract, so the same patch applies verbatim. ⚠️ **[v4.10] No Llama arm exists in the queue** (§19); the paper's Setup paragraph claims this check. |
| scale check (optional, S4) | `Qwen/Qwen2.5-7B` | Apache 2.0; 28 Q / 4 KV; run E3 only, LoRA only, 2 patched layers |

Not `Qwen2.5-3B`: it is under the *Qwen Research License*, not Apache 2.0, and costs 2× the 1.5B
for the same head geometry. The paper's "1–3B decoder" (Sec. 5, Setup) is satisfied by 1.5B + 3B.
Use **base** checkpoints: every arm is fine-tuned under the same schedule (§9), and an Instruct
model's prior on answer formatting would be an uncontrolled advantage for whichever arm it suits.

### 1.2 The patch point

In `transformers/models/qwen2/modeling_qwen2.py` (and identically `modeling_llama.py`):

```python
def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states   = repeat_kv(key,   module.num_key_value_groups)      # GQA expand: [B, H_q, n_k, d]
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling     # S  [B, H_q, n_q, n_k]
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask                             # additive, -inf/min off-visibility
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)   # <-- THE LINE
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output  = torch.matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2).contiguous(), attn_weights
```

`Qwen2Attention.forward` selects the function by
`ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)`
and calls it as `attention_interface(self, q, k, v, attention_mask, dropout=..., scaling=self.scaling,
sliding_window=self.sliding_window, **kwargs)`. **MarSea replaces the softmax line and nothing
else.** `S = attn_weights` (already scaled and masked) is `eq:scores` — the backbone supplies no
`b(gamma_ij)` term and we add none; RoPE is inside `query`/`key` already.

⚠️ **[v4.11] That softmax's row-sum error is a kernel property, and the cap no longer reads
it.** `torch.softmax` (not a hand-written `exp(S−max)/sum`) returns rows whose fp32 sum differs
from 1 by up to `delta_max(n_k) = n_k*eps/(2L)`, where `L` is the reduction's lane count — a
kernel- and device-dependent quantity (~1e-6 on CUDA's pairwise reduction, 3e-5 at 8K on a
16-lane CPU kernel, up to `n_k*eps/2` scalar). v4.10 tried to calibrate a slack against it; v4.11
does not, because the unit cap tests `excess[i] = sum_j (Atil - A_sm)` and projects onto
`sm_mass[i]`, neither of which reads a row total against the literal 1 (§3, `row_masses` /
`unit_cap`; D-37). Nothing here needs re-measuring to be *correct* on a new device. Do still
measure the kernel's own `delta` on the GPU the paper's numbers come from and **report** it beside
the checks (App. F), and state the device.

Every arm is a `Normalizer` (§8) called at that line. The value-mix, output projection and
everything downstream are the backbone's. That is what "a mechanism ablation, not a leaderboard
entry" (paper §5) means operationally.

### 1.3 Facts to re-verify locally before the first run

`config.num_attention_heads`, `num_key_value_heads`, `head_dim`, `max_position_embeddings`,
`torch_dtype`; the exact `eager_attention_forward` signature of the installed `transformers`
(it has changed twice since 2024 — pin the version in `requirements.txt` and assert the
signature at import); and for Llama-3.2 that the gated download succeeded.

---

## 2. Shapes, names, invariants

```
n_q, n_k          queries, keys (self-attention: n_q = n_k = T, the sequence length)
B, H              batch, Q-heads (12 on Qwen2.5-1.5B).  KV-heads H_kv (2); group g = H/H_kv
S      [B,H,n_q,n_k]  scores after scaling AND additive mask (eq:scores); -inf where invisible
vis    [B,1,n_q,n_k] bool, visibility, derived from the backbone's additive mask (§6.1); False =>
                  excluded outright.  Causal with a KV offset: vis[i,j] = (j <= i + n_k - n_q).  Padding
                  rows/cols are False.  [v4] A query row with NO visible key (left-padding) gets A = 0,
                  is excluded from every statistic, and SAN-1 counts only rows with >= 1 visible key.
E      [B,H,n_q,n_k]  bool, THE RELATION.  E[...,:,j] is E_.j ; E[...,i,:] is E_i.   ONE object (INV-5)
A_sm   [B,H,n_q,n_k]  row_softmax(S)  -- standard attention, fp32
Atil   [B,H,n_q,n_k]  Stage-1 output  (paper: \tilde A)
a1     [B,H,n_q,n_k]  fan-in step-1 output
A      [B,H,n_q,n_k]  final attention (cast back to query dtype on exit)
c      [B,H,n_k]      inherited column mass          c_j    = sum_i A_sm[i,j]
cbar_j [B,H,n_k]      the relation's column quota    cbar_j = sum_{i in E_.j} A_sm[i,j]
cbar_i [B,H,n_q]      the relation's row quota       cbar_i = sum_{j in E_i.} a1[i,j]
tau_j  [B,H,n_k]  > 0 fan-out exclusivity     tau_i [B,H,n_q] > 0 fan-in exclusivity
theta  [B,H,n_q]  >= 0 row dual (derived, never a parameter)
kstar  [B,H,n_k]  |supp(p_.j)|;   nu [B,H,n_k] = ||p_.j||_2^{-2}   (state, see §5.3)
Rtil   [B,H,n_q]  sum_{j in E_i.} Atil[i,j]   (the row's relation mass before step 2)
```

**Runtime invariants** — assert all in `MARSEA_DEBUG=1`, on every forward of every patched layer.
[v4.5, 2026-09-08] The list was re-audited item by item: each entry below is a statement the
mechanism makes, is true of the code path as written, and is not implied by another entry.
What was removed, and where it went, is listed after the table.

```
INV-1  sum_{i in E_.j} Atil[i,j] == cbar_j[j]     for every j with E_.j non-empty
       -- Stage 1 carries the relation's quota exactly (prop:twostep(i)); p is a distribution on E_.j
INV-2  Atil[i,j] == A_sm[i,j]                       for every (i,j) NOT in E
       -- Stage 1 writes only inside the relation (prop:twostep(iv), column half).  Exact.
INV-3  A >= 0  and  sum_j A[i,:] <= sm_mass[i] + CAP_TOL + tol_sum     for every real row i
       -- [v4.11] the row-mass bound (prop:twostep(iii)), restated against THE ROW'S OWN SOFTMAX MASS
          rather than the literal 1.  sm_mass[i] = sum_j A_sm[i,j] as the fp32 tensor holds it, summed
          in fp64 (row_masses); it IS one in exact arithmetic, and it differs from one by the KERNEL's
          error, which is not a property of the mechanism.  Comparing sum_j A against a constant is
          what made three rounds of slack constants necessary and each of them device-specific.
          Both sides are fp64 sums of fp32 entries, so the check adds no error of its own -- the
          fp32-era tol_sum is loose here by ~500x and should be tightened (record §59.6).
INV-4  A[i,j] == a1[i,j]                            for every j NOT in E_i.
       -- Stage 2 writes only inside the relation (prop:twostep(iv), row half).  Exact.
INV-5  excess[i] <= CAP_TOL  =>  a1[i,:] == Atil[i,:]   exactly
       -- [v4.11] the unit cap is the identity on a row the relation did not push over its own softmax
          mass: step 1 of Stage 2 is NOT a second softmax.  The test recomputes the excess from the
          FULL row independently of the decision, so it is not a restatement of the branch taken.
INV-5b excess[i] > CAP_TOL  =>  a1[i,:] <= Atil[i,:] elementwise, no Atil zero lifted, and
       sum_j a1[i,:] == sm_mass[i] (+- tol_sum)
       -- [v4.11] what a BINDING row must satisfy.  Before the projection target was changed from the
          constant 1 to sm_mass, a binding row whose fp32 total sat BELOW one was scaled UP: theta went
          negative and Stage-1 zeros were resurrected in a1 (measured: 1110 entries at tau_j = 200,
          rho = 0.6), and a1's off-relation entries ARE A (INV-4), so the output moved device-dependently
          (record §59.3).  EMIT IT AS (True, 0.0) WHEN NOTHING BINDS -- check_invariants only fails on
          keys that are present and False, so an absent key reads as a pass.
INV-6  Atil[i,j] == 0  =>  A[i,j] == 0                exact
       -- a Stage-1 zero is permanent (prop:stagecomp(iv)); equivalently theta_i >= 0 wherever step 2
          runs, which the CAP (not an equality) guarantees.  supp(A_.j) subset supp(Atil_.j) is this
          statement read column-wise, not a second invariant.
INV-7  sum_{j in E_i.} A[i,j] <= cbar_i[i] + tol_sum   for every i with E_i. non-empty
       -- the row quota is a CEILING (prop:twostep(v)), never exceeded, met only when it binds.
          [v4.11] The QUOTA cap decides internally at tolerance 0 (proj_le_masked with binds=None):
          cbar_i is a real mechanism quantity the theory names, not a softmax row sum with a kernel
          error, so widening it would move a threshold the paper states.
INV-8  ~vis[i,j]  =>  E[i,j] False  and  A_sm = Atil = a1 = A = 0 at (i,j),  and (i,j) enters no
       sort, sum or statistic -- masking is total (H5, H10, H13)
INV-9  E all-False  =>  A == A_sm  to 1e-6  (ELEMENTWISE: tolerance does NOT scale with n_k)
       -- an empty relation IS standard attention (prop:twostep(vi)); the only route back to it
```

**Tolerances [v4.10] — index each on the axis actually summed.**
- `tol_sum = (1e-12 if fp64 else 1e-5) * max(1, (N/64)**0.5)` where **`N` is the length of the sum
  being checked**: `n_k` for the row sums of INV-3 and INV-7 and for SAN-1; `n_q` for INV-1, which
  genuinely sums a column. Getting this backwards makes the check loose at one shape and false at
  the other. ⚠️ **[v4.11] INV-3 and INV-5b now compare two fp64 sums**, so this fp32-era value is
  loose by ~500× at 2K and ~1000× at 16K — a `+3e-4` corruption of `A` does not trip INV-3. The
  honest bound is `sm_mass + CAP_TOL + O(n_k·2⁻⁵²)`.
- **SAN-1 is a statement about the DEVICE, not about MarSea** [v4.11]: it bounds the kernel's own
  row-sum error at `n_k·eps/2` (the `L = 1` worst case — 6.0e-4 at 8K), and that number is
  *reported* beside the checks, never used inside one.
- INV-9 is **elementwise** and takes the plain `1e-6` — it sums nothing.
- Exact (`==`) for INV-2, INV-4, INV-5, INV-6, INV-8.
- A quota `cbar_i <= 1e-9` is treated as zero (the harness's `TOL`).
- ⚠️ **A sparse-path check must compare against an INDEPENDENT quantity.** A check that tests
  `rowsum` against a threshold *defined from that same `rowsum`* is a tautology and will stay green
  through an arbitrarily broken cap; either recompute the quantity from the store or list the check
  in `_not_checked` and say so. Half-checking and reporting a pass is worse than not checking.

**Removed from the runtime list (v4.5), and why.**
- *Column total `sum_i Atil[:,j] == c_j`* (old INV-1/1'): a corollary of INV-1 + INV-2; asserting it
  adds nothing they do not already catch.
- *`sum_j c_j == n_q`* (old INV-2): an identity of the row-softmax, not of MarSea → SAN-1 below.
- *"E is one boolean tensor"* (old INV-5): a rule of construction, not a numeric assertion; it is
  §4's first sentence and the code has one `E`.
- *`supp(A_.j) ⊆ supp(Atil_.j)`* (old INV-7) and *`theta_i >= 0`* (old INV-10): both are INV-6.
- *`psi_j^(t)` non-decreasing* (old INV-9): a DECODE-time property at sealed `tau_j` (prop:prefix);
  nothing to assert on a teacher-forced forward.  It is test T7.
- *`tau_j -> 0` gives the uniform re-shaping* (old INV-13): a limit, never a runtime state
  (`tau_min = 0.05`).  It is test T12 and hazard H9.
- *fan-in target is `Atil` (pre-cap), quota from `a1` (post-cap)* (old INV-14): a rule of construction
  with a detector test (T14); not checkable on a forward without recomputing it.

**Backbone / mask sanity (not mechanism invariants) [v4.4].** `SAN-1  every real query row of A_sm sums
to 1 (+-tol_sum on n_k); fully-masked pad rows are exactly 0`. This is the whole content of the former
INV-2 (`sum_j c_j == n_q`), which is an identity of the row-softmax — `sum_j sum_i A_sm_ij = sum_i 1` —
and carries no information about MarSea; it fails only on an unnormalised or NaN row, which is why it
stays as a sanity check on masking and padding. Under inherited capacity nobody cares about the
whole column's mass (Luke, 2026-09-08); the earlier design imposed `sum_j c_j = n_q` as a budget
anchor, and that is the only reason the equation was ever listed.

INV-6/7 are guaranteed only because fan-in step 2 projects onto an *inequality* set. Test them
against the real `eq:rowprog` code path, never against `max(Atil - th, 0)` with a sampled `th`.

**[v4.10] An empty invariant report is a FAILURE, not a pass.** A report with no rows means the
checker never ran; raise. And on any path where a check cannot be made (the sparse path cannot see
a dense row sum), list it in `_not_checked` and print that list beside the report — a report that
silently omits what it could not check overstates itself.

---

## 3. Primitives

```
FUNCTION sorted_prefix_stat(z):          # Lem. 1's G, and k*.  z is a 1-D fp32 vector WITHOUT masked entries
    u  = sort(z, descending, stable=True)     # u[0] is the largest      (0-based arrays throughout [v4])
    cs = cumsum(u)                            # cs[r] = u[0] + ... + u[r]
    k  = arange(1, len(z)+1)                  # k[r] = r + 1, the 1-based prefix length
    G  = cs - k * u                           # G[r] = sum_{t<=r} (u[t] - u[r]), non-decreasing in r
    r_star = max{ r : G[r] < 1 }              # equivalently  1 + k[r]*u[r] > cs[r]
    kstar  = r_star + 1
    psi    = (cs[r_star] - 1) / kstar         # NOT cs[kstar]: that is off by one
    RETURN kstar, psi, G

FUNCTION sparsemax(z):                   # alpha = 2, eq:affine
    kstar, psi, _ = sorted_prefix_stat(z)
    RETURN clamp(z - psi, min=0)         # exact zeros; do NOT add epsilon anywhere

FUNCTION entmax(z, alpha):               # alpha = 2 is the deployed setting
    IF alpha == 2: RETURN sparsemax(z)
    ELSE:                                    # [v4] B4 at alpha = 1.5 only.  Bisection on the threshold t:
        lo, hi = (alpha-1)*max(z) - 1, (alpha-1)*max(z);  50 iters, fp32
        p = clamp((alpha-1)*z - t, min=0) ** (1/(alpha-1));  RETURN p / sum(p)
        # zeros are exact (the clamp); the final renormalisation absorbs the bisection residual.
        # Gradient = the unrolled iteration's.  Monotone exclusivity is proved only at alpha = 2.

FUNCTION row_masses(Atil, A_sm, E):     # [v4.11] -> (excess, sm_mass), both [B,H,n_q] fp64
    excess = zeros(B,H,n_q, float64); sm_mass = zeros(B,H,n_q, float64)
    FOR each key chunk J of width EXCESS_CHUNK (1024):
        a64      = A_sm[..., J].double()                  # cast BEFORE subtracting: the excess of the
        sm_mass += a64.sum(-1)                            #   fp32 tensors is then EXACT (~1e-16)
        excess  += ((Atil[..., J].double() - a64) * E[..., J]).sum(-1)
    RETURN excess, sm_mass
    # excess_i = sum_{j in E_i.} (Atil_ij - A_sm_ij) = sum_j Atil_ij - sum_j A_sm_ij  EXACTLY, since
    # Atil == A_sm bitwise off E (the straight-through gate is exactly 0.0 there).  So the cap's
    # decision READS NO ROW TOTAL and is independent of the softmax reduction kernel.  That matters:
    # an fp32 softmax row sums to 1 + n_k*eps/(2L) with L the reduction's lane count -- ~1e-6 on CUDA
    # (pairwise), 6e-5 at 8K on 16 lanes, 1.2e-4 on 8, up to n_k*eps/2 ~ 1e-3 scalar -- so three rounds
    # of `sum Atil > 1 + slack` constants were each right on ONE device (reviews 30372ae C, e982f83 D,
    # e16a843 B; solved in general at 9bacef9, record §59).
    # ERROR: relative to the excess, and at the DECISION BOUNDARY ~1e-8 measured (far from it, discrete
    # sparsemax-support flips show up as 1e-4; that is not an error of this quantity).  nnz == 0 gives
    # excess EXACTLY 0.0, so a forced-empty relation can never bind on any device -- T0 and sanity 5(a).
    # IMPLEMENTATION: key-chunked masked fp64.  `sum(dtype=float64)` over a whole row COPIES the
    # [B,H,T,T] tensor to fp64 (+1.5 GB at 8K/head_block 2 -- ATen casts in make_reduction); `nonzero`
    # forces a host sync per call and holds ~40 B per relation entry.  The chunked form is rho-independent
    # and sync-free; four fp64 temporaries of [B,H,n_q,EXCESS_CHUNK] are live (~0.50 GiB at 8K/hb2).

FUNCTION unit_cap(Atil, vis, excess, sm_mass):    # [v4.11] fan-in STEP 1 = proj_le(Atil_i., 1)
    binds = excess > CAP_TOL                      # CAP_TOL = 1e-6, on a quantity whose error is ~1e-8
    RETURN proj_le_masked(Atil, s=sm_mass, mask=vis, binds=binds)     # -> (a1, theta, binds_effective)
    # TARGET IS sm_mass, NOT THE LITERAL 1.  With the constant target, a binding row whose fp32 total sat
    # BELOW one (delta_kernel < -excess: a razor-edge row on a lane-wise CPU) was scaled UP by proj_eq,
    # theta went NEGATIVE, and Stage-1 zeros were resurrected in a1 -- and a1's off-relation entries are
    # A (INV-4), so the output moved device-dependently (record §59.3).  Onto sm_mass the projection
    # removes exactly the relation's excess.
    # SECOND GUARD, and it is not optional: proj_le_masked must refuse to bind when ITS OWN fp32
    # arithmetic finds nothing above the target (theta <= 0).  sw and sparsemax's internal prefix sum are
    # a THIRD kernel-dependent total, so theta >= 0 is not guaranteed by the target alone on a razor row.
    # Measured cost of the guard: fired on 0 of 46,246 intended rows; the largest excess that can escape
    # it is 8.7e-8 = 0.087 * CAP_TOL, and it is n_k-INDEPENDENT (the window is half an ulp of the fp32
    # target, not the prefix sum).  ⚠️ UNMEASURED ON A GPU: on a device whose sorted prefix scan is a
    # genuine fp32 sequential reduction the window becomes ~n_k*eps/4 ~ 2.4e-4 at 8K, 240x CAP_TOL.
    # check_unit_cap.py must bisect it on the target device and GATE it (record §59.3).

FUNCTION proj_eq(v, s):                  # argmin .5||a-v||^2  s.t. sum a = s, a >= 0        (s > 0)
    RETURN s * sparsemax(v / s)          # exact: the simplex of radius s

FUNCTION proj_le(v, s, binds):           # argmin .5||a-v||^2  s.t. sum a <= s, a >= 0
    IF s <= 1e-9: RETURN zeros_like(v)      # [v4] the zero-quota case is the MAJORITY row under random
                                            # scores (65% of rows with a non-empty relation); it is
                                            # consistent (cbar_i = 0 => Atil_E = 0) but a batched
                                            #   where(binds, s*sparsemax(v/s), w)
                                            # evaluates v/0 in the dead branch and its BACKWARD is NaN.
                                            # Use s_safe = where(s > 1e-9, s, 1) inside the branch.
    w = clamp(v, min=0)
    sw = sum(w)
    IF binds IS GIVEN:                          # [v4.11] the UNIT cap: the caller decided, from the
        eff = binds AND (sw > s)                #   relation's excess.  But refuse if THIS arithmetic
        RETURN (where(eff, proj_eq(v, s), w), theta, eff)   #   finds nothing above the target.
    ELSE:                                       # the QUOTA cap (fan-in step 2): decide here, tol 0
        RETURN w IF sw <= s ELSE proj_eq(v, s)
    # [v4.11] `bind_tol` is GONE.  The unit cap's decision is made by the caller from row_masses and is
    # kernel-free; the quota cap's threshold cbar_i is a real mechanism quantity the theory names, and
    # its test stays exact.  theta is logged, never a parameter, and MUST be >= 0 on every binding row:
    # theta >= 0 is what makes INV-6 (Stage-1 zeros permanent) true at Stage 2 step 1.
    # ⚠️ diag.theta currently carries STAGE 2's dual, not the cap's, so any assert on it is vacuous --
    # record unit_cap's theta as extra["cap_theta"] and assert on that (record §59.6).

FUNCTION implied_threshold(v, u):        # theta_i for logging: 0 if the cap was slack, else v_j - u_j on supp(u)
```

⚠️ **[v4.11] `cap_binds` is the DECISION, not the intent.** Log the `binds_effective` that
`proj_le_masked` returned — the rows actually projected — on all three paths, or E8's
`frac_cap_binds` and `frac_rows_over_unit` stop describing the run. Record the refusal count
(`extra["cap_refused"]`) beside it: today nothing distinguishes intent from decision on the dense
and decode paths.

⚠️ **[v4.11] E8 comparability.** `frac_cap_binds` and `frac_rows_over_unit` are defined by the
binding rule, and the rule changed at 9bacef9. `unit_cap_record` stamps it into every run README and
every eval table for exactly this reason — **and its string still describes the pre-v4.11 rule; fix
it.** Numbers from either side of that commit are not comparable, though in practice the change was
latent on the CPUs measured (see §17.1).

Batched form: the programs run on *ragged* index sets (`E_.j` differs per column). Implement
`sorted_prefix_stat` on a padded `[n_sets, L_max]` tensor with `-inf` **removed before sorting**
by writing the pad as `-1e30` and masking it out of `cs`/`k` (a `-inf` corrupts `G`; a large
finite pad sorts last and must be *excluded* from the prefix count, not merely sorted). Backward:
`sparsemax` is differentiable a.e. with Jacobian `diag(1_S) - 1_S 1_S^T/|S|` on its support
`S`; implement as a custom autograd Function (the entmax package's `sparsemax` is acceptable
for the unpadded reference path only).

`sorted_prefix_stat` is the one place the mechanism's numerics live. Write it once, test it
against §11 T1/T2, call it from everywhere.

[v4] **Masked scores never meet a multiplication.** `S` carries `-inf` (or `finfo.min`) off `vis`;
`0 * -inf = NaN`. Every use of `S` other than the row-softmax — `column_stats`, `tau_j * S`, the
ST gate — goes through `S.masked_fill(~vis, 0)` first and then a gather/pad of the visible or
relation entries, never `S * mask`.

⚠️ **[v4.10] A safe minimum must be representable in the dtype it guards.** `clamp_min(1e-300)` is
`clamp_min(0.0)` in fp32, so `x >= min` holds and a following `log`'s backward computes `0/0`.
Measured: 96 NaNs in `dQ` on the tree that carried it. Use `torch.finfo(dtype).tiny`.

---

## 4. The relation `E`

### 4.1 Parameterisation (paper §2.3, App. E "how the mask must be parameterized")

```
FUNCTION relation_logits(K_kv, Q, layer):                 # per Q-head
    u = U_phi[layer](K_kv)                # [B, H_kv, n_k, r]   r = 16 default; reads the KV-head key
    v = V_phi[layer](Q)                   # [B, H,    n_q, r]   reads the Q-head query
    u = repeat_kv(u, g)                   # expand to Q-heads, like the backbone's repeat_kv
    logits = einsum('bhir,bhjr->bhij', v, u) / sqrt(r) + b0[layer]      # [B,H,n_q,n_k]
    RETURN logits
E = (logits > 0) & vis
```

**[v4.8, D-31, Luke 2026-09-09] Position 0 is excluded from every relation**: `logits[..., 0, :] = -inf`
and `logits[..., :, 0] = -inf` before the threshold (so `E[:,0] = E[0,:] = False` and the ST gate has no
gradient there), and the `b0` calibration counts coverage over visible pairs EXCLUDING row 0 and column 0.
Reason: the first token of a BOS-less decoder is the attention sink, not a candidate; and the numerical
audit found rows ending with ZERO mass when a query's only visible key re-shaped its quota away from it
— 24 of the 27 such rows were query 0. The sink's row and column stay at standard attention. (Paper
Sec. 2.3 states it.) Other rows can still end sub-unit, including empty, when every visible key in the
row's relation re-shapes away from it; that is permitted by INV-3, rare (3/2592 random rows), and the
E8 block logs the fraction of rows with `A.sum(-1) == 0`.

⚠️ **[v4.10] That zero-mass fraction is the paper's own "rare (3/2592)" number, and it must be
computed the SAME way on both paths.** The dense path can test `A.sum(-1) == 0`, which is exact in
any precision because a sum of non-negatives is zero iff every term is. The sparse path reconstructs
the row sum as `X − Y + Z` with `X ≈ Y ≈ 1`, and the zero case is precisely maximal cancellation
(measured fp32 residual: median 6.0e-8, max 3.6e-7), so a bare `<= 0` decides at random *and*
varies between identical runs because CUDA `index_add` is non-deterministic. Use a tolerance of
`4·eps(dtype)·sqrt(n_k)`, report both the tolerance and the count of rows inside the ambiguous band,
and say in the E8 schema which predicate produced the number. **A key that means one thing on the
dense path and another on the sparse path is not a measurement** (H16).

`U_phi`, `V_phi`: `Linear(d_head, r, bias=False)` per layer, **shared across heads** (a per-head
version is an E9 ablation); weights `~ N(0, 1/d_head)` (variance `1/d_head`, std `1/sqrt(d_head)`).
[v4] `b0` is a trainable scalar per layer; it is the paper's `e_ij = 1[<u,v> > 0]` with a constant
coordinate appended to `u` and a `1` to `v`, so it is still a function of the pair alone and
arrival-sealing is unaffected; `1/sqrt(r)` is a scale, absorbed the same way. The keys are the
**post-RoPE** keys the patch point hands over (`key` in `eager_attention_forward`), so the relation
inherits relative-position dependence; that is deliberate and stated. Pairwise is *required*, not
tolerated: membership must depend on `(k_j, q_i)` alone so it is arrival-sealed (`prop:prefix`).
**Never** let `E` read the query set as a whole.

### 4.2 Initialisation — ⚠️ this follows from the erratum

Training must start at (approximately) standard attention, and the only way there is a
(near-)empty relation (INV-9), *not* a small `tau_j` (T12, H9). But an *exactly* empty relation
receives no gradient (§4.3), so:

```
b0 <- calibrated on one warm-up batch so that  mean(E & vis) / mean(vis)  ==  rho_0 = 0.05
```

i.e. 5 % of visible pairs are in at step 0. [v4] The calibration runs **once, at the start of
Phase B, on the Phase-A weights** (§9), with `U_phi, V_phi` freshly initialised; `b0` is trainable
afterwards. Log `rho = coverage` every step from step 0 (§12.4). `rho_0` is pre-registered; E9
ablates `{0.02, 0.05, 0.2}`.

### 4.3 Gradient path — straight-through on the interpolation

Boolean indexing has no gradient. The relation enters the programs in exactly four places, and
each is written as an *interpolation between the untouched entry and the re-shaped entry* with a
gate `g_ij` that is **hard in the forward pass and sigmoid in the backward pass**:

```
soft = sigmoid(logits / T_st) * vis                                             # [v4] masked: no gradient off vis
g    = E.float() + (soft - soft.detach())                                        # ST, T_st = 1.0
cbar_j  = sum_i g_ij * A_sm_ij                           # (1) differentiable in g
Atil_ij = A_sm_ij + g_ij * (cbar_j * p_ij - A_sm_ij)     # (2) p from the HARD set; p_ij := 0 off it
cbar_i  = sum_j g_ij * a1_ij                             # (3)
A_ij    = a1_ij + g_ij * (tau_i * u_ij - a1_ij)          # (4) u from the HARD set
```

Forward, `g in {0,1}` so (1)–(4) are exactly `eq:step1`–`eq:rowprog` and every INV holds.
Backward, `d/dlogit` flows through the *difference* between the re-shaped and the untouched
entry — the estimator is biased (membership of the simplex is not differentiated) and that is
stated in the paper's limitations; do not "fix" it with a soft relaxation in the forward pass,
which would destroy the exact zeros the mechanism is. Alternative for E9: hard-concrete
(`Gumbel-sigmoid`, temperature annealed 1.0 → 0.2) in place of ST. `tau_j, tau_i` receive exact
gradients through the projections (§3).

⚠️ **[v4.11] The binding test is a boolean branch, and `row_masses` is `@torch.no_grad`**, so no
gradient path runs through the decision and none was lost when the criterion changed: `gradcheck`
passes with `binds` supplied and with `binds=None`, and against the previous tree with coinciding
binding sets every parameter gradient is **bitwise identical**. What the branch still carries is a
forward discontinuity in row mass of size `excess`, which the criterion makes device-independent
where a slack constant did not. Runs across the change of rule are not bit-comparable, and the run
README records the rule in force.

### 4.4 Degeneracy guard (App. E cost (a); the failure mode to instrument first)

At `|E_.j| = 1` cardinality is product-form and `thm:separation` says nothing, while the collapse
variances still look healthy. Log the **distribution** of `|E_.j|` and `|E_i.|` from step 0, not
the mean. A fan-out guarantee is claimed only on columns with `|E_.j| >= 2`. A run whose `|E_.j|`
mass concentrates at 1 has trained its way out of the theorem and its numbers mean nothing;
a run whose coverage falls to 0 has become B0 and must be reported as such.

The unrestricted mechanism is `E = vis`. Every code path must be correct at that setting; it is
the one the theory is stated on and the one `verify_all.py` mostly runs.

---

## 5. The MarSea normaliser (dense reference implementation)

```
FUNCTION marsea_normalize(S, vis, K_kv, Q, state, params):      # S already scaled + masked
  S32  = S.float()
  A_sm = row_softmax(S32)                          # fp32; -inf entries give exact 0
  c    = A_sm.sum(-2)                              # [B,H,n_k]    (SAN-1: rows of A_sm sum to 1)
  logits = relation_logits(K_kv, Q); E = (logits > 0) & vis; g = straight_through(E, logits)

  # ---- STEP 1, fan-out (eq:step1): inherited quota.  cbar_j reads A_sm, NEVER the raw S
  #      (sum_E s_ij is signed, <= 0 on ~half of columns, and cannot scale a simplex vector).
  cbar_j = (g * A_sm).sum(-2)                      # [B,H,n_k]

  # ---- STEP 2, fan-out (eq:program): SHAPE + exact zeros, over the relation only
  col_stats = column_stats(S32, vis)               # §5.2; O(n) per column, no n_q x n_k extra memory
  tau_j = tau_min + softplus(TauK(K_kv[j], col_stats[j], state.nu_prev[j]))     # [B,H,n_k]
  p     = ragged_entmax(tau_j * S32 restricted to E_.j, alpha)    # p_ij = 0 off E_.j
  Atil  = A_sm + g * (cbar_j[...,None,:] * p - A_sm)
  kstar = (p > 0).sum(-2)
  nu    = where(|E_.j| > 0, 1 / (p*p).sum(-2).clamp_min(1e-12), 1.0)    # [v4] nu := 1 on an empty column
  state.nu_next = nu                                                     #      (never 1/eps; nu in [1,|E_.j|])

  # ---- STEP 1, fan-in (eq:rowstep1): the row as Stage 1 left it, CAPPED at one unit.
  #      NEVER softmax(Atil): it is already attention-space; a re-softmax moves the off-relation
  #      entries away from A_sm and resurrects every Stage-1 zero (0 -> exp(0)/Z > 0).
  excess, sm_mass = row_masses(Atil, A_sm, E)      # [v4.11] ONE key-chunked fp64 pass; see §3
  a1, th1, cap_binds = unit_cap(Atil, vis, excess, sm_mass)   # binds iff excess > CAP_TOL; target sm_mass
  cbar_i = (g * a1).sum(-1)                        # [B,H,n_q]
  Rtil   = (E * Atil).sum(-1)

  # ---- STEP 2, fan-in (eq:rowprog): SHAPE + exact zeros; the quota is a CEILING
  tau_i = tau_min + softplus(TauQ(Q[i], row_summary(Atil[i, :])))   # [B,H,n_q]
  u     = ragged_proj_le(Atil restricted to E_i., cbar_i / tau_i, binds=None)  # <=, NOT proj_eq (INV-6, INV-7)
        # [v4] the TARGET is Atil (pre-cap); the QUOTA is from a1 (post-cap).  Rule of construction; T14.
        # [v4.11] binds=None here: the QUOTA cap decides internally at tolerance 0, because cbar_i is a
        # mechanism quantity the theory names, not a softmax row sum with a kernel error.
        # cbar_i <= 1e-9  =>  u = 0 (proj_le's zero-quota branch), consistent since Atil_E = 0 then.
  A     = a1 + g * (tau_i[...,None] * u - a1)
  theta = implied_threshold(Atil[i, E_i.], u)      # theta_i = 0 if slack else (Atil - u) on supp(u); log only
  RETURN A.to(S.dtype), Diagnostics(E, logits, cbar_j, cbar_i, tau_j, tau_i, psi_j, kstar, nu, theta, Rtil,
                                    supp_rel_i = (A * E > 0).sum(-1),          # B5 needs this
                                    cap_binds, excess, sm_mass, cap_theta = th1, p, a1, Atil)
```

### 5.1 `tau_i` direction
`a / tau_i` inside the norm, **not** `a * tau_i`; with `a * tau_i` larger `tau_i` gives a *larger*
support (verified: support 10→2 as `tau_i` grows under `a/tau_i`, 4→10 under `a*tau_i`).

### 5.2 Field statistics and heads

```
column_stats(S, vis)[j] = (max, second max, mean, std, max - second max, (max - mean)/std)  over visible i   -- 6 floats
    [v4] on S.masked_fill(~vis, 0) with a visible-count n_vis_j:  std = BIASED std over visible entries,
    clamped at 1e-6 in the ratio;  second max := max when n_vis_j < 2 (gap 0);  all six := 0 when
    n_vis_j = 0.  n_vis_j = 1 occurs on the LAST key of every causal block -- an unguarded std NaNs
    tau_j there and the whole loss with it.
row_summary(Atil, E)[i] = (Rtil_i, cbar_i, max over VISIBLE j of Atil_ij, H(Atil[i,E_i.] / Rtil_i))     -- 4 floats
    [v4] Shannon entropy of the relation part normalised by Rtil_i; 0 when Rtil_i = 0.

TauK : MLP( concat(LN(k_j) [d_head], f(column_stats)[6], log nu_prev[1]) -> 64 -> 1 ), GELU ; out = tau_min + softplus(.)
TauQ : MLP( concat(LN(q_i) [d_head], f(row_summary)[4]) -> 64 -> 1 ), GELU                   ; out = tau_min + softplus(.)
    [v4] LN = LayerNorm(d_head) on the vector input (post-RoPE keys have norms in the tens);
         f(x) = sign(x) * log1p(|x|) on every scalar feature.  Without these the head's inputs differ
         by two orders of magnitude and it trains badly.
tau_min = 0.05.
Init [v4]: last-layer weight ~ N(0, 0.01^2) so the output is its bias +- O(0.01);  the bias of TauK is
    CALIBRATED on one warm-up batch to softplus^{-1}( median_j 1/std_j - tau_min )  -- "tau_j ~= 1/std"
    is a median over columns, not a per-column value;  the bias of TauQ = softplus^{-1}(1 - tau_min).
```
Per layer, shared across heads (per-head is E9). `B3` is this with `column_stats` and `nu_prev`
zeroed at the input (byte-identical otherwise).

### 5.3 `nu_j` is not circular but it is ordered
`nu_j` is a statistic *of* `p_.j`, so it cannot feed the solve that produces `p_.j`. Take it from
the previous **patched layer** (the state is passed layer to layer through `kwargs`; the first
patched layer uses `1.0`, the value of a singleton support and the bottom of `[1, |E_.j|]`). In the
causal/streaming form (§7) it comes from the key's previous block. Never iterate to a fixed point.
[v4.1] `nu` is NOT detached: the paper says it carries a gradient (unlike `|supp|`), and that
gradient reaches the previous layer's `tau_j` through `p`. The reference implementation detaches it;
change that.

### 5.4 `tau_i = 1` is the identity on step 1's row [v4.1, reverses v3's §5.4]
At `tau_i = 1`, step 2 returns `a1[i, E_i.]` EXACTLY on every row: on a slack row the quota equals
the relation's supply and no floor is needed; on a cap-binding row the quota is what step 1's own
floor left, and the projection's uniqueness recovers that same floor (verified: 0 differences on
4,338 cap-binding rows). So the standard-attention start is `tau_i = 1` (§5.2 init), `tau_i > 1`
raises the floor and sharpens, `tau_i < 1` lowers it and scales the relation down. The v3 claim
"`tau_i = 1` is not an identity; the threshold is `cbar_i/Rtil_i`" compared step 2 with `Atil`
instead of `a1` and was wrong. E8 logs `tau_i` about 1. (`cbar_i/Rtil_i` remains the point where
step 2's OWN dual becomes positive; it is not the regime boundary that matters.)

---

## 6. Backbone integration

### 6.1 Module swap, per layer

```
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=bfloat16, attn_implementation="sdpa")
for l in PATCHED_LAYERS:
    old = model.model.layers[l].self_attn
    new = MarSeaAttention(old.config, layer_idx=l, normalizer=arm.normalizer)   # subclass of Qwen2Attention
    new.load_state_dict(old.state_dict(), strict=False)                          # projections copied; new heads fresh
    model.model.layers[l].self_attn = new
```
`MarSeaAttention.forward` is `Qwen2Attention.forward` with the attention call replaced by
`marsea_eager_forward`, which is `eager_attention_forward` with the softmax line replaced by
`normalizer(S, vis, key, query, state)`. **Unpatched layers keep SDPA/flash.** Do not set
`config._attn_implementation` globally (that would force eager everywhere and 3× the memory).
`attention_mask` reaching a patched layer must be the additive 4-D mask: pass
`attn_implementation="sdpa"` at load but construct the patched layers' mask with
`AttentionMaskConverter` / the model's `_update_causal_mask` in *eager* mode — assert its shape
`[B,1,n_q,n_k]` and that its minimum is `finfo.min` or `-inf`. Derive `vis = mask > -1e30`.
[v4] `vis` keeps its `[B,1,·,·]` batch dimension (padding differs per sequence). We use **right
padding and no packing** (§9), so with a causal mask every real query row has >= 1 visible key;
pad query rows are set fully invisible and get `A = 0`. Under left padding (HF default for
generation) a fully-masked real row would NaN the row-softmax: assert none exists.
[v4.1] **Do not rely on the additive mask reaching the patched layer.** With
`attn_implementation="sdpa"` transformers passes the patched layer `None` (4.48–4.52) or a
boolean mask (>= 4.53) depending on version. Build `vis` INSIDE the patched layer from the 2-D
`attention_mask` (pad mask, `[B, n_k]`) and the causal KV offset:
`vis[b,0,i,j] = pad[b,j] & pad[b, i + n_k - n_q] & (j <= i + n_k - n_q)`; then form the additive
mask yourself for the row-softmax. Assert `vis.any(-1)` on every real row.

### 6.1a [v4.10] Diagnostic retention: which heads are kept, and the rule for merging them

To hold the memory budget the patched layers keep full `[B,H,n_q,n_k]` diagnostics only for the
sites a measurement actually reads (§55 of the record; the `F-8` finding). That retention is where
every instrumentation defect of the review rounds lived, so the contract is stated explicitly.

```
ctx.keep_dense_layers : set of layer indices that retain diagnostics
ctx.keep_dense_head   : {layer -> head} or {layer -> [heads]}     # NEVER a bare layer->head map when
                                                                 # two sites can share a layer (D-9a)
diag.extra["head_sub"]: the head, or the LIST of heads, this diagnostic was narrowed to
```

⚠️ **Rules, each of which a real defect violated.**
1. **Sites are `(layer, head)` PAIRS end to end.** Any container keyed by layer alone — a dict
   literal, a comprehension, a cache — silently drops one site when the coreference head and
   `(l*, h*)` share a layer. That case is not hypothetical: record §52 has coreference best at
   (14,5) and (14,3) with layer 14 already patched.
2. **A reader resolves the head index INSIDE the kept slice**, and stamps the record with the
   corresponding GLOBAL head. Every per-head field of one column — `A`, `S`, `E`, `p`, `tau_j`,
   `kstar`, `nu`, `psi_j` — comes from the same `(diag, head)` pair or the record mixes two heads.
3. **`_kept_head` RAISES when the head was not kept.** Returning `0` as a fallback is a silent
   wrong answer in exactly the failure class the function exists to prevent.
4. **Merging narrowed blocks CONCATENATES along the head axis in the kept order.** With head
   blocking, two kept heads can land in different blocks; assigning each block's tensors in turn
   leaves only the last, while `head_sub` still advertises both. Measured at `head_block=2,
   dense_head=[3,5]`: the value site reads the coreference head's tensors and the coreference site
   raises `IndexError`. `MARSEA_DEBUG=1` cannot catch this — `check_invariants` runs per block,
   before the merge (T18).
5. **A head index re-globalised once is not re-globalised twice.** A narrowed block's sparse store
   is already in kept-head space; adding `block_index * head_block` on top corrupts it.
6. **Narrowing a store that something else consumes is a change to that consumer.** The decode
   cache seeds from `diag.extra["sparse"]`; narrowing that store to the kept heads starves it
   (measured per-head relation counts `[137,0,0,0,0,0,0,0]` against a correct
   `[145,130,155,137,136,152,127,115]`). Either do not narrow it, or narrow the consumer's `H`
   with it.

### 6.2 GQA
Keys are shared across the `g = H/H_kv` Q-heads of a group; scores are not. The programs, `E`,
`tau_j`, `tau_i`, `nu` are all **per Q-head** (`[B,H,...]`). `U_phi` and `TauK` read the KV-head
key (repeat after), `V_phi` and `TauQ` read the Q-head query. Consequently a key has `g`
different relations and `g` different `k*` in one layer; log per Q-head, aggregate never.

### 6.3 dtype
`S` arrives in bf16 under autocast. Cast to fp32 on entry (`S32`), run everything in fp32, cast
`A` back on exit. Exact zeros survive the down-cast (0 is representable); the value-mix is bf16
as in the backbone. Under `torch.autocast` wrap the normaliser in `autocast(enabled=False)`.

### 6.4 Generation and the KV cache
Teacher-forced evaluation (§12.1) needs no cache. For generation, the causal form of §7 applies:
at decode step `t` the visible column of every cached key grows by one query. Implement
**frozen-prefix** decoding (paper `prop:prefix`(iii), over-admits never under-admits).
[v4.1, made explicit] At decode step `t` with cached keys `j <= t`:
```
A_sm[t, :]   = row_softmax(S[t, :])                                   # the new row, as always
E[t, j]      = (logit(q_t, k_j) > 0)                                  # pairwise: no past row changes
tau_j        : FROZEN at the value sealed when key j's block closed (prefill: at prefill)
cbar_j^(t)   = cbar_j^(t-1) + E[t,j] * A_sm[t,j]                      # the inherited quota grows incrementally
p_.j^(t)     = sparsemax(tau_j * S[E_.j^(t), j])   for each j with E[t,j]   # re-solve on the GROWN relation;
                                                                      # cost O(|E_.j| log|E_.j|) per such key
Atil[t, j]   = cbar_j^(t) * p_tj^(t)   if E[t,j]  else  A_sm[t, j]    # ONLY row t's entry is emitted;
                                                                      # earlier rows' outputs already stand
then fan-in step 1/2 on row t as in §5 (row-local: nothing else needed)
```
Past rows are never revised (their outputs were emitted with the prefix they saw), which is
exactly the over-admission `prop:prefix`(iii) allows. Keep per key: `tau_j`, `cbar_j`, the
relation's score list `S[E_.j, j]`. Seal-at-boundary (E9) re-solves all rows of the block at its
close and is not usable for a block's own rows during decode. Correctness test: greedy generation
with `E = ∅` must be token-identical to the unpatched model; MarSea's task-level numbers
(exact-set accuracy) come from THIS path and its cost is reported with them.

⚠️ **[v4.10] The decode path runs the dense invariant report on its single row.** It is one row;
there is no excuse for it to be the only path with no check.

⚠️ **[v4.10] `truncation_events` is TWO counters, not one.** The code summed a near-truncation
*warning* (`kstar >= 0.9 * K_ret`, a statement about whether the hierarchy is still exact) with an
actual *eviction* (a column whose relation exceeds `K_ret`, a statement about cache lossiness).
Different units, different remedies, and E8's reader cannot decompose the sum. Keep
`kstar_near_K_events` and `cache_truncations` separately, and make **both** prefill paths
(`init_from_prefill`, `init_from_sparse`) count the same events — the dense path was blind to the
eviction term entirely (measured 18 vs 94, 0 vs 64, 0 vs 137 on matched inputs).

### 6.5 Memory arithmetic and why the layers are few
Eager attention materialises `[B,H,n_q,n_k]` in fp32: at `T = 16K` that is `1 GiB` per head,
`12 GiB` per patched layer per sequence, and the normaliser keeps ~4 such tensors live
(`A_sm, Atil, a1, A`) plus `E`/`g`. ⭐ **With gradient checkpointing on the patched layers the peak
is set by ONE layer, not by `L_patch`** — under recompute only the layer currently being recomputed
materialises them. That is why 8K training fits at all. With `B = 1` per GPU, **≤ 4 patched layers
at 16K, ≤ 8 at 8K** on 80 GB. One `[1,12,T,T]` fp32 tensor is **0.81 GB at 4K, 3.22 GB at 8K,
12.9 GB at 16K**. §6.6 removes the `T²` from memory (not from time).
⚠️ **[v4.10] None of this is measured on an H100 yet.** `preflight.sh` is what turns it into a
measurement, and it must additionally profile (a) the **dense teacher-forced pass at 8K**, which
five of the seven evaluation job types default to and which no profile covers, and (b) **two
retained sites at 16K** (`--sites 2`), which D-9a's split requires and which the profile has never
been given.

### 6.6 Exact chunked implementation (required for E3 at 16K, optional below 8K)
The fan-out program is column-local given the row-normaliser; the fan-in program is row-local
given `Atil`. Off the relation every `Atil` entry *is* an `A_sm` entry, reproducible from `S`
and the row log-sum-exp. So:

```
pass 0  row LSE:      lse_i = logsumexp_j S_ij            (flash-style, O(T) memory)
pass 1  per key-chunk J (|J| = 1024):  S[:,J] -> A_sm[:,J] = exp(S[:,J] - lse) ;  E[:,J] ;
        cbar_j, p, Atil[:,J] on the relation only  -> store SPARSE (i, j, Atil_ij) for (i,j) in E ;
        accumulate  rowsum_i += sum_J Atil[i,J]   (off-relation entries are A_sm and need not be stored)
pass 2  per row-chunk I:  rows with excess_i <= CAP_TOL are untouched by step 1 (identity).  For the
        others rebuild the dense row from S[I,:], lse and the sparse relation entries and run proj_le exactly
        against that row's own sm_mass.  Both quantities come from the sparse store, not from a row total.
        Then fan-in step 2 on the sparse relation entries of each row.
output  A is returned as (dense A_sm recomputed on the fly per chunk for the value-mix) + sparse corrections.
```
Rows needing pass-2 densification are those a column relation pushed over one unit — in practice
a small fraction (harness: 69 % of rows spend *less* than a unit under random relations); log the
fraction, **count-weighted across head blocks, never max-weighted**, and note that an unweighted
mean across ragged blocks (`H % head_block != 0`) is wrong (measured 0.2865 against 0.2951).
**T11** (§11) asserts chunked == dense to 1e-6 on every tensor at `T ≤ 4K`.

⚠️ **[v4.10] The chunked path's sparse-correction transient is sized by `nnz_block`.** Raising it
from `2^22/d` to `2^26/d` takes the `[nnz, d]` transient from 16 MB to 256 MB in fp32, plus an
equal-size gather, inside a checkpointed recompute. That is the recommended trade, but it
**invalidates the memory-budget numbers until re-profiled**, and any comment claiming it "meets
the same memory target" is false.

### 6.7 Which layers and which heads (pre-registered, ablated in E9)
Run the **retrieval-head detector** of Wu et al. (2024, "Retrieval Head Mechanistically Explains
Long-Context Factuality", arXiv:2404.15574) on the *unpatched* backbone: for NIAH prompts, a
head's retrieval score is the fraction of answer tokens whose argmax attention lands on the
needle copy of that token; fewer than 5 % of heads score high. [v4] Detector set: 200 RULER
S-NIAH prompts at 4K (`type_needle_v = words`, seed 0), teacher-forced, on the unpatched
backbone; score = fraction of answer tokens whose argmax attention (over the whole row) lands on
the needle's copy of that token; a head "retrieves" if its mean score > 0.1. **Patch all Q-heads
of the `L_patch = 4` distinct layers holding the top-scoring heads** (`L_patch ∈ {2, 4, 8}` in E9;
"uniform middle layers" and "all layers at 4K" are the E9 controls). Record the detector's
scores per (layer, head) with the run; the *row* used for evidence precision (§12.2) is the
top-scoring head in the top-scoring patched layer.
⚠️ **[v4.10] `(l*, h*)` is taken from the threshold-clearing heads only**, and the detector file
carries `spec_version`, the git hash and its creation time. Preflight regenerates it unless the
`spec_version` matches, and **fails closed**: if the check itself errors, regenerate — never keep a
stale detector because a probe crashed. The evaluation harness asserts that the detector it loads
is the one the checkpoints trained against.

---

## 7. Scaling and the causal form

```
FUNCTION hierarchical_topk(S_col, tau_j, K_ret, block_size):      # prop:tournament; exact when K_ret >= k*
    per_block_top = [ top_K(S_col[b], K_ret) for b in blocks ]     # by ORIGINAL score
    survivors     = concat(per_block_top)
    RETURN entmax(tau_j * S_col[survivors])                        # ONE exact solve at the root
    # ⚠️ solve on ORIGINAL scores, never on renormalised per-block outputs.  K_ret = 64 default.
    # E8 must report the realized k* distribution: k* approaching K_ret means the hierarchy is
    # silently truncating and every E7 number is void until K_ret is raised.

CAUSAL / STREAMING (sec:method-causal):
    Q_j(t) = { i : i >= j, i <= t }        # the visible column; it GROWS
    Seal at block boundaries (B = 512): tau_j, E[:,j] and cbar_j^(b) are computed when a block
    CLOSES and frozen for that block.  A head re-reading a growing column can move psi_j DOWN
    and forfeit permanence of exclusion (prop:prefix; T7).
    ⚠️ ERRATUM: the paper's causal paragraph says "c_j^(b) is emitted then" in the vocabulary of a
    predicted budget.  Under eq:step1 the per-block quantity is the INHERITED mass over the block's
    visible column: c_j^(b) = sum over the block's visible queries of row-softmax mass on key j.
    Two emission policies, BOTH implemented and ablated (E9):
      seal-at-boundary : exact, up to B tokens of latency
      frozen-prefix    : fully incremental; over-admits, never under-admits (prop:prefix(iii))
```
Training uses the full-sequence (teacher-forced, non-streaming) form; the causal form is a
*decode-time* policy plus T7/T8. The causal mask is in `vis` throughout, so the programs never
see a future query — that is the arrival-sealing the paper relies on, and it is free.

⚠️ **[v4.11] The decode path takes the same criterion and the same target as the dense and chunked
paths**, computed from its single row. T17(a) exercises all three under `force_empty`; the decode
path on a *live* relation is still ungated (record §59.6).

⚠️ **[v4.10] "Both policies ablated" is a paper claim with no arm behind it** (§19). Either add the
arm or cut the sentence.

---

## 8. Baselines — one interface, six implementations

**Matched only if every arm shares the same backbone, the same patched layers, the same LoRA,
and differs solely in the normaliser.** Do not fork the model.

```
INTERFACE Normalizer:  A, diag = normalize(S, vis, K_kv, Q, state)      # [B,H,n_q,n_k] -> same

B0  SoftmaxNorm         A = row_softmax(S).   The backbone unchanged (must be bit-identical to the
                        unpatched model's eager path: T0).  P_j = m_j/|E_.j| exactly at every tau
                        (prop:fullsupportceiling): an identity, expect no variance.

B1  SoftmaxOneNorm      A[i,j] = exp(s_ij) / (1 + sum_{j' visible} exp(s_ij'))   -- Miller (2023), NOT Xiao et al.
                        [v4] = exp(S - logsumexp(concat(S_visible, 0))) on visible entries; the "+1" sits at
                        score 0 AFTER the 1/sqrt(d) scaling; masked entries are excluded from the sum.
                        Rows sum to < 1; NO exact zeros.  Isolates a content-INDEPENDENT sub-unit row.

B2  MESHNorm            [v4, pinned to Zhang et al. ICML 2023, "Unlocking Slot Attention by Changing
                        Optimal Transport Costs", Sec. 3 and App. A]  learned marginals on BOTH sides +
                        entropic Sinkhorn on a cost the MESH step has descended; dense output.
                          a = m * softmax_{j visible}(h_a(LN(K_kv))[j])    h_a: MLP(d_head->64->1), a repeated to Q-heads
                          b = m * softmax_{i real}   (h_b(LN(Q))[i])       h_b: MLP(d_head->64->1)
                          m = n_real_q  (sum a = sum b = the number of real query rows, i.e. the row-softmax total)
                          C = -S on vis, +inf off vis (log-domain: kernel = -inf) ;  eps = 1.0
                              (the plan of sinkhorn(exp(S/eps)) at eps = 1 is the softmax kernel with two
                               marginals: no extra temperature is introduced.  E9: eps in {0.5, 1, 2})
                          MESH step (Eq. 11 of the paper):  C'(0) = C + N(0, 1e-6);  FOR t in 1..4:
                              grad = d/dC' [ H(sinkhorn(C', a, b, eps, 5 iters)) ]     H = plan entropy -sum P log P
                              C'   = C' - lambda * grad / ||grad||_2                    lambda = 1.0 (tune in {0.3,1,3})
                              -- the descent MINIMISES plan entropy (sharpens); it is NOT differentiated
                                 through: gradients of C'(4) pass STRAIGHT-THROUGH to C (paper Sec. 3.3).
                          A = sinkhorn(C'(4), a, b, eps, 20 log-domain iters, fp32), ENDING ON THE ROW SCALING
                          "visible key" = seen by >= 1 real query; eps, lambda are global constants.
                        [v4] Under a causal mask the two-marginal problem is generically INFEASIBLE (the last
                        key is seen by one query whose row must then carry a_j alone); Sinkhorn cannot converge
                        and "row sums = b_i AND column sums = a_j" is false.  We end on the row step so rows
                        carry b_i exactly (the value mix needs a row distribution) and LOG the column residual
                        max_j |sum_i A_ij - a_j| per layer; the paper's App. H states this limitation of the
                        baseline.  Rows are NOT unit (b_i learned); report that.
                        Isolates CAPACITY WITHOUT SHAPE — the mirror image of MarSea.  Prediction: dense
                        output => P_j pinned at m_j/|E_.j| however well a, b are learned.  Decisive for (i).

B3  KeyOnlyTauNorm      MarSea verbatim EXCEPT TauK's column_stats and nu inputs are zeroed.
                        Byte-identical otherwise (same E, same TauQ).  Isolates the FIELD ARGUMENT;
                        cor:necessity predicts failure wherever two keys need disjoint intervals.

B4  RowEntmaxNorm       A[i,:] = entmax(S[i,:], alpha) over visible j, alpha in {1.5, 2} fixed per run
                        (Peters et al.); no temperature (the LoRA on q,k supplies scale).  [v4] alpha = 1.5 via
                        the bisection of §3 (unrolled gradient); alpha = 2 via sparsemax (exact Jacobian).
                        Query-local sparsity: support SIZE controlled, MEMBERSHIP uncontrolled.

B5  MatchedSparsityNorm ⚠️ TWO-PASS, evaluation-time only.  [v4] Pass 1: run the trained MarSea arm on the
                        evaluation set and record, per (example, layer, head, row), BOTH the relation E_i. and
                        n_i = |supp(A[i, E_i.])| = supp_rel_i.  Pass 2: run the trained **B0** weights (the
                        softmax-trained decoder, since the objection is "a sparsifier on top of softmax clears
                        the ceiling"), take A_sm, keep the top-n_i entries of A_sm within E_i., scale them to
                        carry sum_{E_i.} A_sm (the row still sums to 1), zero the rest of E_i., leave the
                        complement at A_sm.  n_i = 0 => the relation's entries are 0 (row < 1, as MarSea's).
                        Same examples, same order, same seed as pass 1 or the comparison is meaningless.
                        [v4.1] TEACHER-FORCED ONLY: its trace is per gold row, so B5 has attention-level fidelity
                        numbers and no task-level (generation) numbers; the table marks the cell "n/a by design".
                        E9 optional: oracle top-m_j within the relation (the strongest form of the objection).
```

---

## 9. Training recipe (one recipe, every arm)

**[v4.2] The full procedure is `marsea_training_procedure.md` (package `records/`; project
`claude/marsea_training_procedure.md`): data pool and order, loss reduction, parameter groups,
the Phase-B initialisation sequence with its sanity checks, the step, the E8 block, evaluation
and checkpoint cadence, pre-committed failure handling, per-arm differences, and what a healthy
run looks like. This section is its summary; where they differ, that document wins on training.**

```
data        RULER-generated NIAH sets (E2/E3/E4 grids, §13) + MuSiQue-Answerable train + HotpotQA-distractor train,
            mixed 1:1:1 by examples; ONE example per sequence at its OWN length (B = 1 per GPU: no padding
            and NO packing [v4] -- packing needs a block-diagonal vis, and a key from one example must never
            enter another's relation); sequences longer than L are dropped, never truncated; RULER training
            pool generated with the backbone's tokenizer at seeds 100-102 (evaluation: 0-2); template §13.1
            ⚠️ [v4.10] The mixture FAILS OPEN: if MuSiQue or HotpotQA is absent the builder silently omits it
            and the arms train on a different mixture with nothing noticing.  HARD-ASSERT all three sources
            present before S2 starts.  (MuSiQue is a manual download the data script does not verify.)
loss        next-token CE on ANSWER tokens only (prompt masked).  No auxiliary loss by default.
            Optional E9 arm: coverage-floor penalty  lambda * relu(rho_0 - rho)  — off unless pre-registered.
trainable   LoRA r=16, alpha=32, dropout 0.05 on q_proj,k_proj,v_proj,o_proj of ALL layers (same for every arm)
            + the arm's own modules (MarSea: U_phi, V_phi, TauK, TauQ; B2: h_a, h_b; B3: same as MarSea minus inputs)
            Backbone weights frozen otherwise.  Embeddings frozen.
optimiser   AdamW, beta (0.9, 0.95), wd 0.01; LR 2e-4 (LoRA), 1e-3 (new modules); cosine to 10 %, 3 % warm-up
schedule    Phase A: 500 steps with the normaliser = SoftmaxNorm for EVERY arm [v4] (for MarSea/B3 that is
            "E forced empty"; for B1/B2/B4 it is literally softmax) == plain LoRA fine-tune.  B0 IS Phase A
            continued.  Phase B: 3000 steps with the arm's normaliser live; b0 calibrated at its first step
            (§4.2); MESH's h_a, h_b and MarSea's heads fresh at its first step.  ONE cosine schedule over the
            run's total steps -- 3500 (full) or 2500 (reduced programme, Phase B = 2000; record §44) -- with no
            restart at B.  Same total steps for every arm.
batch       1 sequence / GPU, grad accumulation to 16 sequences / step; grad-clip 1.0
precision   bf16 autocast; programs fp32 (§6.3); gradient checkpointing on patched layers
seeds       3 (data order + init) on every headline configuration; report mean ± sd
eval        every 250 steps on a held-out RULER grid (fast) ; full E-suite at the end
            ⚠️ [v4.10] This in-training eval is ~125 teacher-forced + 50 greedy sequences, ten times per job --
            roughly a 20 % tax on every training run, and it appears in no compute table.  --eval_every 500
            halves it if the schedule is tight.
compute     S1 ~ 15 H100-h; S2 (E3+E2+E4, 3 seeds, 7 arms) ~ 60 H100-h at 8K; see tab:compute
```
**Why Phase A**: it makes "the same decoder under the same schedule" literal — every arm starts
Phase B from the same adapted weights — and it guarantees the relation is trained from a model
that already reads the task, not from a random one.

**Phase-B sanity check 5(a)** — the loss with `E = ∅` must equal Phase A's to 1e-4 — is the check
that exposed the whole unit-cap problem. Under v4.11 it holds **bitwise**: an empty relation gives
excess exactly `0.0`, so the cap cannot bind and `a1 == Atil == A_sm`. It must stay, and it must
report `gap_5a` as a number, not a boolean — a non-zero value now means something is wrong with the
criterion itself, not with a constant.

---

## 10. Numerical hazards

```
H1  fp32 for sort/cumsum/threshold solves even under AMP.  bf16 loses exact zeros and the zeros ARE
    the mechanism.  Cast S to fp32 entering any program, cast A back on exit.
H2  W_j = 0 whenever m_j = 1.  Then 1/W_j = +inf: the "above the interval" regime DOES NOT EXIST for
    that column.  Guard every interval computation; do not clamp or substitute; record the regime
    as undefined and exclude the column from upper-endpoint statistics.
H3  Ties: k* can fall by more than one at a tie.  Compare with tol=1e-9; never assume unit steps.
H4  NEVER add epsilon to a sparsemax output.  Downstream code must tolerate exact zeros, including
    the value-mix and any logging that divides by a support size.
H5  Masked entries are REMOVED from program vectors (§3), not set to -inf.
H6  [v4, corrected] |E_.j| = 0: no step 2 on that column, g = 0 there and the interpolation is the
    identity.  cbar_i <= 1e-9 with |E_i.| > 0 is DIFFERENT: g = 1 on the relation, so u must be defined;
    proj_le's zero-quota branch returns 0 and, since Atil_E = 0 whenever cbar_i = 0, A_E = a1_E = 0.
    This is the majority row under random scores, not an edge case.
H7  Gradient path: entmax/proj differentiable a.e.; theta is a DERIVED dual — differentiate through
    the projection, never treat theta as a leaf.  E gets the straight-through of §4.3 and nothing else.
H8  Determinism: seed per block; stable sort or k* flickers on ties across runs.
H9  tau_j -> 0 is the UNIFORM re-shaping, not standard attention (T12).  A "warm start" that sets
    tau small would start training at a WRONG-shaped column; warm-start through the relation (§4.2).
H10 The additive mask uses finfo.min, not -inf, in some transformers versions; exp(finfo.min - lse)
    is exactly 0 in fp32, but a SUM of masked scores is not.  Derive vis from the mask and use vis;
    never sum S over masked entries.
H11 [v4] Dead branches of torch.where still run backward.  Every division by a quota, a std, a norm or
    a count goes through a safe denominator (where(d > tol, d, 1)) BEFORE the where, or the gradient is
    NaN on exactly the rows/columns the guard was meant to protect (verified on proj_le and column_stats).
H12 [v4] A column seen by one query (the last key of every causal block) has std 0 and no second max;
    §5.2's guards are mandatory, not defensive.
H13 [v4] S.masked_fill(~vis, 0) before any gather/multiply; S * mask with S = -inf is NaN (§3).
H14 [v4.11] ⚠️⚠️ A REAL fp32 SOFTMAX ROW DOES NOT SUM TO ONE, and the error is a property of the
    REDUCTION KERNEL, not of fp32 arithmetic: n_k*eps/(2L) with L the lane count.  NEVER compare a row
    total against a constant to make a DECISION.  The general fix is not a tolerance but a change of
    tested quantity -- here, the relation's own mass excess (§3, row_masses), which reads no row total.
    Four corollaries.  (a) The reduction's error is FLAT in n_k (pairwise); the growth is in the softmax
    VALUES and peaks on NEAR-ONE-HOT rows with a long tail just under half an ulp of the spike (entropy
    0.002-0.25 nats, gap ~16.64) -- a search that samples random logits at a few scales MISSES it, which
    is how a log2(n_k) constant was recommended and shipped.  Prefer an ANALYTIC model of the kernel and
    a razor-edge construction to a sampler.  (b) A guard whose safe minimum underflows the dtype is not
    a guard (clamp_min(1e-300) in fp32 is clamp_min(0)); use finfo(dtype).tiny.  (c) Cast to fp64 BEFORE
    subtracting, never after: an fp32 subtraction of two entries not within a factor of two rounds, up
    to 1.2e-7 -- 8x rather than 160x under CAP_TOL.  (d) Every quantity of this kind is device-dependent:
    MEASURE it on the target device and RECORD it beside the checks, never inside one (SAN-1).
H15 [v4.10] ⚠️ TWO MEASUREMENT SITES CAN SHARE A LAYER (D-9a).  Any layer-keyed container silently
    drops one; any per-block narrowing must concatenate, not overwrite; any head index already in
    kept-head space must not be re-globalised.  See §6.1a; the failure is silent and MARSEA_DEBUG
    cannot see it.
H16 [v4.10] ⚠️⚠️ THE DENSE AND SPARSE PATHS MUST AGREE ON EVERY REPORTED QUANTITY, not only on the
    mechanism's outputs.  Every defect that survived three review rounds was of this shape: a count
    the dense prefill never computed; a zero-mass predicate exact on one path and a cancelling
    fp32 difference on the other; an E8 key averaged unweighted across ragged head blocks.  When a
    quantity is computed twice, EITHER derive both from one function OR assert them equal in T19.
    A quantity that cannot be computed on one path is listed in _not_checked, never silently omitted.
```

---

## 11. Acceptance tests

Port every block of `verify_all.py` (**1,058,450 evaluations, 0 violations, 63 checks — 62 with
trial counts and 1 structural**) to the tensor implementation and run it against the real code
path, not a numpy re-implementation. Every denominator in it has been reconciled against
`tab:verification` and `tab:transferchecks` in the paper and they match exactly.

```
T0  B0 == backbone     patched model with SoftmaxNorm reproduces the unpatched eager model's logits
                       to 1e-5 (bf16) and greedy tokens exactly; MarSea with E = ∅ likewise (INV-9)
T1  lem:prefix         support == top-k* of sorted order; G non-decreasing
T2  thm:recovery       supp(p_.j) == T_j  IFF  tau_j in [1/(W+m*delta), 1/W)   -- both endpoints
T3  prop:twostep       INV-1..INV-5, INV-7 on random E, random tau, random scores; SAN-1
T4  prop:scopemono     shrinking E: delta non-decreasing, interval containment
T5  cor:capacity       |E_.j| from 4 to 256 at m_j=4: degree stays m_j, precision stays 1
T6  prop:stagecomp     INV-6; a Stage-1 zero never resurrects (theta_i >= 0 logged)
T7  prop:prefix        psi_j non-decreasing under a growing prefix AT SEALED tau_j; frozen-prefix over-admits, never under
T8  prop:tournament    hierarchical == flat whenever K_ret >= k*
T9  prop:fullsupportceiling   every dense arm (B0, B1, B2) sits at exactly m_j/|E_.j| at EVERY tau
T10 fig:prcurve        sweeping tau_j traces the L: no interior point under ass:margin
T11 chunked == dense   every output tensor and diagnostic at T <= 4K, random E and tau: run BOTH in fp64
                       and compare to 1e-9, then in fp32 to a RELATIVE 1e-5 [v4] (re-ordered fp32 sums
                       differ by up to 8e-6 absolute on Atil; a bare 1e-6 would fail for the wrong reason)
T12 tau -> 0 limit     tau_j = 1e-9 on random non-empty relations: entries == cbar_j/|E_.j|; != A_sm  [new]
T13 gradient           finite-difference check of d loss / d logits (ST), d tau_j, d tau_i on a
                       4x6 toy; and gradcheck of the custom sparsemax Function                     [new]
T14 target detector    on rows where the unit cap binds, A[i,E] == tau_i * proj_le(Atil[i,E], cbar_i/tau_i)
                       and != the same with a1 as the target (the two must differ on some row)     [v4]
T15 NaN sweep          a causal batch with n_q = n_k, a left-padded batch, a batch with |E_.j| = 0 columns
                       and cbar_i = 0 rows: loss and every parameter gradient finite                [v4]
T16 tau_i = 1 identity on random rows incl. cap-binding ones, tau_i = 1 gives A[i,E] == a1[i,E] exactly
                       (record §43); the Phase-B init sanity check is this at the model level       [v4.5]
T17 the unit cap     [v4.11] check_unit_cap.py is the preflight gate; T17 is its unit-test form.
                       (a) NO ROW BINDS WITHOUT A RELATION: force_empty at n_k in {8192, 16384} on the
                           DENSE, CHUNKED and DECODE paths -> excess exactly 0.0, cap_binds all False.
                           This is what T0 and sanity 5(a) rest on and it must hold bitwise.
                       (b) A BINDING ROW NEVER GAINS MASS AND NEVER LIFTS A ZERO: on razor-edge rows
                           (spike + flat tail at gap 16.64) with excess straddling CAP_TOL, assert
                           a1 <= Atil elementwise, no Atil == 0 entry positive in a1, cap_theta > 0,
                           and |sum_j a1 - sm_mass| small.  Build the row DIRECTLY too: sum(v) = 1 - 3e-5
                           with exact zeros, binds forced True -- the pre-v4.11 code adds 3.0e-5, gives
                           theta = -2.5e-6 and lifts 8 zeros.
                       (c) THE EXCESS IS EXACT: max |row_masses(...)[0] - (Atil.double()-A_sm.double())
                           summed over E| == 0.0, and record min |excess - CAP_TOL| observed -- that is
                           the real safety margin and it belongs in the run record.
                       (d) ⚠️ THE GUARD'S WINDOW, ON THE TARGET DEVICE: bisect the largest excess that
                           theta <= 0 suppresses and GATE it against CAP_TOL.  8.7e-8 on this CPU;
                           ~n_k*eps/4 ~ 2.4e-4 at 8K if the prefix scan is a true fp32 reduction.
                           NOT YET IMPLEMENTED, and it is the last open question on the mechanism.
                       (e) PATH AGREEMENT off the boundary: dense vs chunked binding sets equal; record
                           the excess difference (~2e-7) and the sm_mass difference (up to 6e-5 on razor
                           rows -- the two paths use different softmax formulas by design).  Do NOT
                           assert exact equality of cap_binds: a row within ~5e-7 of CAP_TOL may differ.
                       Coverage still owed: rho = 0.5, the decode path on a LIVE relation, head_block,
                       hard_concrete, batch > 1 with pad rows.
T18 two sites, one layer [v4.10] keep_dense_head = {l: [h1, h2]} at head_block in {None, 2, 4}:
                       assert A.shape[1] == 2 after the merge; assert every per-head field of a column
                       comes from the head its record names; assert _kept_head RAISES on a head that was
                       not kept; assert the sparse store's head indices are the GLOBAL ones.
                       Run it through the PUBLIC entry, not the per-block function -- the defect this
                       catches lives in the merge, after check_invariants has already passed per block.
T19 dense/sparse parity [v4.10] ONE configuration chosen to exercise the rare branches at once: a
                       zero-mass row, a column with |E_.j| > K_ret, two sites in one layer, two head
                       blocks, a ragged last block (H % head_block != 0).  Assert EVERY E8 key, EVERY
                       cache field (both truncation counters) and EVERY column record equal across the
                       dense and chunked paths.  This is the single test that would have caught every
                       instrumentation defect of the three review rounds (H16); it does not yet exist.
T7 note [v4]           runs with tau_j frozen (normalizer(tau_j_override=...)); with a live head it FAILS
                       and that failure is expected (prop:prefix's hypothesis).
```
T5 and T9 together are E1. If T9 shows variance across seeds for a dense arm, the support is
being thresholded somewhere in the pipeline and the whole comparison is invalid.

---

## 12. Instrumentation, ground truth, aggregation

### 12.1 Two evaluation modes, never mixed
**Teacher-forced** (all attention-level quantities): feed prompt + gold answer, run one forward
pass, read `A` and the diagnostics at the rows/columns of §12.2. **Generation** (all task-level
quantities): greedy decode from the prompt, parse, score. A number from one mode is never
compared with a number from the other.

### 12.2 Which row, which column
A **row** is `(layer l*, Q-head h*, query position i)` with `(l*, h*)` the top retrieval head
among the patched layers (§6.7). For evidence precision the query positions are the **answer
positions**: under teacher forcing, the positions that emit each gold value (RULER) or each gold
answer token (QA). A **column** is `(l*, h*, key position j)`. A *passage's* support is the
**union over its token keys** (`supp` of the row restricted to the passage's span is non-empty).
Report per-(l, h) too; never pool heads.

⚠️ **[v4.10] There are TWO sites, and they may share a layer** (D-9a): the coreference column is
scored at the head S0 licensed, the value column and the rows at the detector's `(l*, h*)`. Carry
them as `(layer, head)` pairs everywhere — resolution, retention, merge, reader, and the paired
relation cache — per §6.1a. A guard that rejects only the *exact pair* `[l*, h*]` does not make two
sites distinct: `(14, 5)` against a detector at `(14, 3)` passes it and then collapses.

### 12.3 Column ground truth `T_j` (audit-5 C2) — constructions and the routing check
The column theory (Thm. 1/2, E7's hit rate) needs, per key `j`, the set `T_j` of queries that
*should* attend to it. No benchmark annotates this. Two constructions on RULER, both **licensed
only if the routing check passes**:

```
C-a  value-key column.  On MV-NIAH (one key, m values): key j = the first token of needle value v;
     T_j = { the answer position(s) that emit value v }.   m_j = 1 typically (one emission per value).
C-b  question-key column.  key j = the last token of the question / the needle key phrase;
     T_j = { all m answer positions }.   m_j = m.   This is the column the m-sweep stresses.
C-c  (QA) supporting-passage column.  key j = first token of gold passage g; T_j = answer positions.
```
**Routing check (S0, on the Phase-A weights at 8K):** for each construction, the mass the `T_j`
positions put on key `j`, relative to the row's maximum over CANDIDATE keys. A construction is
*usable* at a head if that ratio exceeds 0.5 on ≥ 80 % of examples.
[v4.7, refined 2026-09-08 BEFORE any result was used — a preliminary 2K probe on the untrained base
model found C-a at 50 % at the retrieval head and the coreference construction unrouted at layer 19;
record §50.] (i) **Sweep every (layer, head)**, not the patched layers only: coreference is the
INDUCTION pattern and lives in other layers than the retrieval heads the detector selects.
(ii) **Spans, not single tokens**: key span = the first mention plus ONE following token (an
induction head attends to the token AFTER the antecedent); query = any token of the later mention;
ratio taken over the span. (iii) **Exclude the BOS/sink token from the row maximum**: it is not a
candidate key and in a base decoder it is usually the row's argmax. (iv) If a head routes the
coreference construction, the detector gains an INDUCTION score beside the retrieval score, that
head's layer joins `PATCHED_LAYERS`, and the column test runs there (E9's layer ablation covers the
change). (v) Only if no head at any layer routes it does the pre-registered fallback apply. S0 also checks the replication pair: at `(l*, h*)`, do two keys exist whose planted
`T_j` sizes differ (`M > m_j`)? If no construction passes, E7's column half is reported as
*not measurable on this backbone* and the paper's row-side fidelity carries E7.

⚠️ **[v4.10] S0 is a GATE, and the evaluation harness must enforce it.** `column_measurable`
defaults to `True` when no detector licence is present, so a queue run with S0 never executed would
report E7's column half as measurable **without a licence** — the exact failure §12.3 exists to
prevent. `run_evalsuite.sh` refuses to start if the detector lacks `s0_licence`. Note also that the
documented operator order (preflight → S0 → S2) cannot be followed as scripted: S0 needs the Phase-A
checkpoint that only the S2 script produces, and the S2 script has no way to stop after Phase A.
Give it `--phase_a_only` and say so in the error message.

**[v4.3] CONFIRMED by Luke, 2026-09-07 (decision D-9; record §46; paper App. H "E7, the column
ground truth"); [v4.9] SPLIT confirmed by Luke 2026-09-10 (D-9a; record §53) after the S0 sweep
showed answer positions route to VALUES, not to the key phrase's first mention.** Two column
kinds, both by exact string match, never merged into one target set:
```
COREFERENCE COLUMN (the column test: E7 hit rate, E2 stratified by m; run at the head S0 finds, e.g. (14,3))
key j     = the FIRST token of the FIRST mention of the bridge string
            RULER MV/MK-NIAH: the key phrase's first needle mention;  MuSiQue: the hop-h intermediate answer
T_j       = first tokens of every LATER MENTION of the same string: the other needles, the hop-(h+1) passage,
            the question, and the answer-prefix mention.  NOT the answer positions.
m_j       = |T_j|   (RULER MV-NIAH: m + 1;  MuSiQue leaf passage: 2;  bridging passage: 3-5)
VALUE COLUMN (where the ANSWER ROWS are scored; the m_j <= 1 regime, row-side by §12.3a)
key j     = a needle's value span (first token);  T_j = the answer positions emitting that value;  m_j = #emitting
            tokens, typically 1
K_i       (row ground truth, answer position i) = the gold needle VALUE spans (token level) / the gold
            needle sentences or gold passages (passage level)
delta_j   = min_{T_j ∩ E_.j} s - max_{E_.j \ T_j} s ;  if E_.j \ T_j is empty, delta_j = +inf and the lower
            endpoint is 0;  T_j \ E_.j is a RELATION-RECALL miss, logged separately, never charged to tau_j
```
**Dense arms have no relation**, so their precision is reported on TWO domains and both are in the
table: the full visible column (`m_j/n_vis`) and MarSea's paired relation `E_.j` from the same
example (`m_j/|E_.j|`, the paper's identity). The headline is the relation domain (def:fidelity);
the full-column number sits beside it.
⚠️ **[v4.10] The paired-relation cache is keyed by `(layer, head)`, not by layer** (H15). Keying it
by layer hands the coreference head's relation to the value columns and corrupts the A7 headline
precision of every dense baseline. It is a host-side cache and is sized by `n`: one `[T,T]` bool is
67 MB at 8K and **268 MB at 16K**, so the 16K jobs must cap `--n` or build it lazily.

### 12.3a The `m_j ∈ {0,1}` columns are row-side tests; log the Stage-1 split (decided 2026-09-03)

There is **no column abstention component** (considered and dropped: it would be a trained
binary decision, and the composition below already does the work). With one answer query `i*`:
at `m_j = 1`, `τ_j ≥ 1/δ_j` puts the whole quota on the argmax member (Thm. 2 with `W_j = 0`);
at `m_j = 0` no temperature empties a simplex, `P_j = 0` by definition, and the column is
**excluded from every column average** (count reported). For each answer row, split the
distractor keys three ways and log the shares:

```
excluded_by_stage1 :  argmax_i S[i, j] != i*   and  Atil[i*, j] == 0 exactly   (before any row program)
rejected_by_row    :  Atil[i*, j] > 0  and  A[i*, j] == 0                       (theta_i removed it)
surviving          :  A[i*, j] > 0
```
`excluded_by_stage1` is the Stage-1-attributable part of row precision and the quantity that
separates MarSea from B4/B5 on the row margin (they have no Stage 1). Harness reference under
random scores: 8.5 % of keys concentrate onto the answer row and are left to `θ_i`.

### 12.4 What is logged (parquet, one row per unit, keyed by run/step/layer/head)
```
per column (needs T_j):  m_j, delta_j (max over E_.j \ T_j — the RELATION's complement, never the full column),
                         W_j, interval [lo, hi) with hi = inf flagged, realized tau_j, hit (bool),
                         signed distance to the nearer endpoint, k*, nu_j, |E_.j|, relation recall |T_j ∩ E_.j|/m_j,
                         P_j, R_j, rho_j BEFORE and AFTER Stage 2,  site = [layer, GLOBAL head]  [v4.10]
per row:                 P_i, R_i, theta_i, tau_i, Rtil_i, cbar_i, tau_i*Rtil_i/cbar_i, cap_binds, |E_i.|,
                         relation recall |K_i ∩ E_i.|/|K_i|, |supp(A[i, E_i.])|,
                         distractor split (excluded_by_stage1 / rejected_by_row / surviving), §12.3a
per (layer, head, step): coverage rho (col, row), histograms of |E_.j| and |E_i.|, across-key var(tau_j),
                         across-query var(tau_i), k* histogram, fraction of rows over one unit after Stage 1,
                         frac_rows_zero_mass WITH its tolerance and ambiguous-band count [v4.10],
                         the unit-cap rule in force and the device's own softmax row-sum error [v4.11],
                         kstar_near_K_events and cache_truncations SEPARATELY [v4.10]
per example:             answer score, exact-set accuracy, support P/R/F1, joint metrics (HotpotQA), gold position
per row of every parquet [v4.10]:  arm, seed, set, experiment, checkpoint path, git hash
```
⚠️ Every fidelity quantity is computed **on the relation** (`def:fidelity` as revised). Off the
relation the entries are the unmodified layer's. ⚠️ Score the hit rate against the **sharp** lower
endpoint `1/(W_j+m_j*delta_j)`, never `1/(m_j*delta_j)`.

**Relation recall is logged separately**: a target the mask excluded is a recall miss *before any
program runs* and must never be charged to a temperature.

⚠️ **[v4.10] Provenance on the evaluation side.** The training side is exemplary (full RNG state,
cursor, git, config in every checkpoint); the evaluation side writes `{rule, table}` and nothing
else. A table that cannot be traced to a commit, a checkpoint and a seed cannot support a
reproducibility statement. Put `git`, `ckpt`, `set`, `seed`, `arm`, `spec_version` and the resolved
`arm_kwargs` in every `_table.json` and every parquet row.

⚠️ **[v4.10] Nothing aggregates across seeds, and nothing builds the E7/E8 tables.** `rec["e8"]`
is written per row and the aggregator never reads it, so the `|E_.j|`/`|E_i.|` distributions,
realised `k*`, `k*` stratified by `|E_.j|`, the `tau_i*Rtil_i/cbar_i` distribution and the
cap-binding fraction are in the parquet and in no table; E3's coverage rate `|E_.j|/n` is computed
and dropped. `scripts/collect.py` is owed: glob the tagged parquets, group by (arm, experiment,
stratum), emit mean ± sd over the three seeds (D-24) and the E8 blocks. It can be written while the
queue runs, but §5 of the paper cannot be written without it.

---

## 13. Experiments → code

### 13.1 Data, formats, prompt template [v4]

```
RULER      github.com/NVIDIA/RULER, scripts/data/synthetic/niah.py, driven by scripts/synthetic.yaml.
           Generate with the BACKBONE tokenizer, --max_seq_length in {4096, 8192, 16384}, seeds {0,1,2}.
           Task configs used (names as in synthetic.yaml; args: type_haystack, type_needle_k, type_needle_v,
           num_needle_k, num_needle_v, num_needle_q):
             S-NIAH   (E4)  : niah_single_2   essay / words / numbers,  1 / 1 / 1
             [v4.1] niah.py gives EVERY one of num_needle_k keys num_needle_v values (K*V needle sentences,
             ~15 tokens each), and the question asks about one key.  Grids restated for feasible budgets:
             E2 (m-sweep) : K = 8 keys, V = m in {1,2,4,8,16}, Q = 1, L = 8K   (<= 128 sentences ~ 2K tokens)
             E3 (n-sweep) : K = n in {8,16,32,64,128}, V = 4, Q = 1, L = 16K  (n=128: 512 sentences ~ 8K tokens;
                            L is CONSTANT across the sweep, which the control requires)
             E4 (m = 1)   : niah_single_2, K = V = Q = 1, L = 8K
           [v4.1] Needle DEPTH is sampled at random by stock niah.py; the position control needs a
           `--gold_depth` flag (a 10-line patch; ship it with the code and record the diff).
           Output jsonl fields: index, input, outputs (list of gold values), length, answer_prefix.
           [v4.1] niah.py STRIPS answer_prefix out of `input`; the prompt is `input + answer_prefix`, or the
           task changes.  Gold value positions for teacher forcing are found by exact string search of each
           output in the prompt (values are unique numbers/uuids by construction).
           "Inert padding" control (E3): the SAME max_seq_length with num_needle_k = 1 -- RULER's essay
           haystack fills the remaining length, so token count is matched and the distractors are gone.
           ⚠️ [v4.10] ONE such config is generated, not one per n.  With L constant the token count is
           already matched, so the spirit of the control holds -- but the paper's "every point runs twice"
           is not what the grid does.  Either extend the pad arm across n or reword App. H.
           "Gold at fixed relative position": needle depth is a generator parameter; hold it at 0.5 and,
           for the position-marginal profile, sweep {0.1, 0.3, 0.5, 0.7, 0.9} at one (m, n).
MuSiQue    github.com/StonyBrookNLP/musique release: musique_ans_v1.0_{train,dev}.jsonl.  Fields used:
           question, answer, paragraphs[{title, paragraph_text, is_supporting}], question_decomposition
           [{question, answer}] (the intermediate answers are the bridge strings for §12.3).  m = number of
           is_supporting paragraphs (2-4).  For E3-style n-sweeps, distractor paragraphs are sampled from
           OTHER questions' non-supporting paragraphs (same seed policy).
           ⚠️ [v4.10] This is a MANUAL download the data script does not verify; §9's hard assert covers it.
HotpotQA   HF dataset "hotpot_qa", config "distractor"; fields question, answer, context{title, sentences},
           supporting_facts{title, sent_id}.  10 paragraphs, 2 gold.  Sentence-level gold as annotated.
IHEval     github.com/ytyz1307zzh/IHEval (Zhang et al. 2025); the multi-turn rule-following split; the
           number of conflicting tiers is read from each example's metadata.
           ⚠️ [v4.10] NO ACQUISITION STEP EXISTS -- not in the data script, not in the setup script -- and
           the per-task checkers are not vendored, so E6 is unrunnable as it stands and the queue silently
           prints "E6 skipped".  E6 is the only experiment testing the RANKED half of the motivating scene.
           Acquire it or move it explicitly to "not run" in the paper (§17.1).
Prompt     base checkpoint, no chat template (IHEval [v4.1]: "System: {system}\n\nUser: {user}\n\nAssistant:"):
           "{context}\n\nQuestion: {question}\nAnswer:" -> loss/decode on " {answer}" ; RULER: its own
           template from the generator, unchanged.  Passage order in QA prompts is OURS to set (§12.3):
           gold passages in hop order, distractors interleaved at fixed slots by seed.
Eval       greedy, max_new_tokens 32 (RULER) / 16 (QA) / task default (IHEval); exact-set accuracy parses
           the generated list on ", " and compares as a set to outputs.
```

```
E1  v1_probe.py (in the package root; synthetic; no backbone).  Reproduce Fig. 2 before anything else.
S0  §12.3 routing check + replication pair on B0.  GATE for everything after S1.
E2  RULER MV-NIAH, m = num_needle_v in {1,2,4,8,16} at fixed K = 8 keys and fixed L = 8K (§13.1).
    Report: MarSea's own precision, exact-set accuracy, interval-hit rate STRATIFIED BY m.
    Falsification: hit rate falling with m.  The comparator's precision gap SHRINKS with m by arithmetic
    (1 - m/n); do NOT read that as failure.
E3  LOAD-BEARING.  Fixed m = 4; n in {8, 16, 32, 64, 128} candidates (RULER num_needle_k with V = 4;
    MuSiQue with sampled distractor passages, ~100 tokens each), L held at 16K by the generator's
    haystack fill / padding (§13.1) -- 16K needs the chunked path of §6.6 on the patched layers.
    Every point twice: real distractors vs the inert-padding control of §13.1; gold at depth 0.5; report
    the position-marginal profile from the depth sweep.  Report per n: fraction of active columns with delta_j > 0, hit rate on
    those, MarSea precision on those vs on all, comparator precision (== m/|E_.j|: an identity).
    Flat precision on delta>0 columns beside a falling fraction of such columns = the theory's own
    prediction; flat on ALL columns = the relation is doing the selection, not the program.
E4  E2 at m = 1 (S-NIAH).  Prediction: gap ~ 0.  A LARGE gap is the interesting negative.
E5  MuSiQue-Answerable (m in {2,3,4} by hop) + HotpotQA-distractor; support F1 beside answer F1;
    joint metrics on HotpotQA.  Corroborates E2/E3, never replaces them.
    ⚠️ [v4.10] E5 must be given the SAME paired relation as E2/E3 for the dense arms (§12.3), or RULER
    and QA report two different measurements of the headline quantity.
E6  IHEval multi-turn rule-following; accuracy vs number of conflicting tiers, as a SLOPE.  See the
    data note above: unrunnable until IHEval is acquired.
E7  All five def:fidelity quantities; recall twice; PR trace vs Fig. 3 by sweeping tau_j on held-out
    columns at eval; interval-hit rate (§12.4); coverage residual n_q - sum_j m_j.
    ⚠️ [v4.10] NO tau_j sweep on a trained checkpoint exists anywhere in the code.  The only trace rows
    are the harness's synthetic ones, which App. D itself says are "not evidence about real data".
    The paper claims this trace; write the sweep or cut the claim.
E8  §12.4 per-(layer, head, step) block from step 0; the E9 "tau_i pinned at 1" arm's row-recall.
E9  Ablations, each a separate arm under §9: TauK inputs (B3 and "no nu"); TauQ pinned at 1; tau_j
    fixed/global; E pairwise vs key-alone (u_phi only); relation rank r in {4,16,64}; rho_0; ST vs
    hard-concrete; L_patch in {2,4,8} and layer choice (retrieval vs uniform vs all@4K); K_ret;
    block size and emission policy; per-head vs shared heads; inherited quota vs a UNIFORM quota
    (this is "put a predicted budget back" and is reported as such).
    ⚠️ [v4.10] D-24's reduced programme runs three of these.  The paper claims E9 "ablates every design
    decision"; block size, layer set, the emission policy and the tau^K-tau^Q tying have no arm at all.
```

Pre-committed decision rules are in the paper (§5, App. H); the code prints them next to the
numbers they govern so a run's README cannot omit them.

---

## 14. Build order

1. §3 primitives + T1, T2, T13.  Nothing else works until `sorted_prefix_stat` is exact.
2. `marsea_normalize` dense at `E = vis` + T3, T6, T9, T10, T12, **T17**.
3. Relation head + init + ST (§4) + T4, T5; watch the coverage histogram from step 0.
4. Backbone swap (§6.1–6.3) + **T0** on Qwen2.5-1.5B at 2K with a PROVISIONAL layer set (T0 is
   invariant to which layers are patched); then B0 vs unpatched greedy decode; then T15.
5. Chunked implementation (§6.6) + T11, **T18, T19**; memory profile at 8K and 16K, **dense and
   chunked, teacher-forced and training, at one and two retained sites**.
6. Retrieval-head detection (§6.7) on the unpatched backbone; freeze `(l*, h*)` and `PATCHED_LAYERS`.
7. Causal sealing + hierarchical top-K + T7, T8; frozen-prefix decoding (§6.4).
8. Baselines B0, B1, B4 (cheap), then B3, then B2 (MESH; verify against its App. A), then B5 last.
9. Instrumentation (§12.4), RULER generation with the backbone tokenizer, **S0** (§12.3).
10. Phase A/B training (§9) on one seed at 4K; E8 diagnostics; then the S2 grid.

S0 is the gate on everything: it measures whether a replication pair `(a, M)` with `M > m_j`
exists on trained-backbone score columns at `(l*, h*)`, and whether any `T_j` construction is
usable. If neither, `cor:capacity` has no bite on real data and the programme stops there.

---

## 15. Decisions Claude Code must NOT take alone (ask Luke)

- Changing the backbone or the patched-layer policy of §6.7 after S0 has run.
- Any auxiliary loss on the relation (§9) — off by default; turning it on is a paper-level change.
- Replacing the ST estimator with a soft forward pass (it would remove the exact zeros).
- Any change to the four programs' order of operations, the cap in fan-in step 2, or the
  initialisation through the relation rather than through `tau`.
- ⚠️ **[v4.11] Any change to the unit cap's BINDING CRITERION or its PROJECTION TARGET.** The
  criterion is `excess > CAP_TOL` on the relation's own mass and the target is `sm_mass`; both sit in
  the forward pass, both define two reported E8 keys, and together they fix what
  `prop:twostep`(iii) can say. A change to either is a paper change (§17.1) and makes prior E8
  numbers incomparable. In particular **do not reintroduce a comparison of a row total against a
  constant, at any tolerance** — that is the dead end §0's v4.11 note records, and it cost four
  commits. Propose, measure on the target device with `check_unit_cap.py`, and record the rule in
  the run README.
- ⚠️ **[v4.10] Cutting or reducing a pre-registered arm** (D-24's three E9 ablations, the three
  seeds, the inert-padding control) to fit a schedule. Cuts to *controls* are Luke's call too, and
  they change what App. H can claim.

---

## 16. Reference implementation from the cold read [v4]

`reference/marsea_cold.py` and `reference/test_cold.py` (package) are the second engineer's
implementation of v3 **from the text alone**, CPU PyTorch, `[B,H,n_q,n_k]`, GQA, ragged relations
by padding, straight-through relation, MLP heads, and all six baselines. `test_cold.py` passes every
INV and T0/T1/T13 with 0 failures, and `compare_harness.py` matches `verify_all.py`'s Prop. A loop
to 3e-15 in fp64. It embodies v3-era guesses that v4 has since fixed in the text; before reuse, bring
it to v4 on: B2 (the MESH step is straight-through, `eps = 1`, ends on the row scaling — its v3
version differentiates through the descent and uses `eps = 0.1·std`), the head-input normalisation
of §5.2, the detector, and §9's no-packing rule. It is a starting point and a test oracle, not the
deliverable.

---

## 17. Decisions that were Luke's — all closed (2026-09-08) [v4.6]

The round-2 engineer judged the pseudo-code sufficient, without questions, for the primitives,
the normaliser, both heads, the backbone patch, all seven arms, the training loop, the data
pipeline and T0–T16 in teacher-forced mode, subject to five decisions. Luke closed them:

1. **`T_j` construction** — CONFIRMED 2026-09-07 (D-9), SPLIT 2026-09-10 (D-9a): §12.3.
2. **E2/E3 grids** — CONFIRMED 2026-09-08 (D-27): E2 at K = 8 / L = 8K; E3 at L = 16K constant
   across the sweep, which commits S2 to the chunked path (§6.6). §13.1 as written.
3. **Decode-time procedure** — CONFIRMED 2026-09-08 (D-28): §6.4 as written (frozen-prefix,
   per-key re-solve at sealed `tau_j`, incremental `cbar_j`, only row `t` emitted); MarSea's
   task-level numbers and their cost come from this path.
4. **B5 teacher-forced only** — CONFIRMED 2026-09-08 (D-29), "for now": attention-level fidelity
   only; the task-level cell reads "n/a by design". Revisitable after S2 if a task-level B5 is wanted.
5. **`transformers` pin** — CONFIRMED 2026-09-08 (D-30): `>= 4.53`; pin the exact release in
   `requirements.txt` at implementation time and assert the `eager_attention_forward` signature at import.

### 17.1 ⚠️ Reopened by v4.10 — one paper item remains [v4.12]

The v4.6 statement "nothing in this contract now waits on a decision" no longer holds. Item 1 is
**done as of 2026-09-15 (record §60)**; item 2 is still Luke's.

1. ✅ **APPLIED 2026-09-15.** **`prop:twostep`(iii) is restated, and the restatement is a strengthening.** It becomes
   `Σ_j A_ij ≤ Σ_j A^sm_ij` — *the mechanism never gives a row more than the unmodified layer gave
   it* — which is **one unit in exact arithmetic and true on any kernel**, where the old absolute
   wording was falsifiable in ten lines of code and the intermediate `≤ 1 + δ(n_k)` form was
   vacuous (the cap imposes it). The content is the **converse**: *Stage 1 is the identity on every
   row the relation did not push over its own softmax mass.* **Eight exact-match patches are drafted
   in `claude/marsea_paper_edits_cap.md`** — the statement, a new remark giving the finite-precision
   form and the `n_k·eps/(2L)` kernel law, §2.2, the proof, `tab:verification`, App. D, App. E, and
   a paragraph in App. F that reports the kernel's own δ as a **device statistic beside the checks
   rather than inside them**. All eight were applied by `patch_iclr_cap.py` (every `old` string
   matched exactly once), and `verify_all.py`'s `Prop. A(iii)` label and target changed with the
   table, so the reproducibility statement stays true. Build after: 42 pp, 0 err / 0 undef /
   0 multiply-def / 0 overfull, every edit checked in the rendered PDF.
   ⚠️ **Nothing needs re-running.** The regression the change repaired was latent on the kernels
   measured — `torch.softmax` there always gives `sm_mass − 1 > 0` — so old and new produce identical
   binding sets and `A` within 1.19e-7. This is device-safety, not a numbers correction.
2. **Claims with no arm behind them** (record §58.6): the Llama cross-family check, "we
   ablate both emission policies", the hard-variant ablation, the τᴷ–τQ tying, E9 over layer set
   and block size, E6/IHEval, E7's measured PR trace against Fig. 1, efficiency versus measured
   coverage, padding-length invariance, "2–16K", the E3 padding control's "every point", and B1/B4
   appearing in a headline table with a standard deviation. Each is a sentence promising a
   measurement; cut, qualify or fund it. **Not edited unilaterally** (D-2).

---

## 18. Compute plan (decided 2026-09-05; record §44)

```
provider     Nebius AI Cloud (on-demand H100, $3.85/GPU-h list; preemptible $2.15 for E9 reruns only)
fallback     RunPod Secure Cloud, H100 SXM (NOT PCIe: the patched layers are memory-bound) -- same image, same tarball
region       one European region (Finland or Paris) for BOTH VMs and the shared filesystem; request the 8xH100 quota on day 1
stage 1      1-2 x H100 VM, ~4 days: build image (pinned transformers >= 4.53, peft), T0-T19, detector (§6.7), S0 routing
             check (§12.3), S1 (end-to-end at 4K, step time at 8K -> rule: > 1.2 s/seq => 2 patched layers or 4K training),
             RULER generation at 8K/16K + tokenisation (CPU), Phase A once per seed.  Stop the VM when idle.
stage 2      1 x 8xH100 VM, ~2 days wall: 17 training runs (4 decisive arms x 3 seeds + B1, B4 x 1 seed + 3 E9 ablations),
             then the evaluation queue (§19).  Eight independent single-GPU processes (CUDA_VISIBLE_DEVICES per
             process); no distributed training.
storage      shared filesystem ~500 GB (RULER sets, tokenised QA, detector scores, Phase-A checkpoint, run checkpoints,
             parquet logs); checkpoint LoRA + heads every 250 steps; every run idempotent on restart; results written per
             (arm, seed, level) so a restart never recomputes; daily rsync off-platform (egress free).
budget       reduced programme ~380-500 GPU-h ~ $1,500-1,950; full ~700-800 GPU-h ~ $3,000-3,500.
```

⚠️ **[v4.10] The queue as scripted implies ≈1,000 H100-h against `tab:compute`'s ≈336** — a 3×
overrun and ~6.5 days of flawless execution against the 25 Sep deadline. Two cuts restore both
(§19): **E3depth to `marsea` + `B0` seed 0 only** (a *control* currently costing as much as the
load-bearing experiment, six times over — ≈150 h) and **`--n 400` on the three 16K jobs** (≈250 h,
and the paired-relation host cache forces it anyway). That lands at ≈610 h and ~4.5 days. Both are
cuts to pre-registered breadth and are therefore Luke's (§15).

---

## 19. The evaluation queue [v4.10]

The training side of the harness is sound; the evaluation queue as written **cannot produce the
paper's tables**. The contract it must meet:

1. ⚠️⚠️ **Concurrency is bounded by the number of GPUs, per launch — not per outer-loop
   iteration.** The barrier belongs *inside* the launch function, tested against the in-flight PID
   count, exactly as the training queue does it. A cadence keyed on a counter incremented once per
   arm breaks the moment the number of jobs per arm stops dividing the GPU count: at seven jobs per
   arm and eight GPUs it fires **once in the entire double loop**, with 56 processes in flight.
2. ⚠️⚠️ **An output path identifies the job that wrote it.** The stem must carry arm, kind,
   experiment, **set and seed**; every row must carry `seed`, `arm`, `set`, `ckpt` and the git hash.
   Without this, three E3-family jobs × three seeds collapse onto one file, all seven E9 ablations
   collapse onto one table, and the `.done` markers — which *do* carry the seed — make a restart
   report success. The failure is silent and is discovered only after the GPU-hours are spent.
3. **Every job's memory configuration is one preflight has measured.** `--head_block` must be
   settable on the evaluation entry points (it is not, so evaluation runs all heads while preflight
   gates at `head_block=2`); the dense teacher-forced 8K pass, which five of the seven job types
   default to, must be profiled; and a pre-registered arm may never sit behind a `|| echo`.
4. **A failed job is observed and propagates.** Per-PID `wait`, a named failure list, a non-zero
   exit. (This part is now correct in both queues.)
5. **The evaluation configuration matches the arm's training configuration**, or the mismatch is
   deliberate and stated. Note that `truncation_events` is mode-dependent, so E2 (dense) and E3
   (chunked) E8 blocks are not comparable even with the prefill parity of §6.4 fixed.
6. **The S0 licence gates the queue** (§12.3).
7. **A collector exists** (§12.4).
