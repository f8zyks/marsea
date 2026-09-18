# Cold-implementation ambiguity log — MarSea spec v3 + paper §2

Format: **#N [severity]** — quote — problem — what I assumed. Severity = how likely a literal
engineer writes WRONG code (H = silently wrong numbers or NaN in a real run; M = wrong on an
edge case or a test would catch it; L = under-specified but any reasonable choice is equivalent).

## A. Programs (§3–§5, paper eq:step1–eq:rowprog)

**#1 [H] `column_stats` divides by a zero std on every causal batch.**
Spec §5.2: `column_stats(S, vis)[j] = (max, second max, mean, std, max - second max, (max - mean)/std) over visible i`.
In square causal self-attention the last key is visible to exactly one query, so `std = 0`,
`(max-mean)/std = 0/0 = NaN`, and `second max` does not exist. The NaN enters `TauK`, so `tau_j`
is NaN for that key, `p` is NaN, `Atil` is NaN on the whole column and the loss is NaN. Not
mentioned in §10. *Assumed*: biased std over visible entries, clamped at 1e-6 for the ratio;
`second max := max` (gap 0) when fewer than 2 visible; all six stats 0 for an invisible column.

**#2 [H] `proj_le` at `s = cbar_i / tau_i = 0` — the common path, and a NaN in the backward.**
§3: `proj_le ... RETURN w IF sum(w) <= s ELSE proj_eq(v, s)`; §5: `u = ragged_proj_le(Atil restricted to E_i., cbar_i / tau_i)`;
H6: "Empty relation (|E_.j| = 0 or cbar_i = 0): skip step 2 ... With g = 0 the interpolation is the identity anyway."
H6 is wrong on the second clause: at `cbar_i = 0` with `|E_i.| > 0`, `g = 1` on the relation, so the
interpolation is *not* the identity — `u` must be defined. Measured: **65 % of rows with a non-empty
relation have `cbar_i = 0` exactly** (random scores, `tau_j ∈ [0.3, 4]`), because Stage 1 zeroes most
relation entries. One can prove `cbar_i = 0 ⇒ Atil_E = 0` (the cap's survivors would otherwise all be
off-relation entries whose `A_sm` sum is ≤ 1, so the cap could not bind), so the *forward* never
divides by zero in a scalar loop — but any batched `torch.where(binds, s*sparsemax(v/s), w)`
evaluates `v/0` in the dead branch and the backward is **NaN on every such row** (verified: naive
form → `grad = [nan, nan, nan]`; guarded form → `[0, 0, 0]`). *Assumed*: `proj_le(v, 0) := 0` with a
safe divisor; `A_E = a1_E = 0`. The spec must say this; a literal reading NaNs step 1 of training.

**#3 [H] `nu` for a column with an empty relation is `1/eps`, not in `[1, |E_.j|]`.**
§5: `nu = 1 / (p*p).sum(-2).clamp_min(eps)`; paper: `nu_j ∈ [1, |E_.j|]`. With `|E_.j| = 0`,
`p = 0` and `nu = 1/eps` (1e6 or so); `log nu_prev` = ~14 is then fed to the next layer's `TauK`.
`eps` is never given a value. *Assumed*: `nu := 1` on empty columns (the value §5.3 uses for
"no information"), `eps = 1e-12` elsewhere.

**#4 [M] Fan-in step 2 re-shapes `Atil` (pre-cap), while its quota comes from `a1` (post-cap).**
Paper eq:rowprog: `argmin ||a/tau_i - Atil_{i,E}||`, `cbar_i = sum_E a1`. Spec §5: same. But
spec INV-4 and H6 ("step-1 values stand") make `a1` the natural target, and on rows where the cap
binds the two differ. I followed the paper (`Atil`). A reader who uses `a1` gets a different `A` on
every cap-binding row and no invariant catches it (INV-1..13 all still hold).

**#5 [M] `sorted_prefix_stat` mixes 1-based and 0-based indexing.**
§3: `k = arange(1, len(z)+1)`, `kstar = max{k : G[k] < 1}`, `psi = (cs[kstar] - 1) / kstar`.
With `k` 1-based, `cs[kstar]` is 0-based-off-by-one (should be `cs[kstar-1]`). *Assumed* the
harness's convention: `psi = (cs_{kstar-th sorted entry} - 1)/kstar`.

**#6 [M] `vis` is declared `[n_q, n_k]` and "causal: vis[i,j] = (i >= j)".**
That formula is only right when `n_q = n_k`. With a KV cache (`n_q < n_k`, §6.4) the visible set
is `j <= i + (n_k - n_q)`. And §6.1 derives `vis` from the HF 4-D mask, which carries per-batch
padding, so `vis` is really `[B,1,n_q,n_k]`. *Assumed*: `vis` broadcastable to `[B,H,n_q,n_k]`;
tests use the offset-causal form plus random extra masking; every query row keeps ≥ 1 visible
key (a fully masked row makes `row_softmax` NaN; the spec is silent).

**#7 [M] The straight-through gate off visibility.**
§4.3: `g = E.float() + (sigmoid(logits/T_st) - sigmoid(logits/T_st).detach())`. Off `vis`,
`E = False` but the sigmoid term is live, so `logits` at invisible pairs receive gradient
(zero in practice because `A_sm = p = a1 = 0` there, but only by accident of the formula).
*Assumed*: multiply the soft term by `vis`. INV-8 "absent from every program" then holds for the
backward pass too.

**#8 [M] `tau_j` initialisation "≈ 1/std(column)" cannot be a bias.**
§5.2: "Init the last layer's bias so that tau_j ≈ 1/std(column) at step 0". `1/std` is
per-column; a bias is one number. Also unspecified: the last layer's *weight* (a random weight of
the usual scale makes the output vary by O(1) around the bias, so no bias achieves "≈").
*Assumed*: last-layer weight ~ N(0, 0.01²); a `calibrate(S, vis)` helper sets the bias to
`softplus⁻¹(median_j 1/std_j − tau_min)` on a warm-up batch; default bias assumes std = 1.

**#9 [M] `row_summary`: "max_j Atil_ij" over what, and "entropy of the row's relation part" of what.**
§5.2. Max over all visible j or over `E_i.`? Entropy of `Atil[i, E_i.]` unnormalised or
normalised by `Rtil_i`? *Assumed*: max over visible j; Shannon entropy of `Atil[i,E_i.]/Rtil_i`,
0 when `Rtil_i = 0`.

**#10 [M] Inputs to `TauK`/`TauQ` are unnormalised and of wildly different scale.**
`k_j` (post-RoPE key, norm ~ tens), `max`, `mean` (score scale), `std`, the z-score, `log nu`.
No LayerNorm/standardisation is mentioned. *Assumed*: raw concat, as written. Likely to train
badly; not a correctness bug.

**#11 [M] Which key does `U_phi` / `TauK` read: pre- or post-RoPE?**
§1.2 "RoPE is inside query/key already" and §5 passes `K_kv` = the `key` argument of
`eager_attention_forward`, i.e. post-RoPE. Paper says `k_j`. *Assumed*: post-RoPE (what the
patch point hands us). Affects the relation's translation-invariance; not stated anywhere.

**#12 [L] `U_phi, V_phi ~ N(0, 1/d_head)` — variance or std?**
*Assumed*: variance `1/d_head` (std `1/sqrt(d)`), which is what gives "unit-scale noise about b0"
for unit-scale `k, q`. Bias on `U_phi`, `V_phi`: assumed none (paper's `e_ij` has none; `b0` is the
only bias). `b0` assumed a trainable scalar per layer after calibration (the task asks for its grad).

**#13 [L] `implied_threshold(v, u)` argument order/shape and what `Diagnostics` holds.**
`psi_j` is not in the Diagnostics list but INV-9 is a statement about `psi_j`; `|supp(A[i,E_i.])|`
(needed by B5) is not there either. *Assumed*: added `psi_j`, `supp_rel_i`, `p`, `a1`, `Atil`.

**#14 [L] `entmax(z, alpha != 2)` bisection: bounds and final renormalisation unspecified.**
*Assumed* the harness's bounds `[(α−1)max z − 1, (α−1)max z]`, 50 iterations, `p /= p.sum()` at
the end (so B4 at α=1.5 sums to 1 only approximately; exact zeros depend on the tolerance).

**#15 [L] `T_st`, `eps`, `tol` values.** `T_st = 1.0` given; `eps` in `nu` not given (1e-12 assumed);
`tol` in INV-3 not given (1e-6 assumed).

**#16 [M] INV-9 is false for the dense normaliser as written, unless `tau_j` is frozen.**
`tau_j = TauK(k_j, column_stats(S, vis), nu_prev)`; `column_stats` changes as the visible prefix
grows, so `psi_j^(t)` can *decrease* — measured: **3310 decreases in 6686 (column, t) pairs** with the
live head at init, 0 with `tau_j` frozen. §7 says to seal `tau_j` at block boundaries, but §2 lists
INV-9 as a runtime invariant of the normaliser. *Assumed*: INV-9 is tested with `tau_j`
overridden (frozen); the code exposes `tau_j_override` for this and for the E9 "tau_j fixed" arm.

**#17 [M] `-inf` vs `finfo.min` in `S` inside `column_stats` and the ST gate.**
H10 says never sum `S` over masked entries, but §5.2's `column_stats(S32, vis)` and
§5's `tau_j * S32 restricted to E_.j` are written on `S32` that still holds `-inf`; `0 * -inf = NaN`
if implemented as `S*vis`. *Assumed*: `masked_fill(~vis, 0)` before any multiplication.

## B. Relation head (§4)

**#18 [M] Paper vs spec on the relation's form.** Paper §2.3: `e_ij = 1[<u_φ(k_j), v_φ(q_i)> > 0]`
(no bias, no `1/sqrt(r)`); spec §4.1 adds `/sqrt(r) + b0`. Spec wins per §0 ("shapes"), but the
paper says "which reads the pair" — a shared additive bias is not pairwise. Followed the spec.

**#19 [L] `b0` calibration "on one warm-up batch": before or after Phase A; frozen or trained afterwards.**
*Assumed*: calibrated once at the start of Phase B on the Phase-A weights, trainable afterwards.

## C. Baselines (§8)

**#20 [H] B2 (MESH) is not implementable from the spec, and its stated property is false.** Undefined: the paper "MESH" (not in the
provided bib); `H(...)` (entropy of the plan? of its rows?); the sign (`C - 0.1*grad H` *decreases*
plan entropy); whether the gradient is through the unrolled Sinkhorn or a detached plan; whether
the 4 descent steps are in the training graph; `eps = 0.1*std(S over vis)` per (b,h) slice or per
batch ("GLOBAL"); "visible keys" for `a` (a key is visible if some query sees it? per Q-head?);
`h_a` reads `K_kv` so `a` is per KV-head — repeated to Q-heads?; **feasibility**: under a causal
mask the marginal problem `(a, b)` is generically infeasible (the last key is seen by one query
whose row must total `b_i`), so 30 iterations do not converge and "row sums = b_i" is false.
*Assumed*: plan entropy `-Σ P log P`, gradient through the unrolled 30-step log-domain Sinkhorn
with `create_graph` when training, eps per (b,h), key visible iff any query sees it, `a` repeated
across the GQA group, output = the plan after the last Sinkhorn (row sums reported, not enforced).
Measured (n = 12, 30 iterations, eps = 0.1·std): `max|rowsum − b_i|` = 0.10 under full visibility
(not converged — eps is too small for 30 iterations) and 0.46 under a causal mask (infeasible).
"Row sums = b_i" as written in §8 is not what the code produces under either mask.

**#21 [M] B5: what is recorded and how "renormalised".** §8: record `|supp(A[i,E_i.])|`; keep "that
many top entries within E_i. (renormalised), full softmax off it". `E_i.` is MarSea's relation and
must *also* be recorded (not stated). Renormalised to the relation's softmax mass (row still sums
to 1) or to 1 over the kept entries? What if the count is 0 (relation zeroed by `theta_i`)?
*Assumed*: record `E` too; kept entries scaled to carry `Σ_{E_i} A_sm`; count 0 ⇒ relation
entries 0 (row sums < 1, as MarSea's did). Also unspecified: which LoRA weights B5 evaluates with
(it is "evaluation-time only", so it has no Phase B of its own) — assumed MarSea's.

**#22 [M] B1: "exp(s_ij)/(1 + sum exp)" — the `+1` sits at score 0 *after* scaling; with `finfo.min`
masking, the sum must exclude masked entries.** *Assumed*: `exp(S - logsumexp([S_vis, 0]))`.

**#23 [L] B4: no temperature; `alpha` fixed; gradient through bisection at α=1.5 is the unrolled
iteration's (not the exact Jacobian).**

**#24 [L] B0 vs INV-11:** B0 is `row_softmax(S)`; MarSea with `E = ∅` is `proj_le_rows(A_sm, 1)` =
`A_sm` only if `A_sm` sums to ≤ 1 + fp error; a row summing to `1 + 3e-8` in fp32 triggers the
`proj_eq` branch and moves entries by ~1e-8. Fine at 1e-6, but "to fp32 precision" is not
"bit-identical" (T0 asks for tokens exactly, which is fine, and logits to 1e-5).

## D. Training recipe (§9) and experiments → code (§13)

**#25 [M] Phase A "E forced empty (every arm)"** — B1/B2/B4 have no `E`; assumed Phase A = softmax
for every arm (it says `== plain LoRA`). Whether the cosine schedule spans A+B or restarts at B:
unspecified (assumed one schedule over 3500 steps). "packed to L": whether packed examples are
attention-isolated (block-diagonal mask) or not: unspecified — this changes `vis` and hence every
column program (a key in example 1 must not be in example 2's relation). Assumed isolated.

**#26 [M] What LoRA targets in patched layers:** LoRA on `q_proj,k_proj` changes `S` and `K_kv`,
`Q` that the heads read — so the heads' inputs drift during training; fine, but "new modules LR
1e-3" is on parameters whose input scale is not fixed. Also LoRA is applied to "ALL layers" while
only 4 are patched — stated, OK.

**#27 [H] §13 is not runnable:** no dataset loaders/paths (RULER config, MuSiQue/HotpotQA
splits, IHEval source); E2's "fixed n" value; E3's "inert padding" token; "L held at 8K by
padding" (left/right, which token, masked how); `v1_probe.py` is not in the cold directory;
the retrieval-head detector needs a NIAH prompt set, count unspecified; `tab:compute`
referenced but not given; no eval prompt format for base checkpoints.

**#28 [L] §6.7 patched-layer policy depends on a detector run — so `PATCHED_LAYERS` is not a
constant an engineer can write down; the build order needs step 6 before step 4's T0 arm.**

## E. Findings from the tests and the harness comparison

**#29 [M] The harness's Prop. A block is the only place the `cbar_i ≤ TOL` rule is stated** (`if ci > TOL:`
with `TOL = 1e-9`, else the row keeps `a1`). The spec's H6 says `cbar_i = 0`; the harness says `≤ 1e-9`.
Between 0 and 1e-9 the two disagree by ≤ 1e-9 in `A` — harmless, but the spec should adopt one.

**#30 [L] fp32 vs the fp64 reference differ by up to 8e-6 in `Atil`/`A`** (softmax sums over 45 entries
then sort/cumsum). T11's "chunked == dense to 1e-6" is therefore at the edge of fp32 if the chunked
path sums in a different order; the tolerance should be stated relative, or the comparison run in fp64.

**#31 [M] INV-2 and `row_softmax` need "every query row has ≥ 1 visible key".** With HF left-padding
a padded query row is fully masked; the spec never says what `A` is on such rows (0 assumed).
