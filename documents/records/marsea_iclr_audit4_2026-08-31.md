# MarSea — fourth adversarial audit of `marsea_iclr.tex`

*2026-08-31. Draft audited: 34 pp, build clean, 456,621 checks at 0 violations.*
*Method: full read; two independent verification passes (literature against primary sources,
every number against the shipped scripts); each finding re-checked by hand before listing.*

**19 defects. 4 are Tier A.** The most serious is not a citation error but a **novelty threat**:
the paper's central differentiator from the transport line is stated in a form that the closest
prior work contradicts.

---

## Tier A — must fix before submission

### A1 ⭐ MESH already learns a non-uniform, content-predicted marginal. The paper says it does not.

**Paper, App. C (L964) and Sec. 1 (L157–161):**
> "MESH improves the construction by changing the transport *cost*; the marginals are what we
> change, and they are what both leave fixed."
> "*Transport-based attention* … and Slot Attention … identify the marginals as the design surface
> and fix the column marginal to *uniform*, equalizing fan-out at $n_q/n_k$ rather than bounding
> it by a content-dependent constant."

**MESH, Appendix A ("Marginals in Sinkhorn and EMD"), verbatim:**
> "We learn $a = m \cdot \mathrm{softmax}(h(Z))$ with a neural network $h:\mathbb{R}^c\to\mathbb{R}$
> shared across the $n$ input elements… Our $a$ allows the model to put focus on important input
> elements (e.g. the inputs corresponding to objects) and ignore unimportant input elements."
> "Since the softmax sums to one, we have $\sum_i a_i = \sum_j b_j = m$."
> A variant learns **both**: "We learn both $a = m\cdot\mathrm{softmax}(h_a(Z))$ and
> $b = m\cdot\mathrm{softmax}(h_b(X))$."

In MarSea's own orientation (slot = query, input = key, $m$ = slot count = $n_q$) that is:
**a learned, non-uniform, network-predicted key-side marginal, normalized so the budgets sum to
$n_q$.** That is the object §4.1 introduces as $c_j$ with $\sum_j c_j = n_q$, and the object
Fig. 2's "paper's own null" is the uniform version of.

⚠️ This is a **positioning problem, not a wording problem**. Two real distinctions survive and
must now carry the weight the false one was carrying:
1. **The field argument.** MESH's $h$ reads the input element **alone**; the marginal is coupled
   across elements only through the softmax normalization. MarSea's $c_\phi$ reads the key
   **together with $\nu_j = \|p_{\cdot j}\|_2^{-2}$**, a statistic of the realized column. This is
   the paper's own criterion ("no parameter of the mechanism is a function of a key or query
   alone") and MESH fails it. Fig. 2(b) is already the evidence for why the field argument
   matters.
2. **Equality inside a transport vs. a cap outside one.** MESH's marginals are equality
   constraints in a doubly-stochastic problem, so the row marginal is fixed too and
   Prop. `conservation` applies; MarSea's row side is an *inequality* cap and never renormalizes.
   The "not a duality, a matched total" paragraph (L239–245) is where this already lives.

**Verified**: arXiv 2301.13197 App. A; present in the ICML 2023 version.

### A2 ⚠️ Four bibliography entries have author lists that are largely wrong

| key | `.bbl` says | actually |
|---|---|---|
| `hou2024relation` | Xiaocheng Hou, Mingsheng Liu, Sheng Zhang, Pengfei Wei, Bo Chen, Xuguang Lan | **Xiuquan** Hou, **Meiqin** Liu, **Senlin** Zhang, **Ping** Wei, **Badong** Chen, Xuguang Lan |
| `du2025context` | Yufeng Du, Minyang Huang, Bingxuan Peng, Preslav Nakov, Muhao Chen | Yufeng Du, **Minyang Tian, Srikanth Ronanki, Subendhu Rongali, …, Hao Peng** (four listed coauthors are not on the paper) |
| `zhang2025iheval` | …Wenju Xu, Chao Zhang, Tuo Zhao | **Zhihan Zhang, Shiyang Li, …, Meng Jiang** (three listed authors are not on the paper) |
| `herasimchyk2026residual` | Hleb Herasimchyk, Roman Labryga, Tamerlan Prusina, Sören Laue | **Hanna** Herasimchyk, **Robin** Labryga, **Tomislav** Prusina, Sören Laue |

Four entries with invented coauthors in a 39-item bibliography is a credibility problem out of
all proportion to the effort of fixing it. **Every remaining entry was checked and is correct.**

### A3 $\mathrm{softmax}_1$ is misattributed

Paper (L276): "$\softmaxop_1$ \citep{xiao2024efficient}". It is **Evan Miller, "Attention Is Off
By One" (2023)**. StreamingLLM cites it as prior work and explicitly *rejects* it in favour of a
learnable sink (their Sec. 3, Table 3). Keep `xiao2024efficient` for the attention-sink claims;
add Miller for $\mathrm{softmax}_1$.

### A4 Two mechanism claims about others' work are wrong

- **`tay2020sparse` does not fix a column marginal on the attention matrix.** Sparse Sinkhorn
  Attention applies Sinkhorn to a **block sorting matrix**; attention is then ordinary row-softmax
  over reordered blocks. It does not belong in the "column marginal fixed to uniform" group.
- **Slot Attention's slots are not "a fixed bank of learned slots"** (L960–963), so
  "what competes is parameters and not data" is false. Slots are **sampled from a shared learned
  Gaussian on every forward pass**, deliberately, so that they do not specialize; and from
  iteration 2 they are functions of the data. *(The orientation fix from the last pass is correct
  and was re-verified — that half stands.)*

---

## Tier B — the paper's own shipped artifacts contradict it

### B1 ⭐ "No block rejects" is false, and the paper's own denominators show it

Protocol (L2059): *"No block conditions on a hypothesis a draw may fail, so every count below is a
count of trials actually run."* Caption of `tab:verification`: *"No block rejects."*
Reproducibility statement: *"…now sample the interval directly and run every trial they attempt."*

`verify_all.py` has **three `if W > 0:` guards** (L114, L391, L535). $W_j = 0$ whenever $m_j = 1$,
and the skipped trials are visible in the paper's own table:

| row | denominator | attempted |
|---|---|---|
| Thm. 2, sharpness above upper endpoint | **10,098** | 12,000 |
| Thm. 2 on a scope, above upper endpoint | **10,604** | 12,000 |
| Fig. 2, trace above the interval | **20,472** | 24,000 |

There are also **two literal `while n_acc < 3000: … continue` rejection loops** (L147, L328).
They reject nothing on these seeds, so the *counts* are honest — but the structure the paper says
was removed is still in the shipped code, and the guarantee is empirical rather than structural.

The defensible version is true and nearly as strong: *the upper-endpoint test has no content at
$W_j = 0$, where $1/W_j = \infty$, so those columns are not rejected but out of scope; every other
block runs every trial it attempts.*

### B2 ⚠️ `tab:margin`'s caption is contradicted by the table it captions

Caption: *"…whereas from $n_q=128$ Assumption 2 fails and **no** temperature recovers."*
The same rows report recovery of **14.9%** at $n_q=128$ and **1.2%** at $512$. And *"at
$n_q\le32$ the column margin is still positive"* is a statement about the **mean**: at $n_q=32$,
**795/3000 (26.5%)** of columns already have $\delta_j\le0$, which is most of the 36.4% shortfall
the caption attributes to a mis-set temperature.

### B3 A number in the paper the shipped script contradicts

App. H (L2199): *"$0.133$ at a non-target score spread of $1.0$, $0.163$ at $0.3$ and $0.165$ at
$0.2$."* Re-running `v1_probe.py` at each spread: **0.165 ✓, 0.163 ✓, 0.135 ✗**. The script's own
comment repeats the wrong value.

### B4 The E1 slope list has four entries; the figure has five curves

Main text (L709–711) lists softmax, gate, Sinkformer, MarSea. Two sentences later the text says the
uniform-budget null lies on the Sinkformer line — and `v1_probe.py` prints and plots it. The TMLR
version lists all five. A reviewer counting legend entries against the prose will notice.

### B5 "Sinkformer sits at $n_q/n_k$ exactly" overstates twice

- Sinkformers use **3–5 Sinkhorn iterations**, so the matrix is only approximately doubly
  stochastic. Sander et al.: *"Since Sinkhorn's first iteration coincides exactly with the SoftMax,
  Sinkformers include Transformers as a special case"* — at $l=1$ a Sinkformer is **on the softmax
  line**. Exactness is the $l\to\infty$ limit.
- Sander et al. formulate only the **square** case ($K^\infty \mathbf{1}_n = \mathbf{1}_n$). The
  rectangular $n_q/n_k$ reading used in E1 is **this paper's extension**, not their result.
- `v1_probe.py` hands the Sinkhorn routine $n_q/n_k$ as its target marginal, so the check confirms
  convergence to a supplied marginal rather than an independent property.

---

## Tier C — internal, mechanism-level

### C1 ⭐ "$\tilde R_i = 1$ on average" is false in the regime MarSea operates in

L280: *"The budget normalization makes $\tilde R_i=1$ **on average** but not per row, so the clean
reading '$\tau_i$ below one is slack, above one binds' holds exactly on unit-supply rows."*

$\sum_j c_j = n_q$ is taken **over the active keys** (L241). But the paper is explicit that ungated
keys keep ordinary attention and their mass lands in the same rows — L329–331: *"that row carries
budget-scaled beside softmax-scaled entries and the cap applies to the mixture"*; App. D L1151
repeats it. So

$$\mathbb{E}_i[\tilde R_i] \;=\; 1 \;+\; \mathbb{E}_i\Big[\textstyle\sum_{j\ \mathrm{ungated}} A^{\mathrm{sm}}_{ij}\Big] \;\in\; (1,2),$$

strictly above one whenever *any* key is ungated — which is the normal regime, since MarSea is
part-time by design. **The cap therefore binds systematically more often than $\tau_i>1$
suggests, and $\tau_i=1$ is the identity element only on fully gated rows.** The whole
two-regime paragraph (L272–283) rests on this reading. The honest statement is that the *active
part* of a row averages one unit and the ungated remainder sits on top of it. E8 already reports
the fraction of rows on which the cap binds, so the fix is a sentence, not a measurement.

### C2 App. E's central argument leans on unbounded scores, against the paper's own model

L1296–1303: *"A max over a growing pool grows without bound whenever the non-target law has
unbounded support above, so … $\delta_j\to-\infty$ almost surely … the rate is $\sqrt{\ln n_q}$."*

But App. D (L1118) models pre-LN as confining $u_i$ to a $(d-2)$-manifold, and
$s_{ij} = q_i\!\cdot\!k_j/\sqrt d + b$ with norm-constrained $q,k$ is **bounded**. Under the
paper's own representation model the non-target law has bounded support, the max converges *up to*
the essential supremum, and $\delta_j$ converges to a constant rather than diverging.

**The argument survives and is barely weaker**: $\delta_j$ is non-increasing in $n_q$ and
converges to $\min_{i\in T_j} s_{ij} - \operatorname{ess\,sup}(\text{non-target law})$, which is
negative whenever any non-target can outrank the weakest target — so Assumption 2 still fails
eventually, and still for a reason training cannot remove. The Gaussian $\sqrt{\ln n_q}$ rate
should be labelled as *a* rate under an explicitly stated model, not *the* rate.

### C3 $S_j^{(1)}$ and $S_j^{(2)}$ are used but never defined

Def. `fidelity` defines only $S_j := \supp(p_{\cdot j})$. Prop. `stagecomp`, `tab:verification`'s
six Prop. 20 rows, and E7's "both readings" all use the superscripted pair. Inferable
($S^{(1)} = \supp \tilde A_{\cdot j}$, $S^{(2)} = \supp A_{\cdot j}$), but a definition that
covers the object the composition proposition is about costs one line.

### C4 Prop. `stagecomp`'s "verbatim on the row margin" fails for (iii)

(iii)'s reason is *"since $\theta_i$ is indexed by the row while $P_j$ is a column quantity."* On
the row margin $\theta_i$ and $P_i$ are **both** row quantities, so that reason does not transfer.
The *conclusion* does — a common floor removes a weak true source before a strong false one — but
by a different argument. "Verbatim" is wrong for one of the four.

---

## Tier D — the 456,621 is not a count of independent tests

Block-by-block, with the guaranteeing line quoted:

| block | checks | why it cannot fail |
|---|---|---|
| Prop. 20, six rows | 92,643 | `A = np.maximum(At - th, 0)` with `At ≥ 0` **makes** $S^{(2)}\subseteq S^{(1)}$; (i),(ii),(iv) are its corollaries, counted three times on the same columns |
| Assn. 2 on a scope, "every $n_q$" | 18,000 | `gE = default_rng(7)` is **re-seeded inside the $n_q$ loop**, so the six blocks are byte-identical; `pool` never enters the check. 3,000 columns reported as 18,000 — the tell is `delta scope = 1.127` on all six rows |
| Prop. 19, graded row recall | 18,000 | `At = np.sort(...)[::-1]` then `K = set(range(ki))` — the true sources **are** the top-$k$ because the array was sorted, so "weakest-first (a prefix)" is guaranteed |
| Prop. 18, row interval + monotone | 6,000 | `th` is drawn **strictly between** the two groups (min observed gap 0.10), so $S = K$ always |
| Prop. 14 ×2 + Fig. 2 full-support | 4,024 | softmax has full support, so the assertion is $m/n = m/n$ |
| Prop. 4 on a scope, both rows | 6,000 | `colsum.max() ≥ mu/nk` is **max ≥ mean**; the companion "escape" row is arranged by drawing mask density from $U[0.02,0.5]$ |

**146,667 checks (32.1%) cannot report a violation.** A further **60,472** (Fig. 2's three trace
rows) are arithmetic restatements of the Thm. 2 sharpness rows — given $R_j=1$, $P_j = m/k^\star$
*is* arithmetic — so the same evaluations are counted twice. Total **207,139 (45.4%)**.

The units are also not commensurable: `Thm. 2, sharp recovery interval` reports **6,000 columns**
while the identical scope version reports **24,000 τ-evaluations** for the same
6,000-columns × 4-τ structure; and `Prop. 23 under a nested scope` contributes **131,180** by
counting every prefix timestep — 28.7% of the total from one block.

⚠️ And the paragraph *"What these checks do and do not establish"* claims the harness catches
*"hypotheses that are vacuous or accidentally stronger than intended."* It did not catch these.
**The recommendation is not to shrink the number but to stop leading with it**: report the count
per claim, mark the deterministic-identity rows as such (the paper already does this for one of
them), give the units, and drop "456,621" from the reproducibility statement's first sentence.

## Tier D′ — citation hygiene

- **Position bias (L177, L1084).** *"…architectural account: causal masking and residual
  connections produce a U-shaped profile already at initialization, independently of positional
  encodings \citep{chowdhury2026birth,wu2025emergence,herasimchyk2026residual}."* Only Chowdhury
  makes that claim. **Wu et al. (ICML 2025)** analyze *"the competing effects of the causal mask
  and relative positional encodings, such as the decay mask and RoPE"* — positional encoding is
  central, not factored out. **Herasimchyk et al.** make positional encodings one of four
  interacting forces. "Now-established" across all three overstates the consensus.
- **Three preprints are now published**: `velickovic2024softmax` → **ICML 2025** (PMLR v267);
  `ye2024differential` → **ICLR 2025 (Oral)**; `leviathan2024selective` → **ICLR 2025**. The first
  is described as "closest in motivation", so citing the preprint reads as stale.
- `zhang2023mesh` carries `\natexlab{b}` with **no matching 2023a** anywhere — renders as
  "Zhang et al. (2023b)", a visible artifact of a stale `.aux`.
- Stale source comment at L766: *"Limitations is a \paragraph … purely to hold the 9-page
  submission limit"* — moot under the deferred-limit decision.

---

## What did *not* survive an attempt to break it

Stated so the clean parts are on the record. Thm. 1's product identity and its $\supseteq$
direction; Assn. 1 → Cor. 1's $\sum_i A_{ij}\ge Ma$; Thm. 2's endpoints, re-derived by hand
($G(m_j)=\tau W_j$, $G(m_j{+}1)=\tau(W_j+m_j\delta_j)$); Prop. `printerval`'s two equivalences;
the L-shape's "no interior point" **given Assumption 2**, which the main text now states;
Cor. 5's floor $c_j(1-\tau_jW_j)/m_j$, re-derived from the sparsemax threshold;
$\sum_j A_{ij}=\min(\tau_i\tilde R_i,1)$; Stage 2's substitution $u=a/\tau_i$ and the reading
"larger $\tau_i$ is a tighter budget"; $R_i \ge 1/|K_i|$ from $\theta_i < \max_j \tilde A_{ij}$;
the restricted conservation bound $\mu/n_k$; the arithmetic of the two verification tables
(24 rows → 211,978; 9 rows → 244,643; sum 456,621 = the 41 denominators the program prints);
panel (b)'s twelve numbers; the four Sinkformer masses; the 90.7% permanence figure; the compute
arithmetic ($1{+}15{+}38{+}90{+}80 = 224$, $\times1.5 \approx 336$); and every bibliography entry
not listed in A2.

Also verified correct and **not** to be changed: Expert Choice's $k = nc/e$, load balance by
construction, no expert overflow, unselected tokens skipping the layer; Selective Attention's
$F_{i,j}=\sum_{k\le i-1}S_{k,j}$ and head reuse; Sanford's sparse-averaging separation, the
linear triple-detection lower bound, and communication complexity as the method; Differential
Transformer's $1-\lambda$ row sum; Veličković's dispersion result and self-described "ad-hoc"
remedy; Correia's $\alpha = 1+\sigma(a)$ as a per-head constant; DETR's Hungarian assignment;
RULER's *"recall-based accuracy"* checking presence; MuSiQue's support F1 and paired sufficiency;
HotpotQA's $P^{joint}=P^{ans}P^{sup}$ and joint EM; IHEval's multi-turn conflict levels; **and the
Slot Attention normalization orientation corrected in the previous pass.**
