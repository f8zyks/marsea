# MarSea — the fine-tuning procedure in full (2026-09-06)

Companion to the implementation contract (`marsea_implementation_spec.md` v4.1, §9 is the
summary of this document). Normative on the training loop; where it and the spec disagree this
document wins on training, the spec wins on the mechanism. Record §44 gives the compute plan.

The procedure is the same for every arm. What differs per arm is listed once, in §10.

---

## 1. What is being trained, and what is not

```
FROZEN      the backbone (Qwen2.5-1.5B base): embeddings, every attention projection, every MLP, norms, lm_head
TRAINED     (a) LoRA adapters on q_proj, k_proj, v_proj, o_proj of ALL 28 layers   r=16, alpha=32, dropout 0.05
            (b) the arm's own modules, in the PATCHED layers only (L_patch = 4, §6.7 of the spec):
                MarSea/B3 : U_phi, V_phi (Linear d_head->r, no bias, shared across heads), b0 (scalar per layer),
                            TauK (LN + MLP 128+6+1 -> 64 -> 1), TauQ (LN + MLP 128+4 -> 64 -> 1)
                B2 (MESH) : h_a, h_b (LN + MLP 128 -> 64 -> 1)
                B0, B1, B4: nothing beyond the LoRA;   B5: nothing at all (evaluation-only, on B0's weights)
```
Parameter counts (1.5B backbone): LoRA ≈ 9.2M; MarSea modules ≈ 4 × (2·128·16 + 2·(135·64+64)) ≈ 90K.
All trained parameters are kept in **fp32 master copies**; the frozen backbone is bf16.

Why LoRA on all layers rather than only the patched ones: the arms must differ *only* in the
normaliser; giving every arm the same adaptation capacity everywhere is what makes "the same
decoder under the same schedule" true. Why the heads are only in patched layers: they are the
mechanism, and the mechanism lives at the patched softmax line.

---

## 2. Data

### 2.1 Sources and the training pool
```
RULER-NIAH   generated with the backbone tokenizer, TRAINING seeds 100-102 (evaluation uses 0-2; never overlap),
             L = 8K, a uniform mixture over the grid the experiments will sweep:
               task ∈ {niah_single_2, niah_multivalue, niah_multikey_1}, num_needle_k ∈ {1,2,4,8,16,32},
               num_needle_v ∈ {1,2,4,8}, depth ~ U(0.1, 0.9)   (feasibility: K*V <= 128 sentences)
             answer = the generator's `outputs` joined by ", "
MuSiQue      musique_ans_v1.0_train.jsonl (19.9K); passages in hop order with distractors interleaved at seeded
             slots (spec §13.1); answer = `answer`
HotpotQA     train, distractor setting, a 20K seeded subsample; 10 paragraphs as shipped; answer = `answer`
mix          1 : 1 : 1 by examples, interleaved deterministically from a per-seed permutation of each source
```
3,500 steps × 16 sequences = 56,000 sequences ≈ 18.7K per source: MuSiQue is seen about once,
HotpotQA and RULER are subsampled once. No example is repeated within a run (no epochs).

### 2.2 Sequence construction
```
text     = "{context}\n\nQuestion: {question}\nAnswer:"  +  " {answer}"  +  EOS          (RULER: its own template + answer_prefix)
labels   = -100 on every prompt token; the answer tokens and the EOS carry the label
length   = the sequence's own length (B = 1 per GPU: NO padding, NO packing); sequences longer than L are dropped
           at generation/selection time, never truncated (a truncated context can cut the gold passage)
```
With `B = 1` there are no pad rows, so `vis` is the plain causal mask and every INV-2 count is `n_q`.

### 2.3 Order and reproducibility
A run's data order is a function of `(seed)` alone: `perm_source = permutation(seed, source)`,
consumed in lock-step 1:1:1. The data cursor is part of the checkpoint (§8), so a restart resumes
the identical order. The three seeds {0, 1, 2} therefore differ in data order, LoRA init, head
init and dropout, and in nothing else.

---

## 3. Loss

Next-token cross-entropy on answer tokens only (labels `-100` elsewhere), **token-mean over
the accumulation window**: sum the per-token losses of the 16 sequences and divide by the total
number of labelled tokens in the window (not sequence-mean, which would weight a one-token RULER
answer like a five-token MuSiQue answer). No auxiliary loss. No coverage penalty (the relation
is free to close; that is a reported outcome, not a prevented one — spec §4.4). No KL to the base
model. No label smoothing.

---

## 4. Optimiser, schedule, precision

```
optimiser     AdamW, betas (0.9, 0.95), eps 1e-8
param groups  G1 LoRA A/B matrices          lr 2e-4   weight_decay 0.01
              G2 arm modules (weights)      lr 1e-3   weight_decay 0.01      (added at the start of Phase B)
              G3 arm modules (biases, b0, LayerNorm gains)   lr 1e-3   weight_decay 0.0
schedule      ONE cosine over the run's total steps (3,500 full / 2,500 reduced), 3 % linear warm-up (105 / 75 steps),
              decaying to 10 % of peak at the last step; the schedule is NOT restarted at Phase B — G2/G3 join it at
              step 500 at the schedule's current multiplier
clipping      global grad-norm clip 1.0 over all trainable parameters, applied once per optimiser step
accumulation  16 micro-batches of 1 sequence per optimiser step (per GPU)
precision     backbone bf16; torch.autocast(bf16) around the forward; the normaliser of every patched layer runs
              inside autocast(enabled=False) in fp32 (spec §6.3); LoRA and heads in fp32; no GradScaler (bf16)
checkpointing gradient checkpointing on the 4 patched layers always; on all layers if the 8K step exceeds memory
```

---

## 5. Phase A — the shared warm start (steps 0–499)

Run **once per seed**, with the normaliser set to `SoftmaxNorm` in the patched layers (so the
patched model is the unpatched model up to T0's tolerance) and only G1 trainable. Save the
checkpoint `phaseA_seed{s}.pt` (LoRA weights, optimiser state for G1, scheduler state, data
cursor at 500, RNG states). **Every arm forks from this file**; B0 simply continues it.

Acceptance before forking: training loss decreased monotonically in the 50-step moving average;
the 250-step eval (§8) shows RULER exact-set accuracy above the base model's; no NaN.

---

## 6. Phase B initialisation (at step 500, once per arm × seed)

Performed in this order, each step verified, all from the loaded Phase-A checkpoint:

```
1  instantiate the arm's modules (fresh, seeded):  U_phi, V_phi ~ N(0, 1/d_head);  TauK/TauQ hidden layers
   default init, LAST layer weight ~ N(0, 0.01^2);  LayerNorms at identity
2  calibration batch: 8 sequences from the training mix at the data cursor (they are NOT consumed), no grad,
   the arm's normaliser live but with E forced empty (so tau/relation statistics are read from A_sm)
3  b0 per patched layer: bisection (20 iters) on b0 so that  mean over the batch of  |E & vis| / |vis|  = rho_0 = 0.05
   (coverage measured with the hard E, per layer; all Q-heads pooled)
4  TauK last-layer bias per layer := softplus^{-1}( median_j (1/std_j) - tau_min ),  std_j over visible entries of
   column j on the calibration batch (guards of spec §5.2);  TauQ last-layer bias := softplus^{-1}(1 - tau_min)
   => tau_i = 1 at init, the identity on step 1's row (record §43)
5  sanity: (a) with E forced empty, the loss on the calibration batch equals Phase A's to 1e-4 (INV-11 at the
   model level);  (b) with the relation live, the loss differs from (a) by < 5 % and no tensor is NaN.
   A larger gap means the init is wrong; STOP and inspect, do not train through it.
6  add G2/G3 to the optimiser with fresh Adam moments; the scheduler continues from step 500
7  write the calibrated values (b0, both biases, measured coverage per layer) to the run's README
```
For B2 the same steps with `h_a, h_b` fresh (their last-layer bias 0 gives uniform marginals at
init, i.e. `a = b = 1` per element, the closest MESH gets to standard attention) and steps 3–4
skipped. For B0/B1/B4 nothing is added; Phase B is Phase A continued with the arm's normaliser.

---

## 7. The step (Phase B, steps 500–3,499 or 500–2,499)

```
for step in range(start_step, total_steps):
    optimizer.zero_grad(set_to_none=True)
    n_tok = 0; loss_sum = 0
    for micro in range(16):
        ids, labels = next_sequence(cursor); cursor += 1                    # one sequence, its own length
        state = {"nu_prev": None}                                           # per-forward MarSea state
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, marsea_state=state)                         # patched layers write diagnostics
        loss_tok = cross_entropy(logits[:, :-1].float(), labels[:, 1:], ignore_index=-100, reduction="sum")
        (loss_tok / TOKENS_IN_WINDOW).backward()                            # TOKENS_IN_WINDOW pre-counted for the window
        loss_sum += loss_tok.item(); n_tok += (labels != -100).sum()
        diag.accumulate(state["diagnostics"])                               # detached, per patched layer/head
        if not torch.isfinite(loss_tok): raise NaNError(step, micro)        # §9
    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
    optimizer.step(); scheduler.step()
    log_scalar("loss", loss_sum / n_tok, step)
    if step % 50 == 0:  log_E8_block(diag.flush(), step)                    # §8.1
    if step % 250 == 0: evaluate(step); checkpoint(step)                    # §8.2, §8.3
```
Inside a patched layer's forward, `state["nu_prev"]` is read (or `1.0` in the first patched
layer), the normaliser runs in fp32, `state["nu_prev"] = nu` is written for the next patched
layer (**not detached**, spec §5.3), and the diagnostics are appended detached. `tau_j` is
predicted from the full teacher-forced column: training uses no sealing (spec §7 applies to
decode; INV-9 is a decode-time property).

Throughput expectation (spec §6.5 / record §44): ≈ 0.6 s per sequence at 8K in a tight
implementation → ≈ 10 s per optimiser step → Phase B ≈ 8.3 h (3,000 steps) or 5.6 h (2,000).
S1 measures this and applies the pre-committed rule (> 1.2 s/sequence → two patched layers or
4K training).

---

## 8. Monitoring, evaluation, checkpointing

### 8.1 The E8 block (every 50 steps, from the training batches, per patched layer and Q-head)
```
coverage rho (col, row);  histograms of |E_.j| and |E_i.| (bins 0,1,2,3-4,5-8,9-16,17+);  fraction of columns with
|E_.j| = 1;  across-key var(tau_j), across-query var(tau_i);  k* histogram;  fraction of rows the unit cap bound;
fraction of rows with cbar_i = 0;  tau_i median about 1;  b0 value;  grad-norm of each param group;  NaN/inf counts
```
These are **reported, never acted on** — with one exception, §9. Pre-registered readings: coverage
below 0.5 % on every patched layer = "became B0"; `|E_.j| = 1` mass above 80 % = "trained out of
Thm. 1" (spec §4.4); across-key var(τ_j) → 0 = "constant temperature". Each is a finding.

### 8.2 Evaluation (every 250 steps; ≈ 3 min)
A fixed held-out set: 200 RULER examples spanning the E2/E3 grid at 8K (evaluation seed 0) and
100 MuSiQue dev examples. Teacher-forced: loss, relation coverage, interval-hit rate on the
default `T_j` construction, row precision/recall at `(l*, h*)`, the three-way distractor split.
Generation (greedy, 32 tokens, the frozen-prefix decode of spec §6.4) on 50 RULER examples:
exact-set accuracy. The 250-step curve of these is kept with the run; the paper's numbers come
from the full E-suite on the final checkpoint only.

### 8.3 Checkpoint (every 250 steps and at the end)
`{lora adapters (peft), arm modules state_dict, optimizer, scheduler, data cursor, RNG states (torch, cuda, numpy,
python), step, spec version, git hash, container digest}` → shared filesystem; keep the last two plus Phase A and
final. Restart = load, seek the data cursor, continue; the result is bit-for-bit the same order.

---

## 9. Failure handling (pre-committed)

`NaN/inf` in the loss or any gradient: the step is abandoned, the run restarts from the last
checkpoint with the identical data order, and the event is logged with (step, micro, layer). If
the same step NaNs again, the head learning rate (G2/G3) is halved once and the halving is
recorded in the run's README; a third occurrence stops the run and is reported as a failure of
that arm × seed. No other automatic intervention exists: a collapsing relation, a saturating
`τ_j`, or a rising loss are outcomes, and the pre-registered reading in §8.1 is what the paper
says about them.

---

## 10. Per-arm differences (everything else identical)

| arm | normaliser in the patched layers | modules trained in Phase B | Phase B init |
|---|---|---|---|
| B0 | SoftmaxNorm | none | none (Phase A continued) |
| B1 | SoftmaxOneNorm | none | none |
| B2 | MESHNorm | `h_a, h_b` | fresh; last bias 0 |
| B3 | MarSea with `column_stats`, `nu` zeroed at `TauK`'s input | same as MarSea | same as MarSea (§6) |
| B4 | RowEntmaxNorm (α = 1.5 or 2) | none | none |
| B5 | — (evaluation only, B0's weights + MarSea's trace) | — | — |
| **MarSea** | `marsea_normalize` | `U_phi, V_phi, b0, TauK, TauQ` | §6 |
| E9 arms | MarSea with one change (τ_i pinned at 1; uniform quota `cbar_j/|E_.j|` in step 2; key-alone relation `u_phi(k_j)` only) | as MarSea minus the removed part | §6 |

---

## 11. What a healthy MarSea run looks like (so the first one can be judged in an hour)

Steps 500–600: loss within 5 % of Phase A's final; coverage drifting from 5 % (either way is
fine, 0 % is not); `τ_i` median within 0.8–1.3; `k*` histogram mostly at 1–3; no cap-binding on
more than ~10 % of rows. Steps 600–1,500: loss below B0's at the same step on the 250-step eval
*or* not — either is reportable — but the interval-hit rate on the held-out set should rise
above chance (≈ 0.3 on the default construction) by step 1,500 if the field argument is doing
anything; if it has not by step 2,000, E7's pre-registered falsification applies and the run
is still complete and reported. Final: the E8 block's three degeneracy readings all negative.
