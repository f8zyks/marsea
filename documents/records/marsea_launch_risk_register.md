# MarSea — what can get in the way of the 150 H100-hours

*2026-09-15, written against the D-24 reduced programme at scope A, gated on S0 / S1.
**Amended 2026-09-16: E6 / IHEval is decided — NOT RUN. §3's third dependency is closed.**
Sources: record §44 (Nebius, D-24), §45 (training procedure), spec §6/§13/§17, `RUNBOOK_nebius.md`,
reviews e16a843 §H and 9bacef9 §C. Ordered by what can actually stop you.*

---

## ⭐⭐ 1. The 150 H100-h figure is unmeasured, and the gate that would catch it sits exactly where the budget breaks

This is the risk I would resolve first, because everything below is scheduled against a number
nobody has measured.

The training arithmetic, from §45's recipe and D-24's reduction:

```
one optimiser step = 16 sequences   (B = 1, token-mean over a 16-sequence window, no packing)
Phase B            = 2,000 steps    (D-24, down from 3,000)
=> one arm x seed  = 32,000 sequence-forward-backwards at 8K
```

Against the S1 step-time gate (`> 1.2 s per sequence ⇒ two patched layers or train at 4K`):

| measured step time | one arm×seed | 17 runs | + Phase A + eval queue |
|---|---|---|---|
| 0.4 s/seq | 3.6 h | **61 h** | ~85 h |
| 0.6 s/seq | 5.3 h | **91 h** | ~115 h |
| 1.0 s/seq | 8.9 h | **151 h** | ~180 h |
| **1.19 s/seq** (passes the gate) | 10.6 h | **180 h** | **~210 h** |

⚠️ **The estimate's uncertainty band is wider than its margin.** At the gate's own pass threshold
the programme is ~40 % over budget, and the gate still says "proceed". The D-23 rule was written to
protect *training feasibility*, not the budget, and nothing currently connects the two.

**Do this in the first hour on the cheap VM:** run `time_step.py` at 8K with the real patched-layer
set, take the number, and recompute the programme from it before requesting the 8-GPU node. If it
lands above ~0.7 s/seq, decide *then* whether to cut to two patched layers, drop Phase B to 1,500
steps, or drop B1/B4 entirely — rather than discovering it on day six.

---

## 2. Capacity, and a tension the runbook has not resolved

- **File the 8×H100 quota request today.** It is the one item with a lead time you do not control,
  and S2 cannot start without it. Nebius support is 24–48 h; against a ten-day deadline a ticket
  raised on day four is unanswerable in time.
- **Region.** VMs *and* the shared filesystem in the same European region (Finland or Paris). A
  quota granted in a different region than the prepared data means re-staging tens of GB, or
  re-generating it.
- ⚠️ **Stop-the-VM discipline and capacity risk pull in opposite directions.** §44 says stop VMs
  between sessions — right for cost ($30.80/h on-demand for 8×H100; idling it for the remaining
  ten days would be ~$7,400). But 8-GPU nodes are the scarce shape, and releasing one may mean not
  getting it back. **Resolution: keep stage 1 entirely on the 1–2 GPU VM, and when you start S2,
  run it as one continuous block.** Do not plan to stop and resume the 8-GPU node mid-programme.
- **RunPod is the declared fallback account.** It is only a fallback if the container image and the
  data tarball are actually staged there. Worth thirty minutes today to confirm, not on the day.

---

## 3. Three data dependencies, all of them human actions, none of them compute

| what | for | state | blocks |
|---|---|---|---|
| **MuSiQue** | E5 | ⚠️ **manual download — a zip on Google Drive**; `run_evalsuite.sh` explicitly checks for it and refuses | S3 |
| ~~**IHEval**~~ | ~~E6~~ | ✅ **CLOSED 2026-09-16: E6 NOT RUN.** Not a download problem — the benchmark carries no tier count, so E6's x-axis does not exist in the data. Zero code changes; the queue already prints "E6 NOT RUN". ⚠️ Do **not** create `data/iheval/iheval.jsonl` | nothing |
| ~~**RULER `--gold_depth` patch**~~ | ~~E3's position control~~ | ✅ **VERIFIED 2026-09-16**: depth pinning correct (0.102/0.289/0.471/0.674/0.880 against 0.1–0.9). ⚠️ But it exposed a **second bug**: sequences were **29 % short** and the shortfall scaled with needle count, making length a function of `n`. Fixed by `patch_ruler_length.py`; a 4-point depth drift remains and is pre-registered | nothing — but **every pre-2026-09-16 set must be deleted and regenerated** |

⭐ **The third is the expensive one.** If `patch_ruler.py` does not pin the gold position correctly,
E3's position control is invalid — and E3 is the one experiment the paper lives on. An invalid
control is not detected by the queue; it is detected by a reviewer. **Verify it by generating ten
examples and reading the gold offsets before generating the full grid.**

All three could be done on a laptop, for free. E6 turned out to be neither a download nor a
decision about one: the data cannot support the experiment as specified, and it is now closed as
NOT RUN, with the cost moved into the paper's text pass rather than the compute budget.

---

## 4. Data generation is CPU-bound and on the critical path

RULER at 8K and 16K, five *n*-levels × 3 seeds × twice (real distractors and inert padding) at
`--n 400` per stratum. Generation and tokenisation are CPU work, they happen before any GPU is
useful, and at 16K they are not quick.

**Start this first**, on the 1–2 GPU VM, in parallel with the step-time measurement and preflight.
It is the one long pole that costs nothing to begin. Provision the shared filesystem for it —
16K-token examples across the full grid is tens of GB (at $0.08/GiB-month the cost is noise; the
provisioning is not).

---

## 5. The serial chain, and the single failed arm that stalls 98 jobs

```
data → preflight → Phase A (per seed) → S0 → Phase B (17 runs) → eval queue (98 jobs) → collect
```

150 H100-h across 8 GPUs is under a day of *compute*. The 8–10 days are this chain. Two specific
hazards:

- **The eval queue now refuses to start on an incomplete grid** (the `e6ca44f` fix, and it is the
  right behaviour) unless `ALLOW_MISSING_ARMS=1`. So **one Phase-B arm that NaN'd at 3 a.m. blocks
  all 98 jobs until a human decides.** Overnight that is a lost shift. Decide the policy in advance:
  my suggestion is to run the queue with `ALLOW_MISSING_ARMS=1` and the skipped list checked in the
  morning, rather than have the queue wait for you.
- **NaN policy costs steps, not just time.** Pre-committed: restart from checkpoint, then halve
  G2/G3 LR once, then report the arm×seed failed. Checkpoints are every 250 steps, so each restart
  loses up to 250 steps of work plus the time to notice.

**Automate the off-platform copy of checkpoints and parquet logs** (§44 asks for a daily copy). If
it is manual, a VM loss costs everything since you last remembered.

---

## 6. Memory at 16K, and the statistical power you have already spent

E3 runs at **16K through the chunked path** — which only recently became correct, and which D-27
put on the critical path. Two pressures:

- Spec §6 holds ≤ 4 patched layers at 16K on 80 GB. The paired-relation cache is **268 MB at 16K**
  and is keyed by `(layer, head)` (H15); the 16K jobs must cap `--n` or build it lazily.
- ⚠️ **`--n 400` was already forced once** (review e16a843 §H) and it has already cost power:
  per-stratum exact-set accuracy carries **±0.14** at 80 examples/stratum/seed; "flat" needs a
  pre-committed equivalence band of **±0.021 to ±0.057** on total drift; MarSea vs B3 and vs B2 are
  **under-powered at three seeds**.

⭐⭐ **This is the failure mode that does not look like a failure.** The programme can complete, every
gate can pass, 150 H100-h can be spent — and E3's headline can still be *statistically unable to
distinguish flat from drifting*. If memory forces `--n` down again, that becomes likely rather than
possible.

**Two cheap mitigations, both decidable now:** pre-commit the equivalence band in writing before you
see the data (the supportable claim is *"drifts by less than 5 points across a 16× sweep in n while
the comparator falls from 0.5 to 0.03"*, not an unqualified "flat"); and if there is any budget
slack after §1's measurement, spend it on **more examples in E3's strata** rather than on more arms.

---

## 7. What each gate costs if it fires

| gate | when | if it fails | cost in days |
|---|---|---|---|
| detector threshold | preflight | App. H's fallback case, not a run | stop |
| **θ-window** | preflight 4b | now a defined branch: set `MARSEA_TOL_CAP = 4 × measured window`, re-run | **minutes** |
| identities / coverage | S1 | implementation defect; nothing downstream is meaningful | 1–2 |
| **step time > 1.2 s/seq** | S1 | two patched layers, or train at 4K | ½ day + see §1 |
| **S0 replication pair** | S0 | E7's column half is "not measurable on this backbone"; E7 goes row-side | 0 compute, −1 experiment |
| **S2 precision not flat in n** | S2, ~day 6 | pivot: report the training-effect claim and **rescope the paper** | ⚠️ a rewrite with 4 days left |

⭐ The window gate is the one that used to be a stop and is now a five-minute branch — that is what
the last three review rounds bought. The two that remain genuinely expensive are the step time
(because of §1) and S2's own gate (because of when it fires).

---

## 8. Platform reliability, stated plainly

~38 Nebius status-page incidents May–Sep 2026, a 10-hour `us-central1` degradation in August,
credits-only SLA, 24–48 h support. Across ~8 days of wall clock with a 10-day deadline: **one
10-hour degradation is survivable; two are not.** Every run is idempotent on restart and
checkpoints every 250 steps, which is the right design — but it only helps if the restart is
automatic or someone is watching.

---

## The short list for today, none of it needing a GPU

1. Measure the step time and **recompute the budget from it** (§1).
2. File the 8×H100 quota request (§2).
3. ✅ MuSiQue downloaded; ✅ E6/IHEval decided (NOT RUN); verify `patch_ruler.py` pins the gold
   depth (§3) — the last of the three, and the expensive one to get wrong.
4. Start RULER generation for 8K and 16K (§4).
5. Write down the equivalence band for "flat" before seeing any data (§6).
6. Confirm the RunPod fallback image and data tarball actually exist (§2).
7. Decide the `ALLOW_MISSING_ARMS` policy and automate the off-platform copy (§5).
