# MarSea — the 10-day run plan

*2026-09-17. Scope A on D-24's reduced programme, gated and pivotable on S0 / S1.*

⭐⭐ **PROVISIONING DECIDED 2026-09-17: RunPod, one on-demand 8 × H200 SXM pod, both stages.**
Nebius is off the table — it has **zero** on-demand 8-GPU H200 capacity in any region, and its
preemptible path needs a supervisor loop, a boot autostart unit and a SIGTERM handler that do not
exist, plus a resume path nobody has tested under a kill. `patch_multivm.py` goes back on the
shelf **unapplied**. **The provisioning detail lives in `marsea_runpod_playbook.md`**; this plan
carries the science.

**E6 is decided: NOT RUN (§1.4). D0's data work is COMPLETE (§1.5): 126 sets, verified.**
**Abstract: Fri 18 Sep (TOMORROW). Paper: Fri 25 Sep AoE.**
Every command below exists in the repo at `7e05ad6`. Companion documents:
`marsea_runpod_playbook.md` (provisioning → teardown), `marsea_launch_risk_register.md`,
`marsea_experiment_programme_map.md`, `RUNBOOK_nebius.md` (misnamed — nothing in it is
provider-specific; it is the mechanism behind each gate).

---

## 0. The shape of the ten days

| day | date | where | what | drop-dead |
|---|---|---|---|---|
| ~~**D0**~~ | ~~Wed 16~~ | ✅ **done** | data generated and verified; E6 closed; the band written | — |
| **D1** | **Thu 17 (today)** | 1 × H200, then **8 × H200** | staging run + upload (§1.7) → 8-GPU bring-up → preflight → **budget recompute** → Phase A → S0 → S1 | S0 licence by 22:00 |
| **D2** | Fri 18 | same pod | read S1 at one hour; ⭐ **REGISTER THE ABSTRACT**; launch S2 | S2 launched by 20:00 |
| **D3–D5** | Sat 19 – Mon 21 | same pod | the training grid, 20 jobs in **3 waves** (8 / 8 / 4) | training done by Mon 22:00 |
| **D5–D7** | Mon 21 – Wed 23 | same pod | the evaluation queue (**96** jobs), then `collect.py` | tables by Wed 22:00 |
| **D8** | Thu 24 | laptop | numbers into the paper; the unfunded-claims pass | text frozen by 22:00 |
| **D9** | Fri 25 | laptop | build, checks, submit | **AoE — the deadline** |

⚠️ **One provider, one device class, every arm.** The `MARSEA_TOL_CAP` guard refuses a θ-window
measured on a different device, and arms split across providers are confounded by floating-point
non-determinism. Everything on RunPod H200 SXM, or nothing.

⚠️ **One pod, up continuously.** A stopped pod releases its GPUs and may not get them back. Run
stage 1 → S2 → eval as one block, then terminate — after `runs/` is off the pod.

⭐ **The two hard commitments.** Register the abstract on D2 whatever the state of the run — it costs
nothing and depends on no result. And **stop the science at D8 noon**: whatever numbers exist then
are the numbers. Every hour after that belongs to the text.

---

# PART 1 — before the 8-GPU pod

Eight items. **Six are done** (16 Sep) — kept here with their outcomes, because the outcome is what
D1 depends on and because a struck-through item is the cheapest way to stop someone re-opening it.
Two are open and both are for **this morning, before the 8-GPU pod**: §1.7, the staging run that
gets the data onto a network volume without paying for eight idle GPUs, and §1.8, the paper text
that needs no results.

### 1.1 ✅ Provisioning — RESOLVED 2026-09-17: RunPod, 8 × H200 on demand

**Full detail in `marsea_runpod_playbook.md`.** The three things that belong here:

| | |
|---|---|
| pod | 8 × **H200 SXM**, Secure Cloud, CUDA 12.8+ image, 100 GB container disk |
| storage | ⚠️⚠️ **a network volume, created BEFORE the pod** — it cannot be attached later, and it is the only kind that survives terminate. 500 GB, mounted at `/workspace` |
| cost | ~210 GPU-h billed over ~26 h wall clock (you pay all eight during single-GPU stage-1 work) |

⚠️ **Four machine checks before anything else**, now run by `scripts/setup_runpod.sh` step 0 (which
hard-fails on the first two): `nvidia-smi` (8 GPUs, ~141 GB each), the **product name**, CUDA ≥ 12.8
(torch 2.11.0 cu128 fails at *import*, not install), and **`bash --version` ≥ 5.1** —
`run_evalsuite.sh` uses `wait -n -p` and refuses to start without it.

✅ **The product-string check is a deny-list as of `b91a0a8`** — `NVL|PCIe`, plus a memory floor
(`MIN_GPU_MIB`, 140 000) and a visible-count check (`EXPECT_NGPU`, 8). NVIDIA does not put "SXM" in
the name (an H100 SXM5 reports `NVIDIA H100 80GB HBM3`, an H200 SXM reports plain `NVIDIA H200`),
so the old allow-list warned on the correct hardware. **A correct pod reads plain `NVIDIA H200` at
~141 GB**; anything carrying `NVL` or `PCIe` is not what the budget assumes. Read the printed names
once by eye anyway — it costs five seconds and no string test survives a new SKU.

⭐ **Why not the Nebius fleet.** `nebius capacity resource-advice list` on 2026-09-16 showed
on-demand 8-GPU H200 at **0 available in all three regions**; only preemptible 1-GPU was abundant.
The fleet design (`patch_multivm.py`, `reap_claims.sh`) was correct for that world and is **tested
and shelved** — it needs no decision now, and would need a supervisor loop and a boot unit that
were never written. Keep it for a future run where capacity is the constraint again.

### 1.2 ✅ Data — COMPLETE, and it is uploaded, not regenerated

126 sets, ~2 GB, generated and verified on the desktop (§1.5). ⚠️ **Do not regenerate on the pod**:
it is CPU work, it took ~6 hours single-threaded on the desktop, and on the pod it would be billed
at 8-GPU rates while eight H200s sit idle.

⭐ **Run this against the 1-GPU staging pod (§1.7), not the 8-GPU one.** The target is the network
volume, which outlives that pod, so the data is in place before the expensive pod exists.

```bash
rsync -avP ~/Dev/marsea/data/ruler/   root@<pod>:/workspace/marsea/data/ruler/
rsync -avP ~/Dev/marsea/data/musique/ root@<pod>:/workspace/marsea/data/musique/
```

Then confirm nothing truncated — 126 sets, every one carrying `measured_length_*` provenance
(the check is in the playbook §1.3).


### 1.3 ✅ MuSiQue — DONE 2026-09-16

`musique_ans_v1.0_train.jsonl` and `musique_ans_v1.0_dev.jsonl` are in `data/musique/`. This was on
the critical path for the *training* grid, not just E5: nothing else acquires the release,
`build_training_sources` raises without it, and `run_s2.sh` refuses to start.

⚠️ It is therefore part of the D1 upload (§1.2), not something the pod can fetch — the release is a
Google Drive zip and there is no unattended download for it. Size it into the tarball and confirm
both files land under `data/musique/` on the pod before preflight; `run_s2.sh` will otherwise fail
*after* the 40-minute preflight rather than before it.

### 1.4 ✅ E6 / IHEval — **DECIDED 2026-09-16: NOT RUN.** No action required

The oldest open decision in the project, closed. It turned out not to be a download question.

⭐ **Why.** IHEval's records (`github.com/ytyz1307zzh/IHEval`, `benchmark/`) carry
`id, system, conversation_history, instruction, answer{instruction_id_list, kwargs}`. Three
mismatches with `marsea/data/iheval.py` and `scripts/run_e6.py`, the first of them fatal:

1. ⚠️ **There is no tier count, and there cannot be one.** `_tier_count` looks for
   `num_tiers | n_tiers | n_conflicts | num_conflicts | depth | conflict_depth`; none exists, so
   every record is dropped and `run_e6.py` trips its own `assert exs`. Conceptually, IHEval has a
   *fixed* four-level hierarchy (system > user > conversation history > tool output) and injects
   **one** conflicting instruction at **one** level per setting — "number of conflicting tiers" is
   not a variable the benchmark sweeps. **E6's x-axis does not exist in the data.**
2. The loader wants `turns`/`user`; the data has `conversation_history`/`instruction`, so the
   prompt would carry an empty `User:` line.
3. `answer` is a dict of IFEval-style verifiable constraints, not a reference string; `run_e6.py`
   string-matches against `str(ref)`. Against a constraint benchmark that is ~0 for every arm.

Making E6 real means re-aiming it at the *placement* axis and vendoring the IFEval checkers — a
day, landing on a 1.5B base checkpoint that may floor out. Not this week.

⭐ **Zero code changes are required, and this was verified.** `EVAL_DATA_CHECK` does not look for
IHEval, so the 96-job queue cannot refuse over it; the checkpoint inventory adds E6 arms to
`SKIPPED_ARMS` only `if [ -f "$E6_SET" ]`; and the launch block prints
`E6 NOT RUN: no IHEval set at data/iheval/iheval.jsonl` into the queue log, which is the
provenance record.

⚠️⚠️ **DO NOT create `data/iheval/iheval.jsonl`.** If any file exists at that path — a dump of the
HF mirror `zhihz0535/IHEval`, say, which in any case carries only the *aligned* rule-following
split and no conflict setting — the queue runs E6, every record drops, and `e6_marsea`/`e6_B0`
return as failed jobs that make the whole queue exit non-zero. **Leaving the path empty is the
correct action, not an omission.**

⚠️ **What it costs the paper.** E6 was the only direct evidence for the **ranked** half of §1's
motivation. Those sentences now have no arm behind them at all, which makes §6.2's unfunded-claims
pass both larger and more urgent — see the programme map's claims table.

### 1.5 ✅ RULER verification — DONE 2026-09-16, and it found a second, worse bug

**Depth pinning: correct.** Measured fractions 0.102 / 0.289 / 0.471 / 0.674 / 0.880 against
requested 0.1–0.9, sd ≤ 0.010, ranges non-overlapping. `patch_ruler.py` does what it claims.

⚠️⚠️ **Sequence length: was 29 % short, and the shortfall scaled with the needle count.**
`--grid E3depth --L 16384` produced a mean length of **11,598 tokens**, spread 117 — systematic.
Cause: RULER's `tokens_per_haystack` probe divides the WHOLE probe's token count (500 haystack
words **plus every needle plus the template**) by 500, charging the needles' fixed cost to the
per-word rate. At 128 needles that inflates it ~5×, the built-in 3× slack is outgrown, and the
binary search converges to its own upper bound. RULER's own assert never fires because the
sequence is *under* the target, not over.

⭐ **Why that was worse than a wrong number in the paper.** E3 sweeps `K ∈ {8…128}` at `V = 4`
(32→512 needles) with the padding control at `K = 1`. Uncorrected, **sequence length falls
monotonically with the swept variable `n`** and the "inert padding of equal token count" control is
several times longer than the point it controls — a confound inside the load-bearing experiment,
pointing the same way as the claimed effect.

**Fixed by `patch_ruler_length.py`** (delivered 2026-09-16, applied): two-point differencing for the
rate (cancels the needle and template constants exactly), MARKER v2 → v3, and a **length gate** in
`ruler.generate()` that deletes and refuses any set whose mean length falls below
`MARSEA_LENGTH_FLOOR` (0.95) of `L`, recording `measured_length_*` in `config.json`.

**After the fix**, at `--n 20 --seeds 0`:

| | length | % of L | gold `frac` |
|---|---|---|---|
| E3pad K=1 | 16,096 | 98.2 % | 0.464 |
| E3 K=8 | 16,139 | 98.5 % | 0.470 |
| E3 K=16 | 16,156 | 98.6 % | 0.470 |
| E3 K=32 | 16,008 | 97.7 % | 0.471 |
| E3 K=64 | 15,794 | 96.4 % | 0.485 |
| E3 K=128 | 16,188 | 98.8 % | **0.510** |

Length spread 2.4 % and **non-monotone** in `K` — the confound is closed. ⚠️ A **4-point gold-depth
drift remains and IS monotone** (0.470 → 0.510), because needles are placed at uniform random depths
and are not the length of an average haystack sentence. Small — the gold never leaves mid-sequence —
but monotone, so it is **pre-registered in the draft** (`patch_iclr_e3prereg.py`, applied) and will
be bounded by E3depth's own measured slope rather than argued.

⚠️⚠️ **Two traps left behind by all of this:**

1. **The generator's output is only checked at generation time.** `generate()` returns an existing
   target untouched, so a short set on disk is never repaired — it must be **deleted**. Every grid
   made before 2026-09-16 is short: `E2`, `E4`, `detector`, `eval_quick` and the training pool.
2. **The training pool has never been generated under the current naming.** `data/ruler` holds
   `TRAIN8K_*` and `TRAINSMOKE_*` from an older scheme; `grid_training` now names sets `TRAIN_*`
   and `run_s2.sh` globs `data/ruler/TRAIN_*`, which matches **none** of them. ⭐ S2 would refuse to
   start. Delete the orphans and run `gen_data.sh`, which does the training pool first — it is the
   longest CPU job in the plan (≈23 configs × 3 seeds × 400 samples at 8K).

```bash
rm -rf data/ruler/TRAIN8K_* data/ruler/TRAINSMOKE_* data/ruler/SMOKE_*   # orphans, wrong prefix
rm -rf data/ruler/E2_* data/ruler/E4_* data/ruler/DET_* data/ruler/QUICK_*  # short, pre-fix
bash scripts/gen_data.sh 2>&1 | tee runs/gen_data.log
```

⭐ **Do this BEFORE §2.5's step-time measurement.** Training sequences were ~11.6 K and become
~16 K at the same nominal `L`; step time and memory both rise, so a budget measured on the old data
is wrong in the optimistic direction.

### 1.6 ✅ The equivalence band for "flat" — WRITTEN 2026-09-17, before any accuracy exists

⚠️ **The earlier draft of this section staked the band on the wrong quantity.** It pre-committed
*per-stratum exact-set accuracy* — a task-level number with ~200 examples per (`n`, seed), whose
95 % interval is ±0.11 per point. You cannot demonstrate flatness to ±0.05 with an instrument whose
own error bar is ±0.11. Worse, it is not the quantity the theory speaks to:
Prop. `fullsupportceiling` is a statement about **support precision** `P_j`, not about answers.

#### 1.6.1 The quantity, and why it has the sample size the claim needs

⭐ **Primary, and the only thing "flat" is claimed of:** MarSea's **support precision `P_j`**,
averaged over active columns with `δ_j > 0`, as a function of `n`.

Support precision is measured **per column**, not per example. At `n = 128` with 200 examples there
are up to ~25,600 active columns per (`n`, seed) — two orders of magnitude more than the 200
task-level observations. The measurement is therefore tight and the residual uncertainty is
**between-seed**, which is exactly where three seeds hurts and where the honesty below applies.

**Secondary, reported but NOT claimed flat:** task-level exact-set accuracy, supporting-fact
precision/recall, answer F1. These carry ±0.11-class intervals at three seeds. Report them with
intervals; make no equivalence claim on them.

#### 1.6.2 The band, and where the number comes from

> **Pre-registration.** *Across the 16× sweep `n ∈ {8, 16, 32, 64, 128}`, MarSea's support precision
> on `δ_j > 0` columns drifts by no more than **0.05 in total** — equivalently **0.0125 per doubling
> of `n`** — while the full-support comparator's precision falls from **0.50 to 0.031** by the
> identity `m_j/n` of Prop. `fullsupportceiling`.*

The band is anchored to the contrast it has to survive, not chosen for convenience: the comparator's
drop is **0.469**, so ±0.05 is **9.4× smaller than the effect being demonstrated**. A band that
cannot be distinguished from the comparator's fall would be worthless; one much tighter than 0.05
would fail on measurement noise rather than on the science.

#### 1.6.3 The test, stated now so it cannot be chosen later

1. Per seed, fit `P_j` against `log₂ n` over the five levels; take **Δ = P̂(128) − P̂(8)** as the
   total drift. Three seeds give three values of Δ.
2. **TOST** against the band ±0.05 at α = 0.05 one-sided each side — equivalently, **equivalence is
   declared iff the two-sided 90 % CI of mean Δ lies entirely inside ±0.05.**
3. Report, beside it, a **within-seed bootstrap over columns**, so a reader can see that the
   measurement is precise and that whatever width remains is between-seed rather than noise.

⭐⭐ **What three seeds can and cannot do, computed now rather than discovered later.** With
`df = 2`, `t₀.₉₅ = 2.920`, the 90 % CI half-width is `1.686 × s`, where `s` is the between-seed sd
of Δ. For the CI to fit inside ±0.05 with Δ̂ ≈ 0, the design needs

> **`s ≤ 0.030`.**

That is a **pre-registered feasibility condition and it is checkable from the data**. If the observed
between-seed sd exceeds 0.030, the design cannot declare equivalence at this band whatever the point
estimate says — and we say so rather than quietly reporting Δ̂ as though it were flat.

#### 1.6.4 The four outcomes, and what each licenses

| observed | what is written |
|---|---|
| 90 % CI ⊂ ±0.05 | ✅ the pre-registered claim, as worded above |
| Δ̂ inside the band, CI crosses it | ⚠️ *"drifts by Δ̂ (90 % CI …) across a 16× sweep"* — **the word "flat" is not used**, and the underpowering is stated |
| 0.05 < \|Δ̂\| ≪ 0.469 | *"degrades slowly — Δ̂ over 16× against the comparator's identity fall of 0.47"*. A weaker claim, still a result, and still a separation |
| \|Δ̂\| comparable to 0.469 | the prediction fails. Report it, and take §7's S2 pivot |

#### 1.6.5 ⚠️ The honest caveat the claim carries, and it is not small

**`δ_j > 0` is a subset whose membership depends on `n`.** App. `selective`'s own table has the
recovery fraction falling 63.6 % → 14.9 % → 1.2 % at `n` = 32, 128, 512 on the full column. So
"precision is flat on `δ_j > 0` columns" is flatness **on a shrinking, increasingly selected
population** — a weaker statement than flatness on a fixed one, and a reviewer will say so first.

Two things are therefore reported together, and the claim is the **conjunction**, never the first
alone:

1. `P_j` on `δ_j > 0` columns — pre-registered flat to ±0.05;
2. the **fraction** of active columns with `δ_j > 0` at each `n` — predicted to fall.

⭐ And `P_j` on **all** active columns is reported beside both: the paper already says a flat value
*there* would mean the relation, not the program, is doing the selection (App. `selective`, cost (a)).
A drift **upward** in (1) is the signal to watch — it would suggest the surviving subset is becoming
selectively easier as it shrinks, which is the selection effect rather than the mechanism.

#### 1.6.6 Where this goes

Into the E3 paragraph of §5, beside the generator pre-registration already applied
(`patch_iclr_e3prereg.py`), and into every E3 run README. ⭐ It replaces the draft's current
unfalsifiable *"a flat precision on the `δ_j > 0` columns … is the theory's own prediction"*, which
names no band and so cannot fail — **item 7 of the unfunded-claims pass**. That sentence is not an
unfunded claim; it is an unfalsifiable one, which is worse.

### 1.7 ⭐ The staging run — first thing this morning, ~1 hour, **1 GPU not 8**

RunPod is the primary now, not the fallback, so the question is no longer whether it exists but
whether D1's first hour is uneventful. Everything below is **GPU-count-independent** — the image,
the volume, the upload, the environment build — and a 1×H200 hour costs about a tenth of an 8×H200
hour. Doing it on one GPU and *then* launching the 8-GPU pod saves roughly seven GPU-hours of paid
idling while a 2 GB tarball crawls up a home uplink, and, more to the point, moves every discoverable
failure to a cheap hour.

1. **Create the network volume first** (⚠️⚠️ it cannot be attached to an existing pod — see
   `marsea_runpod_playbook.md` Part 0). Pick the datacenter now: the volume pins the region, and the
   8×H200 pod must be available *in that region* an hour from now. Check 8×H200 availability there
   **before** you create the volume.
2. Launch a 1×H200 pod on that volume with the playbook's image and confirm the four verification
   commands (Part 1) pass: CUDA visible, `torch` sees the GPU, the volume is mounted read-write at
   the expected path, and `bash -c 'echo $BASH_VERSION'` is **≥ 5.1** (`run_evalsuite.sh` needs
   `wait -n -p`).
3. **Upload the data tarball onto the volume now.** This is the item with real variance — ~2 GB
   over a home uplink is minutes to an hour, and it is the one step that cannot be parallelised with
   the 8-GPU work. Once it is on the network volume it survives pod termination, so the 8-GPU pod
   starts with the data already there and Part 2 begins directly at preflight.
4. `git clone` the repo onto the volume and build the environment there too, for the same reason:

   ```bash
   cd /workspace && git clone <repo> marsea && cd marsea
   mkdir -p runs && bash scripts/setup_runpod.sh 2>&1 | tee runs/setup.log
   source pod_env.sh
   df --output=target "$HF_HOME" . | tail -2    # ⚠️ both lines must name the SAME mount point
   ```

   `setup_runpod.sh` (renamed from `setup_nebius.sh`; the old name is a symlink) runs the machine
   checks, builds the venv, puts `HF_HOME` **on the volume**, downloads the backbone and HotpotQA
   into it, and verifies the 126 sets. Step 5 of the script skips the RULER clone once the data is
   present — so run it **after** the upload of item 3, or it will clone RULER and fetch ~200 Paul
   Graham URLs you do not need.

   ⚠️ Its data check is **informational — it prints and exits 0**. Read two of its lines yourself:
   `RULER sets: 126` and `sets WITHOUT the D0 length provenance: 0`. A non-zero second number means
   pre-fix data, which silently reintroduces the E3 length confound §1.5 exists to remove.

   ⭐ `source pod_env.sh` is the one new habit, and it carries to the 8-GPU pod: a shell that skips it
   points `HF_HOME` at the container disk and re-downloads 3 GB of backbone plus 600 MB of HotpotQA,
   then loses them at teardown.
5. **Terminate the 1-GPU pod.** The volume and everything on it persists; you stop paying for the
   GPU.

If any of the four checks fails, this is the hour to find out. The same failure discovered after the
8-GPU pod is up costs eight times as much per minute, with a 40-minute preflight queued behind it.

### 1.8 While the upload runs — the paper-side work that needs no results

- the unfunded-claims pass (~11 sentences; `PAPER_EDITS_e982f83.md` has them with line numbers);
- `run_eval.py:214` → `unit_cap_record(L)` with `L` parsed from `args.set` (review 7e05ad6 §2);
- the three bundle-only test failures → `pytest.skip` when the path is absent (§3 there).

---

# PART 2 — D1, stage 1, on the 8-GPU pod

Order is strict: every step gates the next.

### 2.1 Bring the 8-GPU pod up — on the volume §1.7 already filled

Specs, the network-volume ordering trap and the four verification commands are in
`marsea_runpod_playbook.md` Parts 0–1. Because §1.7 put the repo, the environment and the data on
the **network volume**, and the volume outlives the 1-GPU pod that wrote them, this step is a pod
creation and a re-verification — not a rebuild:

```bash
# on the new 8-GPU pod, attached to the SAME network volume
cd /workspace/marsea
source pod_env.sh                                        # ⭐⭐ FIRST, and in every pane — HF_HOME lives here
nvidia-smi --query-gpu=name,memory.total --format=csv    # ⭐ 8 × ~141 GB; see the name check below
git log -1 --oneline                                     # the volume kept it
.venv/bin/python -c "import torch; print(torch.cuda.device_count())"   # 8
tmux new -s marsea                                       # ⭐ always — an SSH drop must not kill the run
```

⚠️ If the environment is *not* there, the pod came up on a different volume — check before rebuilding.
A rebuild on the wrong volume costs an hour and leaves you running against the wrong data. If it is
genuinely missing: `mkdir -p runs && bash scripts/setup_runpod.sh 2>&1 | tee runs/setup.log`
(renamed from `setup_nebius.sh`, which is now a symlink; nothing in it is provider-specific).

⚠ **Read the GPU names once by eye, even though the script checks them.** `setup_runpod.sh` warns
on `NVL|PCIe`, on a smallest-GPU memory below 140 000 MiB and on a visible count other than 8.
**Plain `NVIDIA H200` at ~141 GB is correct.** `MARSEA_TOL_CAP` is measured per device in preflight
gate 4b, so the wrong part changes both the θ-window and the bandwidth assumption the budget rests on
— and a device class that differs between stage 1 and S2 makes the guard refuse outright.

### 2.2 ✅ Data — on the volume already (§1.2, §1.7). Verify, do not re-upload

The long CPU pole of the original plan is behind you, and so is the upload. Confirm the 126 sets and
both MuSiQue files are present and carry `measured_length_*` provenance (playbook §1.3), then go
straight to preflight. **Nothing here regenerates and nothing re-uploads** — at 8-GPU rates, an hour
of either is real money for zero science.

⭐ Note for later: E3 has five configs × 200 = **1000 examples generated**, and the queue evaluates
400 of them. `N16K` is therefore a dial you can turn *up* with no re-training if budget appears —
see §5.4.


### 2.3 Preflight — seven gates, ~40 minutes

```bash
bash scripts/preflight.sh 2>&1 | tee runs/preflight.log
```

| step | what | gate | if it fails |
|---|---|---|---|
| 1 | versions + `eager_attention_forward` signature | contract holds | pin `transformers`; the patch point has moved upstream twice |
| 2 | T0–T16 with `MARSEA_DEBUG=1` | all green | stop; read the failing test |
| 3 | `scripts/verify_all.py` | **0 violations**, exit 0 | stop |
| 4 | five memory profiles, gated at 72 GB | all pass | the dense-8K *training* probe is **recorded, not fatal**; `run_s2.sh` enforces it |
| **4b** | `check_unit_cap.py` — the device's unit cap and the θ-guard window | `window ≤ tol_cap` | **defined branch, see 2.4** |
| 5 | step time at 8K | **≤ 1.2 s/sequence** | see 2.5 — and read 2.5 even when it passes |
| 6 | the detector | some head clears the threshold | App. H's fallback case, not a run |

### 2.4 If gate 4b fails — a five-minute branch, not a stop

```bash
# the failure message prints the number; it is 4 x the window just measured
export MARSEA_TOL_CAP=<value>
bash scripts/preflight.sh 2>&1 | tee runs/preflight2.log
```

Export it in the environment of `run_s2.sh`, `run_evalsuite.sh` and `run_s0.sh` too — not just
preflight. The value is refused if it exceeds `L·ε/4` or 8× the window measured here, and
`unit_cap_record` writes it and its ceiling into every README and eval table. **Quote it in App. F's
device statistics**; the paper now has a paragraph for exactly this number.

### 2.5 ⭐⭐ The budget recompute — the most important half-hour of the ten days

Preflight step 5 writes `runs/step_time.json`. **Read it before you request the 8-GPU node.**

```bash
python scripts/time_step.py --L 8192 --layers 4 --mode chunked --head_block 2 --steps 5 --no_gate
cat runs/step_time.json
```

The programme is `2000 steps × 16 sequences × 17 runs`, so:

| measured s/seq | per arm×seed | 17 runs | + Phase A + eval |
|---|---|---|---|
| 0.4 | 3.6 h | 60 h | ~85 h |
| 0.6 | 5.3 h | 91 h | ~115 h |
| 0.8 | 7.1 h | 121 h | ~150 h |
| 1.0 | 8.9 h | 151 h | ~180 h |
| 1.19 (**still passes the gate**) | 10.6 h | 180 h | ~210 h |

**Decide here, not on D5:**

- **≤ 0.6 s/seq** — the full D-24 grid fits. Proceed as planned.
- **0.6–0.8** — proceed, but drop `EXTENDED_E9` ideas entirely and plan to cut B1/B4's single seeds
  if D5 runs late.
- **0.8–1.0** — cut Phase B to **1500 steps** for every arm (uniform, so the comparison stays
  matched) *or* drop B1 and B4. Do not do both.
- **> 1.0** — the gate may still pass. Take the D-23 lever deliberately: **two patched layers**,
  and note in the record that you invoked a step-time lever for a *budget* constraint.
  ⚠️ If you go to two layers, both the coreference site and `(l*, h*)` must remain in
  `PATCHED_LAYERS` or E7's column half becomes unmeasurable — "the top two by retrieval score" can
  silently drop the coreference column.

### 2.6 Phase A, then S0 — the gate on everything

✅ **The `+dirty` trap is closed (`b91a0a8`).** Five `runs/*.json` were tracked, `detector.json`
among them; S0 writes its licence into that file and `git_hash()` reads the whole tree, so every
README, checkpoint and eval table from S0 onward would have carried `<hash>+dirty`. `runs/` is now
ignored wholesale and a test asserts `git ls-files runs/` is empty. One command if you ever doubt it:

```bash
git ls-files runs/       # must print NOTHING
```

```bash
source pod_env.sh               # ⭐ every shell (§2.1)
bash scripts/run_s2.sh 8        # Phase A for seeds 0-2, then STOPS with exit 3 (by design)
bash scripts/run_s0.sh          # sweeps the PHASE-A weights, writes the licence into runs/detector.json
```

⭐⭐ **A consequence to check, not to fear: the pod now builds its OWN detector.** `runs/` is no
longer in the clone, so preflight step 6 / `run_s0.sh:18` actually run `run_detector.py` instead of
silently reusing the desktop's stale 2026-09-08 file (which had no `created`, `git` or
`spec_version`). That is the right outcome — App. H gets a detector with provenance — but it means
`patched_layers` is **measured fresh on the pod**. Compare it once:

```bash
.venv/bin/python -c "import json;d=json.load(open('runs/detector.json'));print(d['patched_layers'],(d['l_star'],d['h_star']))"
# the desktop measured [14, 19, 22, 23] with (l*, h*) = (19, 3)
```

A small reshuffle is ordinary — the detector ranks heads on the *unpatched* backbone and ties can
flip across devices — and nothing depends on it before Phase A, so a difference is not a stop. A
**wildly** different set means the backbone snapshot is not the one the desktop measured, and
`runs/backbone.json` (written by `setup_runpod.sh` step 3; the desktop's revision is `8faed761…`) is
how you tell in one line. Record whichever layers the pod chose — they are the ones the paper
describes.

**Read the licence:**

```bash
python - <<'PY'
import json; d = json.load(open("runs/detector.json"))
print("patched layers:", d["patched_layers"], "(l*,h*) =", (d["l_star"], d["h_star"]))
print("licence:", json.dumps(d.get("s0_licence"), indent=1)[:800])
PY
```

Two questions, two answers, both in that file:

- **(a) does a `T_j` construction route?** Per kind, ≥ 80 % of examples: `coref` (the key phrase's
  first mention → its later mentions) and `value` (a needle value → the row that emits it).
  On the dev box at base weights: `value` licensed at 60/336 heads, `coref` at 21, best `(14,5)`
  and `(14,3)` — **but S0 must be read on the Phase-A weights**, which is what `run_s0.sh` does.
- **(b) is there a replication pair?** `M(a) > m_j` at `a ∈ {0.5, 1.0}` on the `(l*,h*)` columns —
  the condition under which Cor. `capacity` bites on real data.

**Pivots:**

| outcome | what it means | do |
|---|---|---|
| both licensed | the planned programme | proceed |
| `value` only | E7's coreference column is unmeasurable | proceed; E7's column half is `value`-only, and say so in App. H |
| neither licensed | spec §12.3's pre-registered fallback | **proceed anyway** — E7 goes row-side, the interval-hit rate is reported on synthetic columns and the harness. E3 (load-bearing), Thms. 1–2, Prop. 14 and B2/B3 are untouched |
| no replication pair | Cor. `capacity` has no bite on these columns | ⚠️ report it. It is a real negative and the paper has language for it |
| the induction layer joins `PATCHED_LAYERS` | it **replaces** the weakest retrieval layer | Phase B will refuse the old Phase-A file — `rm runs/phaseA_seed*.pt` and re-run (~6 GPU-h), or `ALLOW_PHASE_A_LAYER_CHANGE=1` (recorded) |

### 2.7 S1 — one end-to-end run before committing the grid

```bash
.venv/bin/python scripts/run_train.py --arm marsea --seed 0 --L 4096 --total_steps 700 \
  --phase_a_steps 500 --mode chunked --detector runs/detector.json \
  --ruler_train "data/ruler/TRAIN_*" --musique data/musique/musique_ans_v1.0_train.jsonl \
  --hotpot_n 20000 --eval_ruler "data/ruler/QUICK_*/validation.jsonl" --out runs/s1 \
  2>&1 | tee runs/s1.log
```

**Read the first hour against the training procedure §11:**

- **Phase-B init**: `gap_5a` **exactly 0** (E forced empty reproduces the Phase-A path) and
  `rel_gap_5b < 5 %`; coverage calibrated to 0.050 per layer. *The run stops itself if either
  fails* — that is a real defect, not a tuning issue.
- **steps 500–600**: loss within 5 % of Phase A's final; coverage drifting from 5 % (either
  direction is fine, **0 % is not**); `τ_i` median 0.8–1.3; `k*` mostly 1–3; cap binding on
  ≲ 10 % of rows.
- the E8 block in `train_log.jsonl` every 50 steps, with **per-head** ρ (never pooled).

**Gate:** identities hold and coverage is non-zero. If S1 is unhealthy, the grid is not worth
launching — this is the cheap place to find that out.

---

# PART 3 — D2–D5, the training grid

### 3.1 Launch

```bash
export MARSEA_TOL_CAP=...            # only if gate 4b required it
bash scripts/run_s2.sh 8 2>&1 | tee runs/s2.log
```

✅ **Applied in `8d70239` (2026-09-17), from the ops-playbook review.** `LAUNCH_I` was file-scope and
`barrier()` never reset it, so Phase B started at `LAUNCH_I = 3` and the waves came out **5 / 8 / 4**
chunked with the dense arms as a fourth wave on GPUs 4–6 — wall-clock neutral, but it made every
published GPU→job mapping wrong and would have put a hand-launched dense arm on a card the launcher
was already using. `barrier()` now resets `LAUNCH_I=0`, and the three dense arms launch **first**,
inside the same round-robin. The 20 Phase-B jobs are now **8 / 8 / 4 — three waves**, ≈16 h instead
of ≈21 h at 0.6 s/seq, and a clean 4 × 5 on four GPUs. `tests/test_ops_playbook_review.py` runs the
real script with recording stand-ins and asserts the GPU sequence.

⚠️ **The one-line check on the pod:** the first eight `[gpu N] <tag>` lines must name eight
**distinct** GPUs, the first three of them the dense arms. If a GPU repeats, the tree on the pod
predates `8d70239`.

Defaults: `MODE=chunked L=8192 STEPS=2500 EVAL_EVERY=250 HEAD_BLOCK=2`. Eight independent
single-GPU processes, `CUDA_VISIBLE_DEVICES` per process, **no distributed training** (spec §18 —
`B = 1`, so there is no batch to split, and the fan-out program is global over the query axis).

It refuses to start unless every training source is present, and refuses **at startup** — not 36 h
in — if dense-8K training did not pass preflight's memory gate; `SKIP_DENSE_E9=1` drops the
uniform-quota and `K_ret` arms knowingly.

D-24's grid, **20 jobs**: 3 seeds on MarSea / B0 / B2 / B3, 1 seed on B1 / B4, three
chunked E9 ablations (`τ_i` pinned, key-alone relation, per-head), three dense E9 arms (uniform
quota, its matched control, `K_ret=64`). `EXTENDED_E9=1` is S4 — leave it off.

⭐ **The one arm worth adding if §2.5 leaves slack:** `--arm_kwargs {"no_nu":true}`, ~5 GPU-h. The
paper calls the field-argument ablation *"contribution (i)'s load-bearing test"* and names `no_nu`
as one of its three members; it sits in `EXTENDED_E9` and is currently off (unfunded-claims A-6).

### 3.2 Keep the pod

⚠️ **Do not stop and resume mid-programme.** A stopped RunPod pod releases its GPUs and may not get
them back, and the container disk is cleared. Run S2 → eval as one continuous block. The network
volume survives everything; the pod's own volume disk does not survive terminate.

⭐ **Copy off-platform daily and automate it** — a pod loss costs everything since the last copy:

```bash
rsync -avP root@<pod>:/workspace/marsea/runs/ ~/Dev/marsea/runs_runpod/
```

Checkpoints and parquet at minimum; `runs/*.json` are small and are the provenance for every gate.


### 3.3 What "done" looks like

Every arm×seed has `runs/<arm>_seed<k>/final.pt`. The queue's own exit is non-zero if any arm
failed. Check before starting evaluation:

```bash
ls runs/*_seed*/final.pt | wc -l        # expect 17 under D-24
```

---

# PART 4 — the realtime monitoring playbook ⭐

This is the part that decides whether you catch a bad run in an hour or in thirty.

### 4.1 One screen, every two minutes

`watch_marsea.sh` ships beside this plan. Put it in `scripts/` and give it its own tmux pane:

```bash
watch -n 120 bash scripts/watch_marsea.sh
```

It reads only files the run already writes, starts nothing, and prints: per-GPU utilisation and
memory; one line per training arm (step, loss, ρ, `τ_i` median, cap-binding rate, log age) with the
**pre-registered red flags** raised explicitly; the evaluation queue's progress, skipped arms and
any job log carrying a traceback; and the device gates as recorded.

⭐ **On one pod this is the whole story, and that is the point.** Everything it reads —
`runs/*_seed*/train_log.jsonl`, `runs/eval/`, the gate JSONs — is on the network volume, and
`nvidia-smi` shows all eight GPUs in one block, so a single pane is the entire programme. Its
"below 5% utilisation" warning means what it says: a GPU that is idle while jobs remain queued is a
crashed worker, not a lull. (On a two-pod split — §7 — the same pane on either pod still shows every
log, because both write to the same tree; only the `nvidia-smi` block is local, and
`ls runs/claims | wc -l` against 17 is then the cross-pod progress signal.)

⚠️ **Run it inside `tmux`, on the pod.** A RunPod web terminal closes with the browser tab, and a
dropped SSH session takes an untmuxed `run_s2.sh` with it — the jobs die, the `.done` files do not
appear, and the relaunch redoes them.

### 4.2 The signals, what each one means, and what to do

**From the console line** — `[B] step N loss X rho [...]` every 50 steps:

| signal | meaning | action |
|---|---|---|
| no new line for > 20 min | the process is dead or wedged | check `nvidia-smi`; the slot's log; restart — runs are idempotent |
| `E8 MISSING on layers [...]` | diagnostics pruned on that path | ⚠️ E8 is the degeneracy detector; if it stays missing, the arm's diagnostics are worthless |

**From `runs/<arm>_seed<k>/train_log.jsonl`** — the E8 block, per patched layer, per head:

| reading | threshold | meaning | action |
|---|---|---|---|
| `rho` / `rho_head` | **< 0.005 on every layer** | the arm **has become B0** — an empty relation is standard attention | stop the arm. It is not a MarSea run |
| `frac_Ecol_singleton` | **> 0.80** | it **trained out of Thm. 1** — cardinality on a singleton is product-form and the theorem says nothing | stop; this is App. E cost (a) and it is invisible to the τ variances |
| `var_tau_j` | **→ 0** | constant temperature; content-dependence gone | record; the E9 story changes |
| `tau_i_median` | outside **0.8–1.3** | the fan-in regime has drifted off the identity | watch; if `τ_i` is large, expect a row-recall deficit in E7 |
| `frac_rows_over_unit` | **> 10 %** | the cap is binding far more than healthy | watch; correlates with over-concentrated fan-out |
| `frac_rows_zero_mass` | **> 0** | rows ending with no mass — D-31 should prevent this | investigate; it was the reason position 0 is excluded |
| `truncation_flag` | `True` | `k*` approaching `K_ret` — the hierarchy is truncating | **no E7 number is meaningful** until `K_ret` is raised. `None` for the exact flat solve is correct, not a miss |
| `row_trigger_violation` | `True` | the row rate fell below the column-induced rate | **an implementation bug, not a finding** |
| `nonfinite_grads` | **> 0** | the NaN ladder has started | see 4.3 |
| `head_lr_scale` | **≠ 1.0** | a NaN already fired and the head LR was halved | this arm is on its second life; a third NaN ends it |

Quick reads without the watcher:

```bash
tail -f runs/marsea_seed0/train_log.jsonl | python -c "
import sys, json
for l in sys.stdin:
    r = json.loads(l)
    e8 = r.get('e8', {})
    print(r['step'], round(r['loss'],4), [round(v.get('rho',0),4) for v in e8.values()], 'nf', r.get('nonfinite_grads'))"
```

### 4.3 The NaN ladder — already automatic, do not intervene

1. first NaN → abandon the step, restore the last checkpoint with the **identical data order**, log
   `(step, micro, layer)`;
2. second at the same step → halve the head learning rate once;
3. third → stop that arm×seed and report it as a failure.

Restarts are idempotent: data cursor, RNG states, optimiser and scheduler all ride in the
checkpoint. **Your only job is to notice** — an arm that has spent its ladder is one you may need to
drop from the grid, which is a scope decision, not a debugging one.

### 4.4 When to stop the whole queue

Three conditions, and only three:

1. **Every** arm shows coverage < 0.5 % — the calibration is wrong, not the arms.
2. `verify_all.py` or a `MARSEA_DEBUG` invariant fails on the box after passing preflight — the
   device changed under you (a reschedule); re-run `check_unit_cap.py` before anything else.
3. You are past the D5 22:00 drop-dead with training unfinished — then it is a scope decision:
   cut B1/B4 (one seed each, least informative), then cut E9 arms, **never cut seeds on
   MarSea / B0**, which is where the comparison lives.

⚠⚠ **After the §3.1 reorder the cut is QUANTISED, and the decision point moved to launch.**
`run_s2.sh` is a barrier launcher: 20 jobs on 8 GPUs is 8 / 8 / 4, three waves. Dropping jobs saves
**nothing** until the total reaches 16 — 20 → 18 is still 8 / 8 / 2, still three waves, still
≈3× one arm's duration. So:

| jobs | waves on 8 GPUs | wall clock at 5.3 h/arm |
|---|---|---|
| 20 (full D-24) | 8 / 8 / 4 | ≈16 h |
| 18 (− B1, B4) | 8 / 8 / 2 | ≈16 h — **no saving** |
| **16 (− B1, B4, − 2 E9 arms)** | **8 / 8** | **≈11 h** |

Two consequences. First, cutting B1/B4 *alone* is a GPU-hour saving (≈11 GPU-h) and not a wall-clock
one — which is the opposite of what the ladder above implies. Second, the job list is hard-coded, so
the only clean place to take the wave is **at launch on D2** (§4.4 of the ops playbook has the two
`sed` edits). At D5 22:00 the lever that remains is killing what is running in the last wave, which
recovers only that wave's remainder. If the §2.5 budget recompute lands above 0.8 s/seq, decide the
16-job grid then, not on D5.

---

# PART 5 — D5–D7, evaluation

### 5.1 Launch

```bash
bash scripts/run_evalsuite.sh 8 2>&1 | tee runs/evalsuite.log
```

One command, one pod, eight GPUs — the same shape as S2 (§3.1). `MARSEA_DISPATCH` is **not set**:
with a single dispatcher process there is nothing to claim against, and `run_evalsuite.sh`'s own
`.done` test is sufficient. (`patch_multivm.py` stays shelved; §1.1. If the pod is ever lost and the
grid has to finish across two smaller pods, that patch is what makes it safe — do not hand-split the
job list instead.)

⚠️ Check `bash --version` on the pod before this runs. `run_evalsuite.sh` uses `wait -n -p`, which
needs **bash ≥ 5.1**; the failure mode is a syntax error at launch, after the queue has already
printed its plan.

Per arm × seed × job, resumable (`runs/eval/<tag>.done`), one job per GPU, each writing
`runs/eval/<tag>.parquet`, `<tag>_table.json` and `runs/eval/log_<tag>.txt` under its own tag.

It refuses to start without an S0 licence, without its data, without the paired MarSea checkpoints
the dense arms score against, or if preflight's `--paired` memory probe failed
(`SKIP_PAIRED_GATE=1` overrides). **An incomplete grid exits 2** with the list in
`runs/eval/SKIPPED_ARMS.txt`.

⭐ **Decide the `ALLOW_MISSING_ARMS` policy before you go to bed on D5.** If an arm NaN'd overnight,
the default refusal means the queue waits for you and you lose a shift. My recommendation:

```bash
ALLOW_MISSING_ARMS=1 bash scripts/run_evalsuite.sh 8
```

and read `SKIPPED_ARMS.txt` in the morning — the arms are named at the top *and* again at the end,
beside the tables they are missing from.

### 5.2 Watching it

```bash
find runs/eval -name '*.done' | wc -l                      # progress
grep -l -E "Traceback|CUDA out of memory" runs/eval/log_*.txt
```

- **OOM on the 16K jobs** is the expected failure: the paired-relation cache is 268 MB at 16K and
  keyed by `(layer, head)`. Lower `N16K` for that job rather than `--head_block`.
- A job that never writes is worse than one that fails — check the newest log's age.

### 5.3 The tables

```bash
.venv/bin/python scripts/collect.py --eval_dir runs/eval    # -> runs/eval/collected.{json,md}
```

Pools seeds (mean, std, n) **only when provenance agrees**, builds the E8 block per patched layer
and E3's coverage rate `|E_.j|/n` per stratum, and **prints each table's pre-committed decision rule
beside the numbers it governs** — so a run's README cannot omit it.

**Read them in this order:**

1. **E3** — the load-bearing one. Per `n`: the fraction of active columns with `δ_j > 0`; the
   interval-hit rate on those; MarSea's precision **on those against on all**.
   ⭐ *Flat precision on the `δ_j>0` columns beside a falling fraction of such columns is the
   theory's own prediction.* Flat precision on **all** columns would mean the relation, not the
   program, is doing the selection. Both are reportable; you pre-committed to whichever you see.
2. **E2** — the interval-hit rate stratified by `m`. Falsification is a hit rate *falling* with `m`.
   The comparator's shrinking gap is arithmetic, not failure.
3. **E4** — a gap near zero is the prediction; a *large* gap is the interesting negative.
4. **E7** — all five fidelity quantities, recall twice, and the interval-hit rate against B3.
5. **E8** — the degeneracy readings, reported either way, never acted on.

### 5.4 If budget appears — the one dial worth turning

`gen_data.sh` already generated **1000** E3 examples (5 configs × 200); the queue evaluates 400.
Raising `N16K` costs **evaluation time only, no retraining**, and buys back the statistical power
`--n 400` spent. Turn it up for **MarSea and B0 first** — they carry the headline comparison — and
only after §2.5's measurement says there is room.

---

# PART 6 — D8–D9, the paper

### 6.1 D8 morning — numbers in

`collected.md` → the tables. Every number that came from a script must be reproducible by a script
the paper ships (record §20.4).

### 6.2 D8 — the text pass, and it is worth more than another experiment

- the **~11 unfunded promises** — cut or qualify each (`PAPER_EDITS_e982f83.md`);
- ⭐ **E6**: decided NOT RUN (§1.4) — every sentence promising the ranked/instruction-hierarchy
  measurement must be cut or re-aimed at what E3/E7 support. This is now part of the ~11, not
  separate from it;
- **the equivalence band** from §1.6, in the E3 paragraph;
- **App. F's device statistics**: the kernel's `δ` *and* the θ-guard window, with
  `MARSEA_TOL_CAP` if it was set;
- `tab:compute` against what actually ran;
- B1/B4's single seed footnoted; the under-powered B2/B3 contrasts stated, not silently null.

### 6.3 D9 — build and submit

```bash
cd documents && pdflatex -interaction=nonstopmode marsea_iclr.tex   # x3
```

Expect **0 errors, 0 undefined refs, 0 undefined citations, 0 multiply-defined labels, 0 overfull**.
Check the changed pages in the **rendered PDF**, not the source (§0.5). The 9-page limit is deferred
(§30.0) until the content is complete — decide on D8 whether to spend D9 on compression or let it
run to 10 pp for the camera-ready.

Guard every `.tex` edit with the §0.3b probes: `patch_iclr_cap.py` and `patch_iclr_window.py` both
refuse to write on a draft that is not the current one.

---

## 7. The pivot table, in one place

| when | signal | pivot |
|---|---|---|
| D1 | 4b window > `tol_cap` | `MARSEA_TOL_CAP = 4 × window`, re-run, quote in App. F |
| D1 | step time > 0.8 s/seq | Phase B to 1500 steps, or drop **four** jobs (B1, B4 and two E9 arms) to 16 and buy a whole wave — not both. Dropping only B1/B4 buys GPU-hours, not wall clock (§4.4) |
| D1 | step time > 1.0 s/seq | two patched layers (keep both D-9a sites), record it as a *budget* lever |
| D1 | no `T_j` construction licensed | E7 column half "not measurable on this backbone"; row side carries E7 |
| D1 | no replication pair | report it — a real negative with language already in the paper |
| D1 | S1 sanity 5(a)/5(b) fails | **do not launch the grid**; it is a defect |
| **D0** | ✅ resolved: **RunPod, 8 × H200 on demand** | Nebius had quota 32 but 8-GPU on-demand capacity 0 in every region, and only preemptible presets. RunPod sells the pod on demand — no fleet, no preemption handling, no untested resume path (§1.1) |
| **D0** | 8 × H200 not available in the region your network volume is in | ⚠️ **check availability before creating the volume** — the volume pins the region and cannot be moved. If it is already created in the wrong region, make a second one; they are cheap next to a lost shift |
| D1 | the 8-GPU pod is unavailable at launch time | take **4 × H200** and run `run_s2.sh 4`. At 0.6 s/seq the grid is ~27 h instead of ~16 h (5 waves vs 3, after the §3.1 reorder) — still inside D2–D5. Do **not** wait for 8 |
| D3–5 | the pod is lost mid-grid | relaunch on the same network volume. ⚠️ **Corrected 2026-09-17:** `run_s2.sh` has **no** `.done` markers (those are the eval queue's) — resume is per arm through `train.py`'s `last.pt`, at **250-step** granularity, and the Phase-B init is *not* re-run. See the ops playbook §0.1 and §B of its review |
| D3–5 | you need two pods to finish in time | apply `patch_multivm.py` and set `MARSEA_DISPATCH=1` on **both** (§1.1). Never hand-split the job list |
| D2 | fewer GPUs than 8, of any shape | the grid simply takes longer — it is 20 independent single-GPU jobs. At 0.6 s/seq and 5.3 h per arm: 8 GPUs ≈ 16 h (3 waves), 4 ≈ 27 h (5), 2 ≈ 53 h (10). **Two GPUs still finishes inside D2–D5**; one does not |
| D2 | no GPU of **either** type at all | run MarSea/B0/B3 × 3 seeds on whatever single GPU exists (~9 runs, ~3 days) — scope B+ minus B2 |
| D1 | stage 1 on a different device class than S2 will use | **re-run `preflight.sh` on the 8-GPU node** — the `MARSEA_TOL_CAP` guard will otherwise refuse a window measured elsewhere |
| D3–5 | an arm exhausts the NaN ladder | drop it; never drop a MarSea or B0 seed |
| D5 22:00 | training unfinished | cut B1/B4, then E9 arms, in that order — but see §4.4: the saving is **quantised at 16 jobs**, so this is a D2-at-launch decision, not a D5 one |
| D5 | an arm missing at eval time | `ALLOW_MISSING_ARMS=1`, read `SKIPPED_ARMS.txt` in the morning |
| D7 22:00 | tables incomplete | `collect.py` on what exists; the queue names what is missing |
| **D8 noon** | anything | **stop the science.** The rest of the time is the text |

---

## 8. What is already decided, so nobody relitigates it at 3 a.m.

- Venue **ICLR 2027**; TMLR is the fallback, not the plan (§0.2).
- **H200 over H100** where available (§1.1), same device class for stage 1, stage 2 and every arm.
- **No job packing this week** — the lever is measured and recorded, not taken (§1.1b).
- Scope **A on D-24's reduced programme**, gated on S0 / S1 — Luke, 2026-09-16.
- **Everything runs on one RunPod 8 × H200 on-demand pod** — Luke, 2026-09-17. Stage 1, the
  training grid and the evaluation queue all use the same pod and the same device class, which is
  also what keeps the `MARSEA_TOL_CAP` window valid across all three (§1.1, §2.3).
- **Nebius is closed**, not deferred: quota was granted (32 H200) but on-demand 8-GPU capacity was 0
  in every region and only preemptible presets were offered. Preemptible would have required a
  supervisor loop, a boot unit, a SIGTERM handler and an **untested** resume path, against an
  unpublished preemption rate — four new failure modes bought with the week's remaining time.
- **`patch_multivm.py` is written and tested but NOT applied** — it is the two-pod contingency
  (§7), not the plan. Applying it on a single pod adds a claim protocol with nothing to claim
  against.
- **The network volume is created before the pod, and it pins the region** — Luke, 2026-09-17.
  Data, repo and environment live on it, so a pod can be terminated and relaunched without
  re-uploading anything.
- **E6 / IHEval: NOT RUN** — Luke, 2026-09-16 (§1.4). The schema does not carry E6's x-axis; the
  paper states it was not run. No code change, and the path stays empty.
- `EXTENDED_E9=1` is **off**. Those arms are S4 and not in the budget.
- **No distributed training** — eight independent single-GPU processes (spec §18), which is exactly
  why **eight 1-GPU VMs substitute for one 8-GPU VM with no scientific cost** (§1.1a).
- E3 at **16K**, constant across the sweep, chunked path; E2/E4 at 8K.
- **B5 is teacher-forced only** (D-29); its task-level cell reads "n/a by design".
- The causal/prefix form is native design and is **never** raised as unimplemented work (§0.2 item 3).
