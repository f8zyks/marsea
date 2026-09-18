# Review — `acda7b6` (setup rewrite + rounds 1–3 fixes)

*2026-09-17. Against the working tree as staged: `scripts/setup_runpod.sh`, `scripts/run_s2.sh`,
`marsea/train.py`, `tests/test_ops_playbook_review.py`, `.gitignore`, `scripts/handoff.sh`, plus
`marsea/backbone.py` for the model id. Bash `set -e` semantics at `run_s2.sh:138` tested empirically
rather than reasoned about.*

---

## Verified fixed

| item | how I checked |
|---|---|
| round-2 §1, durability | `save_checkpoint` opens `tmp` explicitly, `torch.save(…, f)`, `f.flush(); os.fsync(f.fileno())`, `os.replace`, then opens the parent and `os.fsync`es the directory fd, with an `OSError` pass for filesystems that refuse. This is exactly right, including the guard |
| round-2 §3, tests | parameterised on 8 and 4 GPUs with the assertion written as `[i % ngpu for i in range(20)]`; the stand-in holds its lock 0.5 s; the checkpoint test is now behavioural — `torch.save` writes a few bytes and raises, the old `last.pt` must still load, no `step*.pt` appears, a `last.pt.tmp*` is left, and the good path is re-run with `os.fsync` instrumented |
| round-2 §5, `SKIP_DENSE_E9` | now honoured after the gate, which still prints its verdict. `run_s2.sh:138` is at top level with statements after it, so the `[ … ] && { … }` idiom does **not** trip `set -e` (I tested: mid-script it is safe; as the last statement of a script or a function it returns 1) |
| round-1 §A, launcher | unchanged and still correct; I re-simulated |
| the setup rewrite | `bash -n` clean; the `../../../../../../` in step 5 resolves to the repo root (six levels, correct); `pod_env.sh` is gitignored so it cannot dirty the tree; the grid counts in step 4 (`TRAIN 69, QUICK 5, DET 1, E2 15, E3 15, E3pad 3, E3depth 15, E4 3` = 126) match the verified inventory exactly; the prefetched model id matches `backbone.py:410`'s default |

The setup rewrite is the right call and the five problems you found with the old script were real.
Below is what I found in the new one.

---

## ⭐⭐ 1. The SXM check fires on the hardware you want, and so will be ignored

```bash
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -qiv "SXM"; then
  WARN+=("a GPU is not SXM …: the playbook says redeploy")
```

This requires the substring `SXM` to be **present** in the product name. NVIDIA does not put it
there. An H100 SXM5 reports `NVIDIA H100 80GB HBM3`; an H200 SXM reports `NVIDIA H200`. The variants
you are trying to reject are the ones that name themselves: `NVIDIA H200 NVL`, `NVIDIA H100 PCIe`.

So on a correct 8×H200 SXM pod the script will print eight lines of "a GPU is not SXM — the playbook
says redeploy". You saw it fire on the dev box and read that as working; it will fire on the target
too. The cost is not the noise, it is that after the third pod the operator stops reading that
warning — and then a genuine H200 NVL, which is the one thing this check exists for, goes through
silently and invalidates §2.5's bandwidth arithmetic.

Allow-lists on vendor strings fail this way; deny-lists do not:

```bash
BAD=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -iE "NVL|PCIe" || true)
[ -z "$BAD" ] || WARN+=("non-SXM part(s): $BAD -- the memory-bandwidth assumption behind the budget does not hold")
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
[ "${MEM:-0}" -ge 140000 ] || WARN+=("smallest GPU is ${MEM} MiB, below the 141 GB the profile assumes")
```

⚠️ I cannot run `nvidia-smi` on an H200 from here, so **do not take my string on faith either**.
The honest form is: print the name, and have a human read it once on the pod. Both playbooks now say
plain `NVIDIA H200` at ~141 GB is correct and anything carrying `NVL` or `PCIe` is not.

`tests/test_ops_playbook_review.py` asserts `"SXM" in src`, so it pins the current check — the
deny-list version keeps that assertion true, but check the test reads the way you want.

## ⭐⭐ 2. Still open from the cutover review, and S0 makes it permanent tonight

`.gitignore` is byte-identical to before this round — `runs/**` with `!runs/*.json`, and `documents/`
still ignored. Two consequences that I raised and that have not been actioned:

**(a) `runs/detector.json` and the `+dirty` stamp.** `run_s0.sh` runs
`run_s0_sweep.py --update_detector runs/detector.json`, which writes into that file, and `git_hash()`
runs `git status --porcelain` over the **whole repo**. If that file is tracked, then from S0 onward
every README, every checkpoint and every `<tag>_table.json` records `<hash>+dirty` — twenty arms,
ninety-six eval jobs, and an appendix citing a hash nobody can check out. The function's own
docstring is the argument: *"a provenance hash that names a commit the numbers did not come from is
worse than none."*

One command decides it, and it has to happen **before `run_s0.sh`** — which is tonight:

```bash
git ls-files runs/       # must print nothing
git rm --cached runs/*.json && printf 'runs/\n' >> .gitignore && git commit -am "runs/ is output"
```

**(b) the pod has no playbook, and two tests are still dead.** You confirmed it: *"the same 2
runbook-file skips as before"*. Those two are not skipping by design —
`test_review_e6ca44f.py:39` and `test_review_4f754ed.py:51` assert that `RUNBOOK_nebius.md`
documents `MARSEA_TOL_CAP`, `L·ε/4` and "8× the window measured". They are the doc–code guard on
gate 4b, the fiddliest branch of D1, and they have been inert since the runbook moved under
`documents/`. Moving that one file back into the tracked tree revives both and puts the gate-4b
recovery on the pod where it is needed. (Whether the *whole* of `documents/` should be tracked is a
separate call — `RUNBOOK_nebius.md` is not.)

## ⭐ 3. The data check cannot fail, and one of its two numbers is a correctness condition

Step 4 prints and exits 0 — always. Two different things are being reported and only one of them is
a count:

- `RULER sets: N (plan: 126)` and the per-grid shortfall — a completeness check. Informational is
  right here, because on the staging pod you may legitimately run setup before the upload finishes.
- `sets WITHOUT the D0 length provenance` — **not** informational. A non-zero value means pre-fix
  sets, which silently reintroduce the E3 length confound that `patch_ruler_length.py` and a whole
  day of regeneration removed. There is no situation in which proceeding is correct.

Suggest: exit non-zero on `noprov > 0` unconditionally, and add `REQUIRE_DATA=1` (set it on the
8-GPU pod) which additionally fails on a short count. Both playbooks now tell the operator to read
those two lines by eye; a gate is better than a habit.

## ⭐ 4. `HF_HOME` on the volume depends on the volume being at `/workspace`

```bash
case "$PWD" in /workspace/*) HF_HOME="/workspace/hf_cache";; *) HF_HOME="$HOME/.cache/huggingface";; esac
…
case "$HF_HOME" in /workspace/*) ;; *) [ -d /workspace ] && WARN+=("…not under /workspace…");; esac
```

If the network volume mounts anywhere other than `/workspace` — RunPod's mount point is
configurable, and serverless uses `/runpod-volume` — the fallback puts 3.6 GB of cache on the
container disk, **and the warning does not fire**, because it is conditioned on `/workspace`
existing. Silent, and discovered at teardown.

The provider-neutral test is whether the cache and the repo are on the same filesystem, which is
what you actually mean:

```bash
[ "$(stat -c %d "$HF_HOME")" = "$(stat -c %d .)" ] || WARN+=("HF_HOME=$HF_HOME is on a DIFFERENT filesystem from the repo: if the repo is on the network volume, this cache is not, and a terminated pod loses it")
```

and the default becomes `$(dirname "$PWD")/hf_cache` with that check deciding whether to warn. One
line, and it is right on every provider and every mount point.

## ⭐ 5. The backbone has no revision pin and nothing records which snapshot ran

`backbone.py:410` is `from_pretrained(name)` and setup's `snapshot_download(m, …)` — neither passes
`revision`. `README.json` records `config` (which carries the *name* `Qwen/Qwen2.5-1.5B`), `git`,
`spec_version` and the detector stamp; nothing carries the model's commit sha.

This is not a launch risk — Qwen2.5-1.5B is long-published and will not move in eight days. It is a
**paper** gap, and it is specific to this paper: the detector chose `patched_layers [14, 19, 22, 23]`
by measuring *this* checkpoint's retrieval heads, and every claim is about what MarSea does to
*those* heads. "Qwen2.5-1.5B" does not identify them.

Cheapest sufficient fix, in setup, no behaviour change:

```python
p = snapshot_download(m, allow_patterns=[...])
json.dump({"model": m, "snapshot_path": p, "revision": pathlib.Path(p).name},
          open("runs/backbone.json", "w"), indent=1)     # the path ends in snapshots/<sha>
```

then add `runs/backbone.json` to §10's end-of-run checklist beside `detector.json`. A real
`revision=` pin in `load_backbone` is the thorough version and can wait until after the run.

## 6. Small

- **The GPU count is computed and never checked.** `NGPU=$(… | wc -l)` is printed and dropped. A pod
  that comes up with 7 visible GPUs passes setup, passes preflight, and fails mid-wave when
  `CUDA_VISIBLE_DEVICES=7` finds nothing. `[ "$NGPU" = 8 ] || WARN+=("$NGPU GPUs, not 8 — run_s2.sh
  \$NGPU and run_evalsuite.sh \$NGPU must match")` turns a confusing failure into a startup line.
- **Duplicate message.** With a failed verdict *and* `SKIP_DENSE_E9=1` you now get both
  `"…will NOT be trained"` (the `elif`, still needed to avoid the `exit 1`) and `"dropped by
  request"` (line 138). Cosmetic.
- **The fsync assertion could be tighter.** `len(fsyncs) >= 1` passes if the directory fsync is lost
  in a later edit; `>= 2` pins both. (With the `OSError` fallback, `>= 2` is only safe on a
  filesystem that supports directory fsync — worth a `getattr`-style skip if you care.)
- **`run_s2.sh:138`'s idiom is safe only positionally.** `[ … ] && { … }` under `set -e` does not
  exit mid-script, but returns 1 as the status of a function or of a script's last line. It is
  neither here. If it ever moves, make it an `if`.
- **`handoff.sh`** ships `marsea scripts tests` and its manifest covers the whole tree, so the new
  script and the symlink are both in the bundle. Confirmed by reading it; no change needed.

---

## Both of my documents are updated

- **`marsea_runpod_playbook.md`** — §0.2 and §1.1 renamed to `setup_runpod.sh`; §1.1 rewritten around
  the new step 0 (`mkdir -p runs`, `source pod_env.sh` in every shell and pane, the `df --output=target`
  check from §4); §1.2 rewritten as "the machine checks the script runs, and what they mean", with
  the product-name correction from §1; Part 3 gains the `git ls-files runs/` gate before `run_s0.sh`;
  Part 6's job count corrected 98 → 96.
- **`marsea_run_plan_10day.md`** — §1.1's four checks now point at the script and carry the
  product-name correction; §1.7 item 4 is the full step-0 sequence with the RULER-skip ordering
  (run setup **after** the upload or it clones RULER and fetches ~200 URLs you do not need) and the
  instruction to read the two data-check lines by eye; §2.1's bring-up uses `source pod_env.sh` first
  and `.venv/bin/python`; §2.6 opens with the pre-S0 `git ls-files runs/` gate and the note that the
  stale detector will be reused silently unless it is dropped.

## Priority for tonight

1. §2(a), `git ls-files runs/` — one command, and it is the difference between an appendix that cites
   a commit and one that cites `<hash>+dirty`. Must precede `run_s0.sh`.
2. §1, the SXM string — five minutes, and it decides whether a warning you will see eight times per
   pod means anything.
3. §3 and §4 — two `||` clauses in setup; both convert a silent wrong state into a printed one.
4. §2(b) and §5 — worth doing, neither blocks D1.
