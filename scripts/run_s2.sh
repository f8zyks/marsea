#!/usr/bin/env bash
# S2 -- the training grid (record Sec. 44 / D-24) as a queue of single-GPU processes: CUDA_VISIBLE_DEVICES per process,
# no distributed training.  Phase A runs once per seed and EVERY arm forks that one file.
#
#   bash scripts/preflight.sh                 # FIRST: tests, harness, memory and step time on the target GPU
#   bash scripts/run_s2.sh 8                  # Phase A for seeds 0-2, then STOPS (exit 3) until S0 has run
#   bash scripts/run_s0.sh                    # the gate (spec Sec. 14): writes the licence into runs/detector.json
#   bash scripts/run_s2.sh 8                  # again: Phase A is reused, Phase B runs
#   (if S0 changed PATCHED_LAYERS -- the induction layer joining is its expected outcome -- Phase B refuses the
#    Phase-A files: delete runs/phaseA_seed*.pt to retrain them (~6 GPU-h) or ALLOW_PHASE_A_LAYER_CHANGE=1 to fork
#    them anyway, recorded in each run README)
#
# IMPLEMENTATION MODE AND MEMORY.  Measured by differencing 1/2/4/8 patched layers at T <= 2048 and fitting
# activations = b*T + q*T^2 + (n-1)*r*T^2 (runs/memory_model.json).  The binding term is the patched layers' T^2
# tensors, and gradient checkpointing means the peak is set by ONE layer at a time -- so dropping 4 patched layers to 2
# is NOT a halving.  After the 2026-09-12 efficiency fixes the per-patched-layer term at 8K fell from 124 GB to ~10 GB
# and 8K with four patched layers extrapolates to ~27-32 GB: it fits one 80 GB H100, and neither pre-committed
# fallback (4K, or two patched layers) is needed.  16K extrapolates to ~63-83 GB, i.e. AT the gate -- preflight.sh
# decides it on the real device.
#   --head_block k   runs k Q-heads at a time, each block separately checkpointed.  Every program is per Q-head
#                    (spec Sec. 6.2), so it is arithmetically identical on both paths and divides the T^2 terms.
#                    "Identical" to fp64 rounding, not bitwise: blocking changes the GEMM shapes and, for the
#                    per-head (E9) parameters, the gradient reduction order -- measured at 4e-15 on values and 4e-11
#                    on gradients in fp64, so expect ~1e-7 relative drift in fp32.  This is the lever; on by default.
#   MODE=chunked     the chunked path's cost is proportional to the relation's coverage rather than being linear in T
#                    (an earlier claim that it was linear came from a two-point fit and was wrong); it is a
#                    constant-factor saving, and D-27 requires it for E3 at 16K in any case.
#   Exception: the E9 uniform-quota arm CANNOT run chunked (the uniform quota is a reduction over all key chunks --
#   read-through F-3), so it trains dense at MODE_UNIFORM_L with a control at the same length.
# RUN scripts/preflight.sh ON THE TARGET GPU FIRST: it re-measures all of this at the real lengths, with no
# extrapolation lever arm, and gates at 72 GB.
set -e
cd "$(dirname "$0")/.."
NGPU=${1:-8}
PY=${PY:-.venv/bin/python}
# NGPU = 0 or a non-integer made `LAUNCH_I % NGPU` a division by zero / the slot loop a no-sleep busy wait (review 9bacef9 E)
[[ "$NGPU" =~ ^[1-9][0-9]*$ ]] || { echo "run_s2.sh: NGPU must be a positive integer (got '$NGPU')" >&2; exit 1; }
L=${L:-8192}; STEPS=${STEPS:-2500}                 # reduced programme (D-24): Phase A 500 + Phase B 2000
MODE=${MODE:-chunked}; HEAD_BLOCK=${HEAD_BLOCK:-2}     # HEAD_BLOCK=0: every Q-head at once (no blocking)
CHUNK=${CHUNK:-1024}                               # the chunked path's key-chunk width.  It used to be set here and
                                                   # passed nowhere (review e982f83 I); run_train has --chunk now.
                                                   # Pod 1 (H200, 8K): HEAD_BLOCK=0 CHUNK=2048 is 17 % faster than
                                                   # 2 / 1024 at 72 GB rather than 56 GB peak -- same numbers.
UNIFORM_L=${UNIFORM_L:-$L}                         # the dense-only E9 arms' length: the SAME as every other
                                                   # arm, so they are comparable and the eval queue's 8K sets match.
                                                   # Whether dense 8K training FITS is measured and gated by
                                                   # preflight.sh step 4 (runs/preflight_train_dense.json); the
                                                   # "~27 GB" this comment used to quote was extrapolated from
                                                   # 2048/4096, and nothing gated it (review e982f83 B-4).
DATA_RULER=${DATA_RULER:-"data/ruler/TRAIN_L${L}_*"}   # the pool AT THE TRAINING LENGTH.  With TRAIN_L4096_* beside
                                                   # TRAIN_L8192_* (the 4K lever, 2026-09-18) a length-blind glob
                                                   # trains on a mixture of lengths at 8K (the 8K sets are dropped
                                                   # at 4K: mix.py never truncates) -- a recipe change nothing flags
DATA_MUSIQUE=${DATA_MUSIQUE:-data/musique/musique_ans_v1.0_train.jsonl}
HOTPOT_N=${HOTPOT_N:-20000}
EVAL=${EVAL:-"data/ruler/QUICK_L${L}_*/validation.jsonl"}   # the quick eval at the training length, likewise
EXTENDED_E9=${EXTENDED_E9:-0}                      # 1 adds the paper's further field-argument ablations (S4, not D-24)
EVAL_EVERY=${EVAL_EVERY:-250}                      # the training procedure's cadence.  500 was tried for cost (review
                                                   # e982f83 H); measured it is 3-6 % of a job, <1.5 % of the grid, and
                                                   # the procedure is a pre-registration artefact (review e16a843 I).
# S0 may change PATCHED_LAYERS after Phase A is saved; train.py then refuses to fork it (review e16a843 J).  Either
# delete runs/phaseA_seed*.pt and re-run (Phase A retrains, ~6 GPU-h), or set ALLOW_PHASE_A_LAYER_CHANGE=1 (recorded).
PA_FLAG=""; [ "${ALLOW_PHASE_A_LAYER_CHANGE:-0}" = "1" ] && PA_FLAG="--allow_phase_a_layer_change"
COMMON="$PA_FLAG --L $L --total_steps $STEPS --mode $MODE --head_block $HEAD_BLOCK --chunk $CHUNK --detector runs/detector.json \
        --ruler_train $DATA_RULER --musique $DATA_MUSIQUE --hotpot_n $HOTPOT_N --eval_ruler $EVAL --eval_every $EVAL_EVERY"
mkdir -p runs

# ---------------------------------------------------------------- the training mixture must be complete BEFORE anything
# MuSiQue was silently omitted when its file was absent and HotpotQA's failure was caught and printed, so an arm could
# train on a different mixture from the procedure's (review e982f83 I).  build_training_sources now raises too; this
# refuses before a GPU is touched rather than inside every job.
if ! $PY - "$DATA_RULER" "$DATA_MUSIQUE" "$HOTPOT_N" "$EVAL" <<'PYEOF'
import glob, pathlib, sys
ruler, musique, hotpot_n, ev = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
bad = []
rf = [f for pat in ruler.split() for f in glob.glob(pat if pat.endswith(".jsonl") else pat + "/validation.jsonl")]
print(f"RULER training files: {len(rf)}")
if not rf: bad.append(f"no RULER training files match {ruler}")
if not pathlib.Path(musique).exists(): bad.append(f"MuSiQue training file {musique} is missing")
if hotpot_n:
    try:
        sys.path.insert(0, ".")
        from marsea.data.qa import load_hotpot
        print("HotpotQA train rows:", len(load_hotpot("train", n=10)))
    except Exception as e:
        bad.append(f"HotpotQA cannot be loaded: {e}")
if not glob.glob(ev): bad.append(f"no quick-eval files match {ev}")
for b in bad: print("REFUSING TO START:", b)
sys.exit(1 if bad else 0)
PYEOF
then exit 1; fi

# ---------------------------------------------------------------- Phase-A files vs the detector's layers, at STARTUP
# One message here instead of three tracebacks in runs/log_phaseA_seed*.txt (review 7814665 F-2)
if [ "${ALLOW_PHASE_A_LAYER_CHANGE:-0}" != "1" ] && ls runs/phaseA_seed*.pt >/dev/null 2>&1; then
  if ! $PY - <<'PYEOF'
import glob, json, sys, torch
try:
    det = json.load(open("runs/detector.json"))["patched_layers"]
except Exception as e:                       # unguarded, this was a traceback where every other gate prints one line
    print(f"REFUSING TO START: cannot read patched_layers from runs/detector.json ({e}) -- run scripts/preflight.sh"); sys.exit(1)
bad = []
for f in sorted(glob.glob("runs/phaseA_seed*.pt")):
    st = (torch.load(f, map_location="cpu", weights_only=False).get("detector") or {}).get("patched_layers")
    if st is not None and [int(x) for x in st] != [int(x) for x in det]:
        bad.append(f"{f}: trained with {st}")
if bad:
    print("REFUSING TO START: S0 changed PATCHED_LAYERS to", det, "after Phase A was saved:"); print("\n".join("  " + b for b in bad))
    print("  retrain:  rm runs/phaseA_seed*.pt && bash scripts/run_s2.sh   (~6 GPU-h)")
    print("  or fork anyway (recorded):  ALLOW_PHASE_A_LAYER_CHANGE=1 bash scripts/run_s2.sh")
    sys.exit(1)
PYEOF
  then exit 1; fi
fi

# ---------------------------------------------------------------- the dense E9 arms' memory verdict, at STARTUP
# It was read at the end of the queue, ~36 h of training later (review e16a843 F-1).  preflight.sh records it; if dense
# training at UNIFORM_L did not pass, refuse now unless the operator knowingly drops those arms.
DENSE_E9_OK=0
if $PY - "$UNIFORM_L" <<'PYEOF'
import json, sys
L = sys.argv[1]
try:
    r = json.load(open("runs/preflight_train_dense.json"))
except Exception as e:
    print(f"dense E9 arms: runs/preflight_train_dense.json unreadable ({e}) -- run preflight.sh"); sys.exit(1)
g = r.get("gate") or {}
if not g.get("passed") or str(L) not in {str(t) for t in r.get("targets", [])}:
    print(f"dense E9 arms: dense training at L = {L} has not passed preflight's memory gate ({g})"); sys.exit(1)
print(f"dense training at L = {L}: gate passed ({g.get('worst_GB')} GB <= {g.get('gate_GB')} GB)")
PYEOF
then
  DENSE_E9_OK=1
elif [ "${SKIP_DENSE_E9:-0}" = "1" ]; then
  :                                              # the drop is announced once, below
else
  echo "REFUSING TO START: the dense E9 arms are not memory-gated (above).  Fix it, or run with SKIP_DENSE_E9=1 to" >&2
  echo "  drop the uniform-quota and K_ret ablations knowingly." >&2
  exit 1
fi
# the flag means what its name says whatever the verdict: it was consulted only when the gate had FAILED, so with a
# passed verdict the dense arms ran regardless (ops playbook review 2, item 5).  After the gate, which still prints
# its verdict for the record.
if [ "${SKIP_DENSE_E9:-0}" = "1" ]; then       # an `if`, not `[ ] && { }`: that idiom returns 1 under set -e if it ever ends a function
  DENSE_E9_OK=0; echo "SKIP_DENSE_E9=1: the uniform-quota, its control and the K_ret E9 arms are dropped by request"
fi

# ---------------------------------------------------------------- Phase A: once per seed, the FULL 500 steps, then stop
# One process per GPU at a time.  `s % NGPU` alone put all three seeds on device 0 when NGPU = 1 -- three concurrent
# trainers on one card, i.e. a guaranteed OOM on the single-GPU configuration preflight.sh sizes for.
# A bare `wait` returns 0 whatever the jobs did, so `set -e` never fired and the queue ran its whole Phase-B section
# after every arm had crashed.  Each PID is waited on individually and its exit status recorded (review 30372ae D).
LAUNCH_I=0; PIDS=(); PTAGS=(); FAILED_JOBS=()
barrier() {
  local i
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      FAILED_JOBS+=("${PTAGS[$i]}"); echo "!! FAILED: ${PTAGS[$i]}  (runs/log_${PTAGS[$i]}.txt)" >&2
    fi
  done
  PIDS=(); PTAGS=()
  # a drained wave starts the GPU round-robin at 0 again.  LAUNCH_I used to carry over: Phase A's three launches
  # left it at 3, so wave 1 of Phase B held FIVE jobs on GPUs 3-7, the last chunked wave four, and the dense arms
  # landed on GPUs 4-6 -- 5 / 8 / 4 / 3 instead of the 8 / 8 / 4 the grid is sized for (ops playbook review A).
  LAUNCH_I=0
}
launch() {                                   # launch "<log tag>" <command...>: round-robins GPUs, barriers when full
  local tag="$1"; shift
  local gpu=$((LAUNCH_I % NGPU)); LAUNCH_I=$((LAUNCH_I + 1))
  echo "[gpu $gpu] $tag"
  CUDA_VISIBLE_DEVICES=$gpu "$@" > "runs/log_${tag}.txt" 2>&1 &
  PIDS+=($!); PTAGS+=("$tag")
  if (( LAUNCH_I % NGPU == 0 )); then barrier; fi
}
for s in 0 1 2; do
  launch "phaseA_seed$s" $PY scripts/run_train.py --arm B0 --seed $s $COMMON --phase_a_only
done; barrier
[ ${#FAILED_JOBS[@]} -eq 0 ] || { echo "Phase A had ${#FAILED_JOBS[@]} failed job(s); aborting" >&2; exit 1; }
for s in 0 1 2; do [ -f runs/phaseA_seed$s.pt ] || { echo "Phase A seed $s missing; aborting"; exit 1; }; done
echo "Phase A complete for seeds 0 1 2"

# ---------------------------------------------------------------- S0: the gate on everything (spec Sec. 14)
# run_s0.sh needs the Phase-A file, which only this script produces -- so following the documented order used to exit 1
# at S0, and the natural recovery (run this script) trained the whole grid with S0 never run (review e982f83 I).  Phase
# B now refuses until S0 has written its licence into the detector; re-running this script reuses Phase A (train.py
# verifies an existing Phase-A checkpoint instead of retraining it).
if ! $PY -c "import json,sys; sys.exit(0 if 's0_licence' in json.load(open('runs/detector.json')) else 1)"; then
  echo "=== STOP: Phase A is done, S0 has not been run.  Next:" >&2
  echo "      bash scripts/run_s0.sh        # writes the S0 licence into runs/detector.json" >&2
  echo "      bash scripts/run_s2.sh $NGPU   # again: Phase A is reused, Phase B starts" >&2
  exit 3
fi

# ---------------------------------------------------------------- Phase B: the arms (D-24: 3 seeds on the decisive four)
JOBS=()
for s in 0 1 2; do for arm in B0 marsea B2 B3; do JOBS+=("--arm $arm --seed $s"); done; done
JOBS+=("--arm B1 --seed 0" "--arm B4 --seed 0")
# E9 (D-24): tau_i pinned at 1; inherited vs uniform quota; pairwise vs key-alone relation.  Every one forks the SHARED
# Phase-A file (--phase_a_ckpt), never its own.
JOBS+=("--arm marsea --seed 0 --arm_kwargs {\"tau_i_pinned\":true} --out runs/e9_tau_i_pinned --phase_a_ckpt runs/phaseA_seed0.pt")
JOBS+=("--arm marsea --seed 0 --arm_kwargs {\"key_only_relation\":true} --out runs/e9_key_only_relation --phase_a_ckpt runs/phaseA_seed0.pt")
# per_head runs CHUNKED with head blocking on: composing the per-head parameters with head_block is the fix this
# round turns on, and a unit test is not the same as running it (review 30372ae D).
JOBS+=("--arm marsea --seed 0 --arm_kwargs {\"per_head\":12} --out runs/e9_per_head --phase_a_ckpt runs/phaseA_seed0.pt")
if [ "$EXTENDED_E9" = "1" ]; then
  # the paper's further field-argument ablations (Sec. 5 E9): "contribution (i)'s load-bearing test"
  JOBS+=("--arm marsea --seed 0 --arm_kwargs {\"no_nu\":true} --out runs/e9_no_nu --phase_a_ckpt runs/phaseA_seed0.pt")
  JOBS+=("--arm marsea --seed 0 --arm_kwargs {\"tau_j_global\":true} --out runs/e9_tau_j_global --phase_a_ckpt runs/phaseA_seed0.pt")
  JOBS+=("--arm marsea --seed 0 --arm_kwargs {\"r\":4} --out runs/e9_rank4 --phase_a_ckpt runs/phaseA_seed0.pt")
  JOBS+=("--arm marsea --seed 0 --arm_kwargs {\"r\":64} --out runs/e9_rank64 --phase_a_ckpt runs/phaseA_seed0.pt")
fi
# ---------------------------------------------------------------- the arms that must run DENSE, at UNIFORM_L -- launched FIRST
# quota_mode=uniform  the uniform quota is a reduction over ALL key chunks (read-through F-3).
# K_ret              the chunked fan-out has no root stage, so running the tournament arm chunked would silently
#                    measure the exact flat solve it is meant to ablate against (chunked.py refuses it outright).
# Both share ONE matched MarSea control at the same mode and length.
# FIRST, not after the chunked loop: as a block of their own they were a fourth wave of three jobs behind a third wave
# of one -- 20 jobs in 8 / 8 / 1 / 3 on 8 GPUs.  In the same round-robin as the chunked arms it is 8 / 8 / 4, one
# arm-duration less of wall clock (ops playbook review A).  The gate (DENSE_E9_OK) is decided at startup, so the
# block moves cleanly; wave 1 then lasts as long as its slowest member, dense or chunked.
DENSE_COMMON="$PA_FLAG --phase_a_ckpt runs/phaseA_seed0.pt --mode dense --L $UNIFORM_L --head_block $HEAD_BLOCK \
        --total_steps $STEPS --detector runs/detector.json --ruler_train $DATA_RULER --musique $DATA_MUSIQUE \
        --hotpot_n $HOTPOT_N --eval_ruler $EVAL --eval_every $EVAL_EVERY"
if [ "$DENSE_E9_OK" != "1" ]; then
  echo "dense E9 arms skipped (SKIP_DENSE_E9=1)"
else
echo "[dense E9 arms at L = $UNIFORM_L, with one matched MarSea control]"
launch "e9_uniform_quota" $PY scripts/run_train.py --arm marsea --seed 0 --arm_kwargs '{"quota_mode":"uniform"}' \
    --out runs/e9_uniform_quota $DENSE_COMMON
launch "e9_uniform_quota_control" $PY scripts/run_train.py --arm marsea --seed 0 \
    --out runs/e9_uniform_quota_control $DENSE_COMMON
# prop:tournament's ablation: dense, because the chunked fan-out has no root stage and would measure the exact
# flat solve instead.  On by default -- it is a pre-registered E9 arm, not an extra.
launch "e9_hierarchical_K64" $PY scripts/run_train.py --arm marsea --seed 0 --arm_kwargs '{"K_ret":64}' \
    --out runs/e9_hierarchical_K64 $DENSE_COMMON
fi
for job in "${JOBS[@]}"; do
  tag=$(echo "$job" | tr -cd '[:alnum:]_-' | cut -c1-60)
  launch "$tag" $PY scripts/run_train.py $job $COMMON
done; barrier                                # drains the dense arms too: one round-robin, 8 / 8 / 4 on 8 GPUs
if [ ${#FAILED_JOBS[@]} -gt 0 ]; then
  echo "=== ${#FAILED_JOBS[@]} job(s) FAILED:" >&2; printf '  %s\n' "${FAILED_JOBS[@]}" >&2; exit 1
fi
echo "S2 training queue done.  Evaluation: bash scripts/run_evalsuite.sh"
