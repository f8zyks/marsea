#!/usr/bin/env bash
# The evaluation queue after S2 (spec Sec. 13): E2, E3 (load-bearing, 16K, chunked), E4, E5, E7/E8 from the same passes.
# One process per GPU; every job writes runs/eval/<tag>.parquet + <tag>_table.json under its OWN tag, so a restart
# never recomputes a finished cell and no two jobs share a file (review e982f83 B-2).
set -e
cd "$(dirname "$0")/.."
NGPU=${1:-8}; PY=${PY:-.venv/bin/python}
# NGPU = 0 or a non-integer left the slot loop spinning with no sleep and no job (review 9bacef9 E)
[[ "$NGPU" =~ ^[1-9][0-9]*$ ]] || { echo "run_evalsuite.sh: NGPU must be a positive integer (got '$NGPU')" >&2; exit 1; }
DET=runs/detector.json
# the E9 arms run_s2.sh trains (its JOBS list and the dense group); EXTENDED_E9=1 adds the S4 ablations.  Listed here so
# an arm that was never trained is REPORTED rather than passed over: `for d in runs/e9_*/` cannot see a directory that
# does not exist (review 9bacef9 C-2).  A directory under runs/e9_*/ that is not in the list is evaluated too.
E9_ARMS="e9_tau_i_pinned e9_key_only_relation e9_per_head e9_uniform_quota e9_uniform_quota_control e9_hierarchical_K64"
[ "${EXTENDED_E9:-0}" = "1" ] && E9_ARMS="$E9_ARMS e9_no_nu e9_tau_j_global e9_rank4 e9_rank64"
for d in runs/e9_*/; do
  [ -d "$d" ] || continue
  b=$(basename "$d"); [[ " $E9_ARMS " == *" $b "* ]] || E9_ARMS="$E9_ARMS $b"
done
HEAD_BLOCK=${HEAD_BLOCK:-2}                                # as preflight profiles it: the memory gate must be the queue's
N16K=${N16K:-400}           # examples per 16K job, spread evenly over its configs.  n = 1000 (5 configs x 200) is ~3x
                            # the compute table's evaluation budget (review e982f83 H) and E3's paired pass was
                            # sized at ~268 GB of host RAM per process (B-3).
N16K_PAD=${N16K_PAD:-80}    # E3pad is ONE config: the same count as each E3 config gets at N16K = 400
E3DEPTH_ARMS=${E3DEPTH_ARMS:-"marsea B0"}; E3DEPTH_SEEDS=${E3DEPTH_SEEDS:-0}
# E3depth is a CONTROL -- the paper needs a position-marginal profile, not a per-arm sweep -- and at every arm x seed it
# cost as much as E3 itself (review e982f83 H)

# ---------------------------------------------------------------- gates: the S0 licence and the detector
# E7's column half is reported as measurable only under S0's licence (spec Sec. 12.3).  Following the documented order
# with S0 skipped used to evaluate the whole grid with every column "measurable" by default (review e982f83 I).
if ! $PY - "$DET" <<'PYEOF'
import json, sys
p = sys.argv[1]
try:
    d = json.load(open(p))
except Exception as e:
    print(f"REFUSING TO START: cannot read {p} ({e})"); sys.exit(1)
if "s0_licence" not in d:
    print(f"REFUSING TO START: {p} carries no S0 licence -- run scripts/run_s0.sh on the Phase-A weights first"); sys.exit(1)
print("S0 licence present:", d.get("column_measurable"))
PYEOF
then exit 1; fi

# ---------------------------------------------------------------- the evaluation data, before the 96 jobs are launched
# run_s2.sh asserts the TRAINING mixture; nothing checked the evaluation sets, and the MuSiQue dev file is a separate
# manual download -- absent, 14 E5musique jobs failed one at a time hours into the queue (review e16a843 F-2).
if [ "${EVAL_DATA_CHECK:-1}" = "1" ] && ! $PY - "${MUSIQUE_DEV:-data/musique/musique_ans_v1.0_dev.jsonl}" <<'PYEOF'
import glob, json, os, pathlib, sys
bad = []
for seed in (0, 1, 2):
    for pat in (f"data/ruler/E2_L8192_K8_V*_s{seed}", f"data/ruler/E3_L16384_K*_s{seed}", f"data/ruler/E3pad_L16384_*_s{seed}",
                f"data/ruler/E4_L8192_*_s{seed}"):
        if not glob.glob(pat + "/validation.jsonl"):
            bad.append(f"no RULER set matches {pat}")
if not glob.glob("data/ruler/E3depth_L16384_*_s0/validation.jsonl"):
    bad.append("no RULER set matches data/ruler/E3depth_L16384_*_s0")
if not pathlib.Path(sys.argv[1]).exists():
    bad.append(f"MuSiQue dev file {sys.argv[1]} is missing (manual download)")
# the paired MarSea checkpoints the dense arms score against: a failed marsea_seed1 used to cost twelve jobs, one at a
# time (review 7814665 F-4)
for seed in (0, 1, 2):
    if any(pathlib.Path(f"runs/{arm}_seed{seed}/final.pt").exists() for arm in ("B0", "B1", "B2", "B4")) \
            and not pathlib.Path(f"runs/marsea_seed{seed}/final.pt").exists():
        bad.append(f"runs/marsea_seed{seed}/final.pt is missing but dense arms of seed {seed} exist (their paired relation)")
# the two-model paired evaluation's memory verdict (preflight 4, recorded rather than fatal there)
try:
    g = (json.load(open("runs/preflight_eval_paired.json")).get("gate") or {})
    if not g.get("passed") and os.environ.get("SKIP_PAIRED_GATE", "0") != "1":
        bad.append(f"the paired two-model evaluation did not pass preflight's memory gate ({g}); SKIP_PAIRED_GATE=1 to run anyway")
except FileNotFoundError:
    print("NOTE: runs/preflight_eval_paired.json absent -- the paired evaluation's memory is ungated (run preflight.sh)")
except Exception as e:                    # a truncated JSON was a traceback that skipped the remaining checks (review 9bacef9 E)
    bad.append(f"runs/preflight_eval_paired.json is unreadable ({e}); re-run preflight.sh or SKIP_PAIRED_GATE=1")
try:
    sys.path.insert(0, ".")
    from marsea.data.qa import load_hotpot
    load_hotpot("validation", n=5)
except Exception as e:
    bad.append(f"HotpotQA validation cannot be loaded: {e}")
for b in bad: print("REFUSING TO START:", b)
sys.exit(1 if bad else 0)
PYEOF
then exit 1; fi

# ---------------------------------------------------------------- the checkpoint inventory, BEFORE the first job
# Every arm the queue below will look for, checked here in one pass.  An arm with no checkpoint used to be dropped as
# the loop reached it -- the E9 loop without a word, the main loop with a note printed after the final barrier, ~30 h in
# (review 7814665 F-4, 9bacef9 C-2).  The list is written now and the queue refuses to start on an incomplete grid
# unless ALLOW_MISSING_ARMS=1, in which case the missing arms are named up front and again at the end.
SKIPPED_ARMS=()
for seed in 0 1 2; do
  ARMS="marsea B0 B2 B3"; [ "$seed" = "0" ] && ARMS="$ARMS B1 B4"
  for arm in $ARMS; do [ -f "runs/${arm}_seed${seed}/final.pt" ] || SKIPPED_ARMS+=("${arm}_seed${seed}"); done
  [ -f "runs/B0_seed${seed}/final.pt" ] || SKIPPED_ARMS+=("B0_seed${seed} (B5's weights)")
done
[ -f runs/marsea_seed0/final.pt ] || SKIPPED_ARMS+=("marsea_seed0 (E9's matched chunked reference)")
for e9 in $E9_ARMS; do [ -f "runs/${e9}/marsea_seed0/final.pt" ] || SKIPPED_ARMS+=("${e9} (E9)"); done
if [ -f "${E6_SET:-data/iheval/iheval.jsonl}" ]; then
  for arm in marsea B0; do [ -f "runs/${arm}_seed0/final.pt" ] || SKIPPED_ARMS+=("${arm}_seed0 (E6)"); done
fi
mkdir -p runs/eval
if [ ${#SKIPPED_ARMS[@]} -gt 0 ]; then
  echo "=== ${#SKIPPED_ARMS[@]} arm(s) have NO checkpoint and will NOT be evaluated:" >&2; printf '  %s\n' "${SKIPPED_ARMS[@]}" >&2
  printf '%s\n' "${SKIPPED_ARMS[@]}" > runs/eval/SKIPPED_ARMS.txt
  [ "${ALLOW_MISSING_ARMS:-0}" = "1" ] || { echo "REFUSING TO START on an incomplete grid (ALLOW_MISSING_ARMS=1 to evaluate what exists; the list is in runs/eval/SKIPPED_ARMS.txt)" >&2; exit 2; }
else
  rm -f runs/eval/SKIPPED_ARMS.txt
fi

# A bare `wait` returns 0 whatever the jobs did, so no failed evaluation was ever observed (review 30372ae D).
# Work-conserving: each GPU is a slot, a job takes the first free one, and a finished job frees its slot at once
# (`wait -n`).  The per-batch barrier made a 0.1 h E3pad wait for a 2.2 h E3 in the same batch -- ~6 h of wall clock
# (review e16a843 F).  Concurrency is still at most NGPU and never two jobs on one GPU (review e982f83 B-1).
# `wait -n -p` needs bash >= 5.1 (Ubuntu 22.04 ships 5.1; 20.04 ships 5.0, where -p is unknown and every job would be
# reported failed) -- assert it rather than find out in the log (review 7814665)
if (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
  echo "run_evalsuite.sh needs bash >= 5.1 for 'wait -n -p' (this is $BASH_VERSION)" >&2; exit 1
fi
SLOT_PID=(); SLOT_TAG=(); FAILED_JOBS=()
wait_one() {                                  # block until ANY running job ends; record its status and free its GPU
  local pid="" rc=0 g
  wait -n -p pid "${SLOT_PID[@]}" || rc=$?
  [ -n "$pid" ] || return 0                   # nothing was waited on (no children): never loop on an empty slot table
  for g in "${!SLOT_PID[@]}"; do
    if [ "${SLOT_PID[$g]}" = "$pid" ]; then
      if [ "$rc" != "0" ]; then
        FAILED_JOBS+=("${SLOT_TAG[$g]}"); echo "!! FAILED: ${SLOT_TAG[$g]}  (runs/eval/log_${SLOT_TAG[$g]}.txt)" >&2
      fi
      unset "SLOT_PID[$g]" "SLOT_TAG[$g]"
    fi
  done
}
barrier() {                                   # drain everything (between sections, and at the end)
  while [ ${#SLOT_PID[@]} -gt 0 ]; do wait_one; done
}
launch() { # launch <tag> <command...>
  local tag=$1; shift
  [ -f "runs/eval/$tag.done" ] && { echo "skip $tag (done)"; return 0; }
  local gpu="" g
  while [ -z "$gpu" ]; do
    for ((g = 0; g < NGPU; g++)); do
      if [ -z "${SLOT_PID[$g]:-}" ]; then gpu=$g; break; fi
    done
    [ -n "$gpu" ] || wait_one
  done
  echo "[gpu $gpu] $tag"
  ( CUDA_VISIBLE_DEVICES=$gpu "$@" > "runs/eval/log_$tag.txt" 2>&1 && touch "runs/eval/$tag.done" ) &
  SLOT_PID[$gpu]=$!; SLOT_TAG[$gpu]=$tag
  return 0
}
run() { # run <tag> <run_eval args...>
  local tag=$1; shift
  launch "$tag" $PY scripts/run_eval.py --tag "$tag" --head_block "$HEAD_BLOCK" --out runs/eval "$@"
}
for seed in 0 1 2; do
  # B1 and B4 train on seed 0 only (D-24), so they are added to that seed's list rather than skipped entirely --
  # the loop used to read `for arm in marsea B0 B2 B3`, and B1, B4 were never evaluated at all (review d5bd980 F)
  ARMS="marsea B0 B2 B3"; [ "$seed" = "0" ] && ARMS="$ARMS B1 B4"
  for arm in $ARMS; do
    ck="runs/${arm}_seed${seed}/final.pt"; [ -f "$ck" ] || continue          # inventoried above
    paired=""; case $arm in B0|B1|B2|B4) paired="--paired_marsea_ckpt runs/marsea_seed${seed}/final.pt";; esac
    base="--arm $arm --ckpt $ck --detector $DET --seed $seed"
    # E2: the m-sweep at 8K
    run "${arm}_s${seed}_E2" $base --set "data/ruler/E2_L8192_K8_V*_s${seed}" --stratify m --experiment E2 $paired
    # E3: the n-sweep at 16K -- the load-bearing experiment, chunked path, with the inert-padding control
    run "${arm}_s${seed}_E3" $base --set "data/ruler/E3_L16384_K*_s${seed}" --n "$N16K" \
        --stratify n --experiment E3 --mode chunked $paired
    run "${arm}_s${seed}_E3pad" $base --set "data/ruler/E3pad_L16384_*_s${seed}" --n "$N16K_PAD" \
        --stratify n --experiment E3 --mode chunked $paired
    # E3depth: the gold needle's depth swept at fixed n, stratified by depth -- marsea and B0 at seed 0 only
    if [[ " $E3DEPTH_ARMS " == *" $arm "* && " $E3DEPTH_SEEDS " == *" $seed "* ]]; then
      run "${arm}_s${seed}_E3depth" $base --set "data/ruler/E3depth_L16384_*_s${seed}" --n "$N16K" \
          --stratify depth --experiment E3 --mode chunked $paired
    fi
    # E4: m = 1
    run "${arm}_s${seed}_E4" $base --set "data/ruler/E4_L8192_*_s${seed}" --stratify m --experiment E4 $paired
    # E5: MuSiQue and HotpotQA -- on the paired relation too, or the dense arms' E5 precision is a different
    # measurement from their RULER one (review e982f83 G)
    run "${arm}_s${seed}_E5musique" $base --kind musique --set data/musique/musique_ans_v1.0_dev.jsonl --n 500 \
        --experiment E5 $paired
    run "${arm}_s${seed}_E5hotpot" $base --kind hotpot --set hotpot --n 500 --experiment E5 $paired
  done
done
barrier
# E9 (D-24: "contribution (i)'s load-bearing test").  Its checkpoints land under runs/e9_*/marsea_seed0/final.pt, a
# path the loop above cannot match, so no E9 arm had an evaluation invocation ANYWHERE (review d5bd980 F).  Each arm
# is evaluated on E2 with the kwargs it trained with -- run_eval reads those out of the checkpoint.  --mode chunked:
# the arms that TRAINED chunked are evaluated on the path they trained on, so per_head x head_block x chunked is
# measured rather than only unit-tested (review e982f83 I); run_eval forces dense for the arms that trained dense.
# The E9 comparator on the SAME path: the E9 arms that trained chunked are evaluated chunked (below), while the main
# loop's marsea E2 job runs dense -- so the load-bearing comparison read across a mode boundary, and E8's counters are
# mode-dependent (review e16a843 F-3).  One matched chunked reference; the dense-trained arms compare against
# e9_uniform_quota_control, which is evaluated dense.
if [ -f runs/marsea_seed0/final.pt ]; then
  run "e9_reference_E2" --arm marsea --ckpt runs/marsea_seed0/final.pt --detector $DET --seed 0 \
      --set "data/ruler/E2_L8192_K8_V*_s0" --stratify m --experiment E9 --mode chunked
fi
for e9 in $E9_ARMS; do                        # the inventoried list, not a glob over what happens to exist (C-2)
  ck="runs/${e9}/marsea_seed0/final.pt"; [ -f "$ck" ] || continue
  run "e9_${e9}_E2" --arm marsea --ckpt "$ck" --detector $DET --seed 0 --set "data/ruler/E2_L8192_K8_V*_s0" \
      --stratify m --experiment E9 --mode chunked
done
barrier
# E6 (instruction-following under the deployed normaliser): queued here rather than nowhere
if [ -f "${E6_SET:-data/iheval/iheval.jsonl}" ]; then
  for arm in marsea B0; do
    ck="runs/${arm}_seed0/final.pt"; [ -f "$ck" ] || continue                 # inventoried above
    launch "e6_${arm}" $PY scripts/run_e6.py --arm $arm --ckpt "$ck" --detector $DET --head_block "$HEAD_BLOCK" \
        --set "${E6_SET:-data/iheval/iheval.jsonl}" --tag "e6_${arm}" --out runs/eval
  done
  barrier
else
  echo "E6 NOT RUN: no IHEval set at ${E6_SET:-data/iheval/iheval.jsonl} (its per-task checkers are not vendored; see run_e6.py)"
fi
# B5 last (D-29: teacher-forced only; its task-level cell reads "n/a by design")
for seed in 0 1 2; do
  [ -f "runs/B0_seed${seed}/final.pt" ] || continue                             # inventoried above
  run "B5_s${seed}_E2" --arm B5 --ckpt "runs/B0_seed${seed}/final.pt" --detector $DET --seed $seed \
      --set "data/ruler/E2_L8192_K8_V*_s${seed}" --stratify m --experiment E2 --modes teacher \
      --b5_marsea_ckpt "runs/marsea_seed${seed}/final.pt"
done
barrier
if [ ${#SKIPPED_ARMS[@]} -gt 0 ]; then
  # an arm with no checkpoint used to be dropped without a word and the queue exited 0: B3 missing, collect.py
  # reporting on what exists, nothing saying so (review 7814665 F-4).  Named at the top (ALLOW_MISSING_ARMS=1 got us
  # here) and again now, beside the tables it is missing from -- before the failed-job verdict, so both are seen.
  echo "=== ${#SKIPPED_ARMS[@]} arm(s) had NO checkpoint and were NOT evaluated (runs/eval/SKIPPED_ARMS.txt):" >&2
  printf '  %s\n' "${SKIPPED_ARMS[@]}" >&2
fi
if [ ${#FAILED_JOBS[@]} -gt 0 ]; then
  echo "=== ${#FAILED_JOBS[@]} evaluation job(s) FAILED:" >&2; printf '  %s\n' "${FAILED_JOBS[@]}" >&2; exit 1
fi
if [ ${#SKIPPED_ARMS[@]} -gt 0 ]; then
  echo "evaluation queue done on an INCOMPLETE grid -> runs/eval/<tag>.parquet + <tag>_table.json;  tables: $PY scripts/collect.py"
  exit 2
fi
echo "evaluation queue done -> runs/eval/<tag>.parquet + <tag>_table.json;  tables: $PY scripts/collect.py"
