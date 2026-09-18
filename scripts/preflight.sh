#!/usr/bin/env bash
# Nebius preflight: everything that must hold before the S0 gate and the S2 queue are launched, measured ON THE TARGET
# GPU at the REAL sequence lengths.  Nothing here trains; it takes ~30 minutes on one H100.
#   bash scripts/preflight.sh 2>&1 | tee runs/preflight.log
#
# EVERY step is a gate: the script exits non-zero the moment one fails, and the queue behind it must not start.  (It
# used to print "peak < 72 GB" while comparing nothing, and profile_memory swallowed OutOfMemoryError and exited 0.)
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
# GATE_GB: the card's memory with 10 % headroom -- 72 on an 80 GB H100, 126 on a 143 GB H200 -- taken from the smallest
# visible GPU.  It was the literal 72: pod 1 (H200, 2026-09-18) measured the two-model paired evaluation at 86 GB at 16K,
# a pass on the card it ran on and a failure against the H100's ceiling, and the eval queue would have refused the dense
# arms' paired jobs on a number that fits with 55 GB to spare.  Override with GATE_GB=<n> only to be STRICTER.
# L / CHUNK / HEAD_BLOCK: the TRAINING configuration the queue will run (run_s2.sh reads the same variables), so the
# training memory probes and the step time measure it and not a default.  Pod 1 (H200, 2026-09-18) measured the 8K
# recipe at 7.6 s/sequence -- 6x the D-23 ceiling and Phase B at 67 h per arm -- and the 4K fallback at 1.0-1.7 s.
# The evaluation probes stay at the evaluation lengths (8K and 16K) and MarSeaContext's default chunk: run_eval has
# no --chunk.
L=${L:-8192}; CHUNK=${CHUNK:-1024}; HEAD_BLOCK=${HEAD_BLOCK:-2}
# EVAL_HEAD_BLOCK: the EVALUATION queue's head blocking (run_evalsuite.sh reads the same variable), separate from the
# training knob.  Pod 1 run 3: with HEAD_BLOCK=0 exported for training, the eval probes ran unblocked and the sites-only
# pass ran out of memory at 16K (extrapolated 130 GB) where head_block 2 had measured 54 GB -- and run_evalsuite.sh,
# reading the same HEAD_BLOCK, would have done the same to every 16K job.
EVAL_HEAD_BLOCK=${EVAL_HEAD_BLOCK:-2}
GATE_GB=${GATE_GB:-$($PY -c "import torch; print(int(0.9 * min(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / 2**30))")}
mkdir -p runs
echo "=============================== 1. environment and contract"
echo "GATE_GB=$GATE_GB (the smallest visible GPU's memory with 10 % headroom, unless set in the environment)"
echo "training configuration under test: L=$L CHUNK=$CHUNK HEAD_BLOCK=$HEAD_BLOCK (run_s2.sh reads the same variables)"
echo "evaluation configuration under test: EVAL_HEAD_BLOCK=$EVAL_HEAD_BLOCK, 8K and 16K, chunk 1024 (run_evalsuite.sh reads EVAL_HEAD_BLOCK)"
$PY - <<'PYEOF'
import torch, transformers, peft, marsea
from marsea.backbone import assert_transformers_contract
print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)
print("spec version implemented:", marsea.SPEC_VERSION)
print("eager_attention_forward contract:", assert_transformers_contract())
print("GPUs:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("per-GPU memory GB:", [round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1) for i in range(torch.cuda.device_count())])
assert torch.cuda.device_count() >= 1, "no GPU"
PYEOF

echo "=============================== 2. acceptance tests T0-T16 with the runtime invariants on"
MARSEA_DEBUG=1 $PY -m pytest tests/ -q

echo "=============================== 3. the numerical harness (numpy reference, 0 violations expected)"
$PY scripts/verify_all.py > runs/preflight_verify.log 2>&1 || \
  { echo "GATE FAILED: verify_all.py reported violations (see runs/preflight_verify.log)"; tail -20 runs/preflight_verify.log; exit 1; }
tail -2 runs/preflight_verify.log

echo "=============================== 4. memory at the REAL lengths (the S2 configuration decision)"
echo "--- training, chunked, 2 and 4 patched layers (differenced: the MARGINAL per-layer cost), head_block = ${HEAD_BLOCK:-2}"
# --targets 8192 only: NOTHING trains at 16K (E3 is evaluation-only), and gating a legitimate 8K training
# configuration on an extrapolated 16K TRAINING peak no job incurs would block the queue for nothing
# (review 30372ae D).  The 16K figure that matters is the evaluation one, measured in the third profile below.
$PY scripts/profile_memory.py --lengths 2048 4096 $L --layers 2 4 --mode_impl chunked --modes train --chunk $CHUNK \
    --head_block $HEAD_BLOCK --targets $L --gate_GB $GATE_GB --out runs/preflight_train_chunked.json
echo "--- training, DENSE, 4 patched layers, MEASURED at 8K and gated"
# This was "NOT a gate", falling through to "the queue uses --mode chunked anyway" -- but run_s2.sh trains three E9
# arms dense at 8K, one of them the pre-registered uniform-quota ablation, and quoted an extrapolated ~27 GB for it.
# The verdict lands in the JSON and run_s2.sh refuses the dense arms without it (review e982f83 B-4).
# THIS probe's gate failure (exit 3) is recorded, not fatal: under `set -e` it used to abort preflight before the cap-slack
# measurement, the step time and the detector -- the most likely gate failure taking three unrelated gates with it
# (review e16a843 F-1).  The verdict is enforced by run_s2.sh at startup.  Any OTHER failure (a crash) stays fatal.
DENSE_TRAIN_NOTE=""
$PY scripts/profile_memory.py --lengths 2048 4096 $L --layers 4 --mode_impl dense --modes train \
    --head_block $HEAD_BLOCK --targets $L --gate_GB $GATE_GB --out runs/preflight_train_dense.json || {
  rc=$?; [ $rc -eq 3 ] || exit $rc
  DENSE_TRAIN_NOTE="dense 8K training did NOT pass the memory gate (runs/preflight_train_dense.json): run_s2.sh will refuse to start unless SKIP_DENSE_E9=1, which drops the uniform-quota and K_ret E9 arms"
  echo "   NOTE: $DENSE_TRAIN_NOTE"
}
echo "--- teacher-forced evaluation, chunked, 4 patched layers, at 8K and 16K, with BOTH D-9a measurement heads kept"
# --sites 2: the second retained head at 16K was unmeasured (review e982f83 I).  The two heads (3, 4) fall in
# different head blocks at head_block = 2, which is also the configuration C-1's merge bug lived in.
# --gate_keys teacher_head: every 16K job (E3, E3pad, E3depth) keeps the measurement sites only.  The all-heads
# `teacher` figure is still measured and printed, but it is a pass NOTHING runs at 16K: on pod 1 (H200, 2026-09-18)
# it ran out of memory at 16K (extrapolated 156 GB, p = 2.05) while teacher_head measured 54 GB, and the gate
# stopped preflight -- and the cap proof, the step time and the detector behind it -- for a configuration the queue
# never uses.  The one all-heads pass the queue DOES run is B5's, gated in the dense probe below.
$PY scripts/profile_memory.py --lengths 4096 8192 16384 --layers 4 --mode_impl chunked --modes teacher --chunk 1024 \
    --head_block $EVAL_HEAD_BLOCK --sites 2 --targets 8192 16384 --gate_GB $GATE_GB --gate_keys teacher_head \
    --out runs/preflight_eval_chunked.json
echo "--- teacher-forced evaluation, DENSE, 4 patched layers, at 8K (five of the seven eval job types run dense; B5's pass 1 too)"
# gated on what the eval queue runs -- the measurement heads kept (teacher_head), plus B5's pass 1 (teacher_b5: every
# head of every patched layer with ctx.keep_dense on, fields E and supp_rel), which runs dense at 8K on E2 and was
# measured by no probe before (the `teacher` key keeps l*'s heads with every field: neither B5 nor anything else)
$PY scripts/profile_memory.py --lengths 4096 8192 --layers 4 --mode_impl dense --modes teacher --b5 \
    --head_block $EVAL_HEAD_BLOCK --sites 2 --targets 8192 --gate_GB $GATE_GB --gate_keys teacher_head teacher_b5 \
    --out runs/preflight_eval_dense.json
echo "--- teacher-forced evaluation of a DENSE arm on MarSea's paired relation, at 8K and 16K: TWO models resident"
# B0/B1/B2/B4 on E2..E5 -- most of the eval queue -- keep the paired MarSea model loaded for the whole job and run its
# relation pass per example before the arm's own; nothing measured that (review e16a843 F-4)
# Recorded, not fatal (as the dense training probe above): the most memory-hungry probe in the file must not abort the
# cap-binding proof, the step time and the detector behind it; run_evalsuite.sh reads the verdict (review 7814665 F-3).
PAIRED_NOTE=""
$PY scripts/profile_memory.py --arm B0 --paired --lengths 4096 8192 16384 --layers 4 --mode_impl chunked --modes teacher \
    --chunk 1024 --head_block $EVAL_HEAD_BLOCK --sites 2 --targets 8192 16384 --gate_GB $GATE_GB --gate_keys teacher_head \
    --out runs/preflight_eval_paired.json || {
  rc=$?; [ $rc -eq 3 ] || exit $rc
  PAIRED_NOTE="the two-model paired evaluation did NOT pass the memory gate (runs/preflight_eval_paired.json): run_evalsuite.sh will refuse the dense arms' --paired_marsea_ckpt jobs unless SKIP_PAIRED_GATE=1"
  echo "   NOTE: $PAIRED_NOTE"
}

echo "=============================== 4b. the unit cap on THIS device: kernel row-sum error (reported) and the binding rule (gated)"
# The cap's binding decision is taken from the relation's excess and is independent of the softmax kernel by
# construction (normalizer.row_masses); this step PROVES that on the target device -- razor-edge rows through the
# real dense, chunked and decode paths, forced empty and live at rho 0.05 and 0.5 -- records the kernel's own row-sum
# error, which is the number SAN-1 and the appendix carry (review e16a843 B/C), and bisects the theta guard's window
# (the fp32 prefix scan's error: 2.0e-6 at 16K on the dev GPU, above CAP_TOL) against the tolerance INV-3 allows
# (review 9bacef9 D).
$PY scripts/check_unit_cap.py --lengths 2048 4096 8192 16384 --out runs/unit_cap_device.json

echo "=============================== 5. step time (spec Sec. 18: > 1.2 s/sequence at 8K => 2 patched layers or 4K)"
$PY scripts/time_step.py --L $L --layers 4 --mode chunked --head_block $HEAD_BLOCK --chunk $CHUNK --steps 5 \
    --out runs/step_time.json
echo "--- step time, DENSE, at 8K: the three dense E9 arms' path (recorded; the D-23 rule is stated for the chunked queue)"
# row_masses replaced an nnz-sized reduction with an O(n_q n_k) fp64 pass and the cap's projection gained a second sweep:
# +17 % on the dominant Stage-1 primitive on CPU, quadratic in T, and nothing timed the dense path (review 9bacef9 E).
# Recorded, not fatal, like the dense memory probe above: it is the same configuration and may not fit at all.
DENSE_TIME_NOTE=""
$PY scripts/time_step.py --L $L --layers 4 --mode dense --head_block $HEAD_BLOCK --steps 5 --no_gate \
    --out runs/step_time_dense.json || {
  DENSE_TIME_NOTE="the dense 8K step-time probe failed (rc $?; runs/step_time_dense.json may be absent): the dense E9 arms' throughput is unmeasured"
  echo "   NOTE: $DENSE_TIME_NOTE"
}

echo "=============================== 6. the detector: regenerated unless it was produced by THIS code"
# `[ -f ] ||` preserved exactly the artefact that has to be replaced, and answered with a warning that exits 0
# (review 30372ae D).  A detector with no spec stamp predates the sink-exclusion and threshold-rejection fixes.
# Regenerate-on-doubt: keep the file ONLY when the check positively prints 0.  `NEED_DET=$(...)` used to be empty when
# the inline python raised before printing, the `= 1` test was then false, and the stale detector was kept
# (review e982f83 I).
NEED_DET=1
if [ -f runs/detector.json ]; then
  NEED_DET=$($PY - <<'PYEOF' 2>/dev/null || echo 1
import json
from marsea import SPEC_VERSION
d = json.load(open("runs/detector.json"))
print(0 if d.get("spec_version") == SPEC_VERSION and d.get("usable") is not None else 1)
PYEOF
)
fi
if [ "$NEED_DET" != "0" ] && [ -f runs/detector.json ] && [ "${PREFLIGHT_FORCE_DETECTOR:-0}" != "1" ] && \
   $PY -c "import json,sys; sys.exit(0 if 's0_licence' in json.load(open('runs/detector.json')) else 1)" 2>/dev/null; then
  # S0 writes its licence into this same file, and every trained checkpoint is stamped with its layers.  Bumping
  # SPEC_VERSION and re-running preflight used to move it aside and regenerate it: the licence was gone and, if the new
  # detector named different layers, check_detector refused every checkpoint (review e16a843 F).
  echo "GATE FAILED: runs/detector.json is not stamped by this code but carries an S0 licence; refusing to regenerate it."
  echo "  keep it:        fix the spec stamp deliberately, or"
  echo "  regenerate:     PREFLIGHT_FORCE_DETECTOR=1 bash scripts/preflight.sh   (then re-run S0; checkpoints trained on the"
  echo "                  old layer set will be refused by run_eval.py)"
  exit 1
fi
if [ "$NEED_DET" != "0" ]; then
  echo "--- regenerating runs/detector.json (absent, or not stamped by this code)"
  [ -f runs/detector.json ] && mv runs/detector.json runs/detector.json.superseded && \
    echo "    (old file kept: restore with  mv runs/detector.json.superseded runs/detector.json)"
  $PY scripts/run_detector.py --n 200 --L 4096
else
  echo "--- runs/detector.json carries this spec version; keeping it"
fi
$PY - <<'PYEOF'
import json, sys
d = json.load(open("runs/detector.json"))
print("patched layers:", d["patched_layers"], "| (l*, h*) =", (d["l_star"], d["h_star"]),
      "| heads above threshold:", d.get("n_heads_above_threshold"), "| usable:", d.get("usable"),
      "| spec", d.get("spec_version"), "git", d.get("git"), "at", d.get("created"))
if not d.get("usable"):
    print("GATE FAILED: no head clears the detector threshold -- App. H's fallback applies, not an S2 run"); sys.exit(1)
if d.get("layers_below_threshold"):
    print("NOTE: these layers entered PATCHED_LAYERS without clearing the threshold:", d["layers_below_threshold"])
PYEOF

if [ -n "$DENSE_TRAIN_NOTE$PAIRED_NOTE$DENSE_TIME_NOTE" ]; then
  echo "=============================== PREFLIGHT GATES PASSED EXCEPT (recorded, enforced by the queue scripts):"
  # `[ -n "$X" ] && echo` as the branch's LAST command made the script exit 1 exactly when only the dense gate had
  # failed -- the likeliest recorded failure, the one this branch exists to survive (review 9bacef9 C-1).  if/fi.
  if [ -n "$DENSE_TRAIN_NOTE" ]; then echo "  $DENSE_TRAIN_NOTE"; fi
  if [ -n "$PAIRED_NOTE" ]; then echo "  $PAIRED_NOTE"; fi
  if [ -n "$DENSE_TIME_NOTE" ]; then echo "  $DENSE_TIME_NOTE"; fi
else
  echo "=============================== ALL PREFLIGHT GATES PASSED (peak <= ${GATE_GB} GB, step time within D-23)"
fi
exit 0
