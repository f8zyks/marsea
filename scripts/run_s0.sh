#!/usr/bin/env bash
# S0 -- the gate on everything (spec Sec. 14).  Two questions, both on the PHASE-A weights (not the raw base model):
#   (a) does any T_j construction route?  Per kind (D-9a's split: coreference columns and value columns), every
#       (layer, head), span-tolerant, the attention sink excluded, licensed at >= 80 % of EXAMPLES.
#   (b) is there a replication pair -- M(a) > m_j at stated a -- so that Cor. capacity has bite on real data?
# If neither construction is licensed, spec Sec. 12.3 applies: E7's column half is reported as "not measurable on this
# backbone" and the row side carries E7.  run_eval.py reads that licence out of runs/detector.json automatically.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
SETS=${SETS:-"data/ruler/E2_L8192_K8_V1_Q1_d0.5_s0,data/ruler/E2_L8192_K8_V4_Q1_d0.5_s0"}
N=${N:-200}
SEED=${SEED:-0}
CKPT=${CKPT:-runs/phaseA_seed$SEED.pt}
[ -f "$CKPT" ] || { echo "Phase-A checkpoint $CKPT missing: run 'bash scripts/run_s2.sh' (it builds Phase A first) or"; \
                    echo "  $PY scripts/run_train.py --arm B0 --seed $SEED --phase_a_only ..."; exit 1; }
echo "=== detector (Sec. 6.7): retrieval heads on the UNPATCHED backbone -> PATCHED_LAYERS, (l*, h*)"
[ -f runs/detector.json ] || $PY scripts/run_detector.py --n 200 --L 4096
echo "=== S0 sweep on the Phase-A weights"
$PY scripts/run_s0_sweep.py --ckpt "$CKPT" --set "$SETS" --n $N --out runs/s0_sweep_phaseA.json \
    --update_detector runs/detector.json
$PY - <<'PYEOF'
import json
det = json.load(open("runs/detector.json"))
lic = det.get("column_measurable", {}); rep = det.get("s0_replication", {})
print("\n=== S0 VERDICT ===")
print("  column measurable per T_j kind:", lic)
for k, v in (det.get("s0_licence") or {}).items():
    print(f"    {k:6s} {v['construction']:16s} at (l*,h*) {v['at_l_star_h_star']:.2f}  best {v['best']}"
          f"  -> measured at site {v.get('site')}")
for name, r in rep.items():
    if isinstance(r, dict) and any("M_gt_m" in k for k in r):
        print(f"    replication {name}: " + ", ".join(f"{k.split('=')[-1]}: {v:.2f}" for k, v in r.items() if "M_gt_m" in k)
              + f"  m_j sizes {r.get('m_sizes')}")
ok = any(lic.values()) if isinstance(lic, dict) else bool(lic)
print("\n  GATE:", "at least one construction licensed -- E7's column half is measurable" if ok else
      "NO construction licensed -- report E7's column half as 'not measurable on this backbone' (spec Sec. 12.3);\n"
      "        the row side carries E7 and the interval-hit rate is reported on the synthetic columns and the harness.")
print("  (run_eval.py honours this automatically: it scores each kind at the SITE printed above, passing")
print("   --coref_site / --value_site, so a kind licensed only 'anywhere' is measured where it was licensed.)")
PYEOF
