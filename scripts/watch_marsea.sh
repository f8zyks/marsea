#!/usr/bin/env bash
# watch_marsea.sh -- one screen that answers "is this run healthy, and is the queue moving?"
#
# Put it in scripts/ on the Nebius box and run it in its own tmux pane:
#     watch -n 120 bash scripts/watch_marsea.sh
# or once:
#     bash scripts/watch_marsea.sh
#
# Everything it prints is read from files the run already writes; it starts nothing and changes
# nothing.  The thresholds are the pre-registered ones (training procedure §11, RUNBOOK §6).
set -uo pipefail
cd "$(dirname "$0")/.." 2>/dev/null || true
PY=${PY:-.venv/bin/python}
NOW=$(date -u +"%Y-%m-%d %H:%M:%SZ")

hr() { printf '%s\n' "----------------------------------------------------------------------------------"; }
echo "MarSea watch  $NOW"; hr

# ---------------------------------------------------------------- GPUs
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU  idx  util%  mem_used/total  temp  procs"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu \
             --format=csv,noheader,nounits | awk -F', ' '{printf "     %-4s %-6s %6s/%-6s MiB %4sC\n",$1,$2,$3,$4,$5}'
  IDLE=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '$1<5' | wc -l)
  [ "$IDLE" -gt 0 ] && echo "  !! $IDLE GPU(s) below 5% utilisation -- a slot has died or the queue has drained"
fi
hr

# ---------------------------------------------------------------- training runs
echo "TRAINING"
shopt -s nullglob
FOUND=0
for d in runs/*_seed*/; do
  log="$d/train_log.jsonl"; [ -f "$log" ] || continue
  FOUND=1
  $PY - "$d" "$log" <<'PYEOF'
import json, sys, os, time
d, log = sys.argv[1], sys.argv[2]
name = os.path.basename(d.rstrip("/"))
last = None; n = 0
with open(log) as fh:
    for line in fh:
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except Exception: continue
        if "step" in r: last, n = r, n + 1
if last is None:
    print(f"  {name:28s}  no steps logged yet"); raise SystemExit
age = (time.time() - os.path.getmtime(log)) / 60.0
e8 = last.get("e8") or {}
rhos = [v.get("rho") for v in e8.values() if isinstance(v, dict) and "rho" in v]
taui = [v.get("tau_i_median") for v in e8.values() if isinstance(v, dict) and v.get("tau_i_median")]
cap  = [v.get("frac_rows_over_unit") for v in e8.values() if isinstance(v, dict) and v.get("frac_rows_over_unit")]
sing = [v.get("frac_Ecol_singleton") for v in e8.values() if isinstance(v, dict) and v.get("frac_Ecol_singleton")]
vtj  = [v.get("var_tau_j") for v in e8.values() if isinstance(v, dict) and v.get("var_tau_j")]
zero = [v.get("frac_rows_zero_mass") for v in e8.values() if isinstance(v, dict) and "frac_rows_zero_mass" in v]
flat = lambda xs: [x for s in xs for x in (s if isinstance(s, list) else [s]) if isinstance(x, (int, float))]
mean = lambda xs: (sum(xs)/len(xs)) if xs else float("nan")
nf = last.get("nonfinite_grads") or 0
flags = []
if rhos and max(rhos) < 0.005:              flags.append("COVERAGE<0.5% -- the arm has become B0")
if flat(sing) and mean(flat(sing)) > 0.80:  flags.append("|E_.j|=1 on >80% -- trained OUT of Thm. 1")
if flat(vtj) and mean(flat(vtj)) < 1e-6:    flags.append("var(tau_j)~0 -- constant temperature")
ti = mean(flat(taui))
if ti == ti and not (0.8 <= ti <= 1.3):     flags.append(f"tau_i median {ti:.2f} outside 0.8-1.3")
cb = mean(flat(cap))
if cb == cb and cb > 0.10:                  flags.append(f"cap binding on {100*cb:.1f}% of rows (>10%)")
zm = mean(flat(zero))
if zm == zm and zm > 0.01:                  flags.append(f"{100*zm:.1f}% of rows end with zero mass")
if nf:                                      flags.append(f"NON-FINITE GRADS: {nf}")
if age > 20:                                flags.append(f"log stale {age:.0f} min -- process may be dead")
if last.get("head_lr_scale", 1.0) != 1.0:   flags.append(f"head LR halved (scale {last['head_lr_scale']}) -- a NaN already fired")
print(f"  {name:28s}  step {last['step']:>5}  loss {last.get('loss', float('nan')):7.4f}  "
      f"rho {('%.3f' % mean(rhos)) if rhos else '   -  '}  tau_i {('%.2f' % ti) if ti == ti else ' -  '}  "
      f"cap {('%.1f%%' % (100*cb)) if cb == cb else '  -  '}  {age:4.0f}m ago")
for f in flags: print(f"      !! {f}")
PYEOF
done
[ "$FOUND" = 0 ] && echo "  (no training runs yet)"
hr

# ---------------------------------------------------------------- evaluation queue
echo "EVALUATION QUEUE"
if [ -d runs/eval ]; then
  # `ls glob` under nullglob drops the argument and lists the CWD instead -- count with find.
  DONE=$(find runs/eval -maxdepth 1 -name '*.done' 2>/dev/null | wc -l)
  PARQ=$(find runs/eval -maxdepth 1 -name '*.parquet' 2>/dev/null | wc -l)
  LOGS=$(find runs/eval -maxdepth 1 -name 'log_*.txt' 2>/dev/null | wc -l)
  echo "  done markers: $DONE   parquet: $PARQ   job logs: $LOGS"
  if [ -f runs/eval/SKIPPED_ARMS.txt ]; then
    echo "  !! SKIPPED ARMS (no checkpoint):"; sed 's/^/       /' runs/eval/SKIPPED_ARMS.txt
  fi
  BAD=$(find runs/eval -maxdepth 1 -name 'log_*.txt' -exec grep -l -E "Traceback|CUDA out of memory|RuntimeError|InvariantError" {} + 2>/dev/null | head -20)
  if [ -n "$BAD" ]; then
    echo "  !! job logs carrying an error:"
    for f in $BAD; do
      echo "       $(basename "$f"): $(grep -m1 -E 'Traceback|CUDA out of memory|RuntimeError|InvariantError' "$f" | cut -c1-90)"
    done
  fi
  # the newest job log, so a stall is visible
  NEW=$(find runs/eval -maxdepth 1 -name 'log_*.txt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
  [ -n "$NEW" ] && echo "  newest: $(basename "$NEW")  ($(( ( $(date +%s) - $(stat -c %Y "$NEW") ) / 60 )) min since last write)"
else
  echo "  (not started)"
fi
hr

# ---------------------------------------------------------------- device gates, recorded once
echo "DEVICE GATES"
for f in runs/unit_cap_device.json runs/step_time.json runs/step_time_dense.json runs/detector.json; do
  [ -f "$f" ] || { echo "  $(basename "$f"): MISSING"; continue; }
  $PY - "$f" <<'PYEOF'
import json, sys, os
f = sys.argv[1]; b = os.path.basename(f)
try: d = json.load(open(f))
except Exception as e: print(f"  {b}: unreadable ({e})"); raise SystemExit
if "unit_cap" in b:
    ls = d.get("lengths", {})
    for n, v in ls.items():
        gw = v.get("theta_guard_window", {})
        print(f"  unit_cap  n_k={n:>6}  window {gw.get('window', float('nan')):.2e}  "
              f"INV-3 allows {v.get('inv3_tolerance', float('nan')):.2e}  "
              f"{'ok' if v.get('window_within_inv3') else 'EXCEEDED'}")
    if d.get("MARSEA_TOL_CAP"): print(f"  unit_cap  override in force: {d['MARSEA_TOL_CAP']}")
elif "step_time" in b:
    print(f"  {b}: {json.dumps({k: v for k, v in list(d.items())[:6]})[:110]}")
elif "detector" in b:
    print(f"  detector  layers {d.get('patched_layers')}  (l*,h*)=({d.get('l_star')},{d.get('h_star')})  "
          f"licence {'YES' if 's0_licence' in d else 'NO -- S0 has not run'}")
    lic = d.get("s0_licence") or {}
    if lic: print(f"            licensed kinds: {list(lic.get('by_kind', lic).keys()) if isinstance(lic, dict) else lic}")
PYEOF
done
hr
echo "next:  tail -f runs/<arm>_seed<k>/train_log.jsonl   |   less runs/eval/log_<tag>.txt   |   $PY scripts/collect.py --eval_dir runs/eval"
