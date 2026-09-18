#!/usr/bin/env bash
# Generate every RULER set S0/S1/S2 needs, with the BACKBONE tokenizer (spec Sec. 13.1, D-27).  CPU-bound and
# parallelisable; ~2-4 h on a many-core VM.  Training seeds 100-102, evaluation seeds 0-2, quick eval seed 3,
# never overlapping.
#
# The TRAINING POOL goes first and every grid is attempted: under `set -e` a single infeasible config used to
# abort the script before `--grid train` ran at all, so S2 had nothing to train on.  Failures are collected and
# re-reported at the end, and the script still exits non-zero.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
mkdir -p data/ruler
FAILED=()
run() { echo "+ $*"; if ! "$@"; then FAILED+=("$*"); echo "!! FAILED: $*" >&2; fi; }
# the training pool (training procedure Sec. 2.1): the mixture the arms train on, seeds 100-102.  S2 blocks on this.
run $PY scripts/gen_ruler.py --grid train --n 400 --seeds 100 101 102
# the detector's set (4K, words, seed 0) and the quick held-out eval grid (seed 3: not an eval seed, not a train seed)
run $PY scripts/gen_ruler.py --grid detector
run $PY scripts/gen_ruler.py --grid eval_quick --n 25
# the evaluation grids (D-27): E2 at K = 8 / L = 8K; E3 at L = 16K held constant across the n-sweep; E4 at m = 1
for g in E2 E3 E4 E3depth; do run $PY scripts/gen_ruler.py --grid $g --n 200 --seeds 0 1 2; done
echo "=== generated:"; ls -d data/ruler/*/ 2>/dev/null | wc -l; du -sh data/ruler
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "=== ${#FAILED[@]} grid(s) FAILED:" >&2; printf '  %s\n' "${FAILED[@]}" >&2; exit 1
fi
echo "=== all grids ok"
