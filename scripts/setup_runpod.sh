#!/usr/bin/env bash
# One-shot setup of a fresh GPU pod -- RunPod 8 x H200 (the 2026-09-17 plan) or a Nebius VM (record Sec. 44 / D-26);
# scripts/setup_nebius.sh is a symlink to this file for the documents that still use the old name.
# nothing in it is provider-specific.  Idempotent; safe to re-run.  Step 0 of the ops playbook:
#   mkdir -p runs && bash scripts/setup_runpod.sh 2>&1 | tee runs/setup.log
#   source pod_env.sh                     # in EVERY shell / tmux pane afterwards (the cache location, see below)
#
# What it does, in order: (1) the machine checks the run scripts assume -- bash >= 5.1 (run_evalsuite.sh's `wait -n -p`),
# a CUDA >= 12.8 driver (torch 2.11.0 cu128 fails at IMPORT below that), the GPUs' product names (the budget assumes
# SXM), tmux / flock / rsync present; (2) the venv with the pinned wheels; (3) the Hugging Face cache on the NETWORK
# VOLUME -- by default it is ~/.cache on the container disk, which a terminated pod loses, and every shell must point at
# the same place or the 3 GB backbone and the 600 MB HotpotQA set download again; (4) the backbone and HotpotQA into
# that cache; (5) the data check: 126 RULER sets with the D0 length provenance and both MuSiQue files (uploaded, never
# regenerated on the pod); (6) RULER + the --gold_depth patch ONLY when generation is actually needed (NEED_RULER=1, or
# no data present) -- the Paul Graham essay download is slow and fetches ~200 URLs.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python3}
mkdir -p runs
FAIL=0; WARN=()
say() { echo "  $*"; }

echo "=== 0. the machine"
if (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
  echo "  !! bash $BASH_VERSION < 5.1: run_evalsuite.sh needs 'wait -n -p' and refuses to start.  apt-get install -y bash" >&2; FAIL=1
else say "bash $BASH_VERSION ok"; fi
if command -v nvidia-smi >/dev/null 2>&1; then
  DRV=$(nvidia-smi | grep -o "CUDA Version: [0-9.]*" | awk '{print $3}')
  NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  say "GPUs: $NGPU  driver CUDA $DRV"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/    /'
  if [ -n "$DRV" ] && [ "$(printf '%s\n' "$DRV" 12.8 | sort -V | head -1)" != "12.8" ]; then
    echo "  !! driver CUDA $DRV < 12.8: torch 2.11.0+cu128 will fail at import.  Wrong image / host -- redeploy." >&2; FAIL=1
  fi
  # the product string names the parts to REJECT, not the one to accept: an H200 SXM reports plain "NVIDIA H200" and an
  # H100 SXM "NVIDIA H100 80GB HBM3", while the variants the budget's bandwidth arithmetic excludes call themselves
  # "NVIDIA H200 NVL" / "NVIDIA H100 PCIe".  A `grep -v SXM` allow-list warned on the correct hardware every time
  # (review acda7b6 1).  Read the names printed above once by eye as well.
  BAD=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -iE "NVL|PCIe" || true)
  [ -z "$BAD" ] || WARN+=("non-SXM part(s): $(echo "$BAD" | sort -u | tr '\n' ' ') -- the memory-bandwidth assumption behind the budget does not hold; the playbook says redeploy")
  MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
  [ "${MEM:-0}" -ge "${MIN_GPU_MIB:-140000}" ] || WARN+=("smallest GPU is ${MEM} MiB, below the ${MIN_GPU_MIB:-140000} MiB (H200, 141 GB) the memory profiles assume -- MIN_GPU_MIB=<mib> if that is intended")
  [ "$NGPU" = "${EXPECT_NGPU:-8}" ] || WARN+=("$NGPU GPU(s) visible, not ${EXPECT_NGPU:-8}: run_s2.sh <N> and run_evalsuite.sh <N> must be given the visible count, or a wave dies at CUDA_VISIBLE_DEVICES=$NGPU (EXPECT_NGPU=<n> to silence on a smaller pod)")
else
  WARN+=("nvidia-smi not found: no GPU visible (a CPU staging pod?) -- torch will install but nothing here can run")
fi
for t in tmux flock rsync git; do command -v $t >/dev/null 2>&1 || WARN+=("$t missing (apt-get install -y $t): tmux keeps the run alive across SSH drops, flock is used by the tests, rsync by the off-platform copy"); done
[ "$FAIL" = 0 ] || { echo "=== MACHINE CHECK FAILED -- fix the above before installing anything"; exit 1; }

echo "=== 1. python venv + pinned wheels"
if [ ! -d .venv ]; then
  $PY -m venv .venv || { echo "  !! venv creation failed: apt-get install -y python3-venv (Debian/Ubuntu images)" >&2; exit 1; }
fi
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python - <<'PYEOF'
import torch, transformers, peft
print(f"  torch {torch.__version__}  transformers {transformers.__version__}  peft {peft.__version__}")
if torch.cuda.is_available():
    print("  cuda ok:", torch.cuda.device_count(), "x", torch.cuda.get_device_name(0), "| torch built for CUDA", torch.version.cuda)
else:
    print("  !! torch sees no CUDA device")
PYEOF

echo "=== 2. the Hugging Face cache, on the volume that survives the pod"
# The rule is provider-neutral: the cache must live on the SAME filesystem as the repo (which is on the network volume),
# not on the container disk.  Default: the stock ~/.cache/huggingface when it shares the repo's filesystem (a laptop, a
# VM with one disk); otherwise <parent of the repo>/hf_cache -- /workspace/hf_cache on RunPod, /runpod-volume/hf_cache on
# serverless, whatever the mount point is.  An existing HF_HOME is honoured.  Testing for "/workspace" by name put the
# cache on the container disk WITHOUT a warning whenever the volume was mounted elsewhere (review acda7b6 4).
fsid() { stat -c %d "$1" 2>/dev/null || echo "?"; }
if [ -z "${HF_HOME:-}" ]; then
  mkdir -p "$HOME/.cache"
  if [ "$(fsid "$HOME/.cache")" = "$(fsid .)" ]; then HF_HOME="$HOME/.cache/huggingface"; else HF_HOME="$(dirname "$PWD")/hf_cache"; fi
fi
export HF_HOME; mkdir -p "$HF_HOME"
cat > pod_env.sh <<ENV
# generated by scripts/setup_runpod.sh -- source this in every shell (and before tmux new): the model / dataset cache
export HF_HOME="$HF_HOME"
export TOKENIZERS_PARALLELISM=false
ENV
say "HF_HOME=$HF_HOME   (pod_env.sh written; 'source pod_env.sh' in every shell)"
[ "$(fsid "$HF_HOME")" = "$(fsid .)" ] || WARN+=("HF_HOME=$HF_HOME is on a DIFFERENT filesystem from the repo: if the repo is on the network volume this cache is not, and a terminated pod loses the 3 GB backbone and the 600 MB HotpotQA set")

echo "=== 3. backbone weights + HotpotQA into that cache"
.venv/bin/python - <<'PYEOF'
import json, pathlib, time
from huggingface_hub import snapshot_download
rec = {}
for m in ['Qwen/Qwen2.5-1.5B']:
    p = snapshot_download(m, allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*','merges.txt','vocab.json'])
    print("  ", m, '->', p)
    # WHICH snapshot: the detector chose the patched layers by measuring this checkpoint's heads, and "Qwen2.5-1.5B"
    # alone does not identify them.  The snapshot path ends in snapshots/<commit sha> (review acda7b6 5).
    rec[m] = dict(snapshot_path=p, revision=pathlib.Path(p).name, recorded=time.strftime("%Y-%m-%dT%H:%M:%S"))
pathlib.Path("runs").mkdir(exist_ok=True); json.dump(rec, open("runs/backbone.json", "w"), indent=1)
print("   -> runs/backbone.json (model revision for the paper's appendix)")
from datasets import load_dataset
for split in ('train','validation'):
    print('   hotpot', split, len(load_dataset('hotpotqa/hotpot_qa','distractor',split=split)))
PYEOF

echo "=== 4. the data (uploaded, never regenerated here)"
# A set without the D0 length provenance is a PRE-FIX set (the E3 length confound): exit 1, always.  A short count or a
# missing MuSiQue file is only an error under REQUIRE_DATA=1 -- set it on the 8-GPU pod; on the staging pod setup may
# legitimately run before the upload has finished (review acda7b6 3).
if ! REQUIRE_DATA="${REQUIRE_DATA:-0}" .venv/bin/python - <<'PYEOF'
import glob, json, os, pathlib, sys
sets = sorted(glob.glob("data/ruler/*/validation.jsonl"))
noprov = [c for c in glob.glob("data/ruler/*/config.json") if "measured_length_mean" not in json.load(open(c))]
by = {}
for s in sets:
    by.setdefault(pathlib.Path(s).parent.name.split("_")[0], 0); by[pathlib.Path(s).parent.name.split("_")[0]] += 1
print(f"   RULER sets: {len(sets)} (plan: 126)  by grid: {by}")
print(f"   sets WITHOUT the D0 length provenance (pre-fix, must be regenerated on the desktop): {len(noprov)}")
for f in ("train", "dev"):
    p = pathlib.Path(f"data/musique/musique_ans_v1.0_{f}.jsonl"); print(f"   musique {f}: {'present' if p.exists() else 'MISSING (manual download; run_s2.sh / the eval queue refuse without it)'}")
need = {"TRAIN": 69, "QUICK": 5, "DET": 1, "E2": 15, "E3": 15, "E3pad": 3, "E3depth": 15, "E4": 3}
short = {k: (by.get(k, 0), v) for k, v in need.items() if by.get(k, 0) < v}
print("   grids short of the plan's count:", short or "none")
musique_missing = [f for f in ("train", "dev") if not pathlib.Path(f"data/musique/musique_ans_v1.0_{f}.jsonl").exists()]
if noprov:
    print(f"   !! {len(noprov)} set(s) predate the length fix: delete them and regenerate on the desktop (never here); proceeding is never correct"); sys.exit(1)
if os.environ.get("REQUIRE_DATA") == "1" and (short or musique_missing):
    print(f"   !! REQUIRE_DATA=1: short grids {short} / missing musique {musique_missing} -- the upload is incomplete"); sys.exit(1)
PYEOF
then echo "=== DATA CHECK FAILED"; exit 1; fi

echo "=== 5. RULER generator (only if generation is needed here)"
if [ "${NEED_RULER:-0}" = "1" ] || [ -z "$(ls -d data/ruler/*/ 2>/dev/null | head -1)" ]; then
  .venv/bin/python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); print('   nltk ok')"
  [ -d third_party/RULER ] || git clone --depth 1 https://github.com/NVIDIA/RULER.git third_party/RULER
  .venv/bin/python scripts/patch_ruler.py
  [ -f third_party/RULER/scripts/data/synthetic/json/PaulGrahamEssays.json ] || \
    (cd third_party/RULER/scripts/data/synthetic/json && ../../../../../../.venv/bin/python download_paulgraham_essay.py)
else
  say "skipped: data/ruler is populated and NEED_RULER is not set (the plan uploads the data; generating on an 8-GPU pod is billed idle time)"
fi

echo "=== done"
for w in "${WARN[@]}"; do echo "  WARNING: $w"; done
echo "  next:  source pod_env.sh  &&  tmux new -s marsea  &&  bash scripts/preflight.sh 2>&1 | tee runs/preflight.log"
