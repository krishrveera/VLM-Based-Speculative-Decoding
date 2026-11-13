#!/bin/bash
# Run a set of MMSpec methods on ONE GPU, log the environment, then summarize.
# Put this file and summarize_mmspec.py in the MMSpec repo root (or a tools/ folder next to it).
#
# Usage:
#   bash run_mmspec.sh testmini baseline vispec eagle2 msd pld      # quick check, ~1 h total
#   bash run_mmspec.sh test baseline eagle eagle2 eagle3 medusa msd vispec lookahead recycling pld sam
#
# Always run baseline FIRST and on the SAME GPU type as the methods: speedup is relative to it.

set -euo pipefail
SPLIT="${1:?split: testmini|test}"; shift
METHODS=("$@")
MODEL_DIR="${MODEL_DIR:-Qwen2.5-VL-7B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # one GPU: device_map=auto would otherwise split the model
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -d "$ROOT/scripts" ] || ROOT="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"
RESULT_DIR="results/${MODEL_DIR}/mmspec_${SPLIT}"
mkdir -p "$RESULT_DIR/logs"

# Medusa's script points at a local folder; use the released checkpoint instead.
export MEDUSA_CKPT="${MEDUSA_CKPT:-ping69852/Medusa-Qwen2.5-VL-7B-Instruct}"

# Environment record (reproducibility: 10% of the grade)
{
  echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "commit: $(git rev-parse HEAD 2>/dev/null || echo n/a)"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  python -c "import torch,transformers;print('torch',torch.__version__,'cuda',torch.version.cuda,'transformers',transformers.__version__)"
  pip freeze | grep -Ei '^(torch|transformers|accelerate|qwen-vl-utils|flash)' || true
} > "$RESULT_DIR/logs/env.txt"
cat "$RESULT_DIR/logs/env.txt"

for m in "${METHODS[@]}"; do
  s="scripts/${MODEL_DIR}/eval_${m}_mmspec.sh"
  if [ "$m" = "baseline_greedy" ]; then
    # pure-greedy COPY of the baseline (repetition_penalty forced to 1.0); original untouched
    python "$(dirname "${BASH_SOURCE[0]}")/make_baseline_greedy.py" "$MODEL_DIR" \
      || { echo "!! could not build baseline_greedy — skipping"; printf "baseline_greedy\t99\t%s\n" "$(date -Is)" >> "$RESULT_DIR/logs/status.tsv"; continue; }
  fi
  if [ "$m" = "msd_e2e" ]; then
    # regenerate the instrumented COPY every time so it never goes stale; originals untouched
    python "$(dirname "${BASH_SOURCE[0]}")/make_msd_e2e.py" "$MODEL_DIR" \
      || { echo "!! could not build msd_e2e — skipping"; printf "msd_e2e\t99\t%s\n" "$(date -Is)" >> "$RESULT_DIR/logs/status.tsv"; continue; }
  fi
  [ -f "$s" ] || { echo "skip: no $s"; continue; }
  echo "=== $m ($SPLIT) $(date +%T) ==="
  if [ "$m" = "medusa" ]; then
    # patched copy must live in scripts/<model>/ because each script finds the repo root from its own path
    sed "s#^SPEC_MODEL=.*#SPEC_MODEL=\"${MEDUSA_CKPT}\"#" "$s" > "scripts/${MODEL_DIR}/eval_medusa_hf_mmspec.sh"
    s="scripts/${MODEL_DIR}/eval_medusa_hf_mmspec.sh"
  fi
  rc=0
  ( time bash "$s" "$SPLIT" ) > "$RESULT_DIR/logs/${m}.log" 2>&1 || rc=$?
  # one line per attempt: method, exit code, finish time. The summary reads the LAST line per method.
  printf "%s\t%s\t%s\n" "$m" "$rc" "$(date -Is)" >> "$RESULT_DIR/logs/status.tsv"
  if [ "$rc" -ne 0 ]; then
    echo "!! $m FAILED (exit $rc) — see $RESULT_DIR/logs/${m}.log. Rerun it: finished samples are skipped."
  fi
  tail -n 2 "$RESULT_DIR/logs/${m}.log"
done

# Summary marks any method that failed or has fewer samples than the split and leaves it out of the table.
python "$(dirname "${BASH_SOURCE[0]}")/summarize_mmspec.py" "$RESULT_DIR" || true
if [ -f "$RESULT_DIR/msd-temperature-0.jsonl" ] && [ -f "$RESULT_DIR/msd_e2e-temperature-0.jsonl" ]; then
  python "$(dirname "${BASH_SOURCE[0]}")/validate_msd_e2e.py" "$RESULT_DIR" || true
fi
