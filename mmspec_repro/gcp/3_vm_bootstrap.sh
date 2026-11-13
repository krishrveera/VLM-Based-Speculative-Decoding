#!/bin/bash
# Run ON THE VM after `gcloud compute ssh`. Sets up MMSpec exactly as the runbook pins it,
# pre-downloads weights, and runs the sanity checks. Safe to re-run.
#
#   bash 3_vm_bootstrap.sh            # env + clone + downloads + sanity
#   SKIP_DOWNLOAD=1 bash 3_vm_bootstrap.sh
set -euo pipefail
WORK="$HOME/vlm-specdec"; mkdir -p "$WORK"; cd "$WORK"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$WORK/hf_cache"

echo ">> GPU"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# 1) Python env pinned to the repo (torch 2.7.0, transformers 4.51.3)
if [ ! -d venv ]; then
  python3 -m venv venv
fi
source venv/bin/activate
pip install -q --upgrade pip

# 2) MMSpec at the audited commit
if [ ! -d MMSpec ]; then
  git clone https://github.com/killthefullmoon/MMSpec.git
fi
cd MMSpec
git checkout -q 290486861eabd746075ca776adb66175636c1fc0
pip install -q -r requirements.txt
pip install -q "huggingface_hub[cli]" hf_transfer
python -c "import torch,transformers;print('torch',torch.__version__,'| cuda ok:',torch.cuda.is_available(),'| transformers',transformers.__version__)"

# our tools (copy summarize_mmspec.py + run_mmspec.sh next to this script before running)
mkdir -p tools
for f in summarize_mmspec.py run_mmspec.sh; do
  [ -f "$HOME/$f" ] && cp "$HOME/$f" tools/ || true
done

# 3) Weights (~16 GB target + ~1 GB per drafter)
if [ -z "${SKIP_DOWNLOAD:-}" ]; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
  huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
  for m in EAGLE EAGLE3 ViSpec MSD; do huggingface-cli download "Cloudriver/$m-Qwen2.5-VL-7B-Instruct"; done
  huggingface-cli download ping69852/Medusa-Qwen2.5-VL-7B-Instruct
  huggingface-cli download JLKang/ViSpec-Qwen2.5-VL-7B-Instruct
fi

# 4) Sanity checks: load + 1 sample, no timing. Logs to sanity/<method>.log
mkdir -p sanity
B=Qwen/Qwen2.5-VL-7B-Instruct; D=dataset/MMSpec/testmini; A=/tmp/sanity.jsonl
run() { local name=$1; shift; echo ">> sanity: $name"
  if python -m "evaluation.eval_${name}_mmspec" --base-model-path "$B" --data-folder "$D" --answer-file "$A" --sanity "$@" \
      > "sanity/${name}.log" 2>&1; then
    grep -E "PASSED|FAILED|missing|unexpected" "sanity/${name}.log" | head -5
  else echo "!! $name crashed — see sanity/${name}.log"; tail -5 "sanity/${name}.log"; fi; }

run baseline
run vispec   --spec-model-path Cloudriver/ViSpec-Qwen2.5-VL-7B-Instruct
run msd      --msd-model-path  Cloudriver/MSD-Qwen2.5-VL-7B-Instruct --use-msd
run eagle    --spec-model-path Cloudriver/EAGLE-Qwen2.5-VL-7B-Instruct
run eagle2   --spec-model-path Cloudriver/EAGLE-Qwen2.5-VL-7B-Instruct
run pld
run sam

echo
echo "Done. If all show PASSED with no missing keys, start testmini:"
echo "  cd $WORK/MMSpec && tmux new -s mmspec"
echo "  bash tools/run_mmspec.sh testmini baseline vispec msd eagle eagle2 pld"
echo "Copy results off the VM:  gsutil -m cp -r results gs://\$(gcloud config get-value project)-results/"
