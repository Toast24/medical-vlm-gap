#!/bin/bash
# Background-safe, resumable model downloads (Hugging Face). Run from the project root.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=logs/downloads; mkdir -p "$LOG"
source slurm/local.env
HF_CLI="$(dirname "$PYTHON_JUDGE")/hf"; [ -x "$HF_CLI" ] || HF_CLI="$(dirname "$PYTHON_JUDGE")/huggingface-cli"
MODELS=(
  "Qwen/Qwen2.5-VL-7B-Instruct             qwen2.5-vl-7b-instruct"
  "StanfordAIMI/CheXagent-2-3b             chexagent-2-3b"
  "StanfordAIMI/GREEN-RadLlama2-7b         green-radllama2-7b"
  "FreedomIntelligence/HuatuoGPT-Vision-7B huatuogpt-vision-7b"
)
for entry in "${MODELS[@]}"; do
  read -r repo dir <<< "$entry"
  if [ -f "$LOG/model_$dir.done" ]; then echo "[$(date "+%F %T")] $repo: already done, skipped"; continue; fi
  echo "[$(date "+%F %T")] $repo: START"
  if "$HF_CLI" download "$repo" --local-dir "models/$dir" > "$LOG/model_$dir.log" 2>&1; then
    touch "$LOG/model_$dir.done"; echo "[$(date "+%F %T")] $repo: DONE ($(du -sh "models/$dir" | cut -f1))"
  else
    echo "[$(date "+%F %T")] $repo: FAILED (see $LOG/model_$dir.log)"
  fi
done
echo "[$(date "+%F %T")] model downloads finished"
