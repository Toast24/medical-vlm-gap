#!/bin/bash
#SBATCH --job-name=pa_judge_pipe
#SBATCH --partition=h100-mig
#SBATCH --gres=gpu:nvidia_h100_nvl_3g.47gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# Build judge items for one model's results, then judge them.
# Submit from the project root with --export=ALL,TAG=<model tag>,RESULTS=<results .jsonl>
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
source slurm/local.env
export HF_HUB_OFFLINE=1
if [ -n "${HOME_OVERRIDE:-}" ]; then
  export HOME="$HOME_OVERRIDE" HF_HOME="$HOME_OVERRIDE/.cache/huggingface" XDG_CACHE_HOME="$HOME_OVERRIDE/.cache"
fi
P=$PWD
export JUDGE_CONDITIONS=$P/data/project_a_${DS:-scaled}_conditions.jsonl
export JUDGE_SRC_REFS=$P/results/project_a_${DS:-scaled}_source_references.jsonl
export JUDGE_RESULTS=${RESULTS:?set RESULTS} JUDGE_DIR=$P/results/project_a_judge_${DS:-scaled}_${TAG:?set TAG}
n=$("$PYTHON_LINGSHU" - "$RESULTS" << 'PY'
import json, sys
print(len({(str(r["target_study_id"]), r["condition"]) for r in map(json.loads, open(sys.argv[1])) if r.get("prediction")}))
PY
)
echo "successful outputs for $TAG: $n"
[ "$n" -ge 500 ] || { echo "fewer than 500 successful outputs; not judging"; exit 1; }
rm -rf "$JUDGE_DIR"
"$PYTHON_LINGSHU" evaluation/build_judge_items.py
"$PYTHON_LINGSHU" evaluation/run_judge_gpu.py
