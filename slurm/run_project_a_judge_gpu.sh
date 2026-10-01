#!/bin/bash
#SBATCH --job-name=pa_judge_gpu
#SBATCH --partition=h100-mig
#SBATCH --gres=gpu:nvidia_h100_nvl_3g.47gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
# Submit from the project root: sbatch slurm/run_project_a_judge_gpu.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
source slurm/local.env
export HF_HUB_OFFLINE=1
if [ -n "${HOME_OVERRIDE:-}" ]; then
  export HOME="$HOME_OVERRIDE" HF_HOME="$HOME_OVERRIDE/.cache/huggingface" XDG_CACHE_HOME="$HOME_OVERRIDE/.cache"
fi
"$PYTHON_LINGSHU" evaluation/run_judge_gpu.py
