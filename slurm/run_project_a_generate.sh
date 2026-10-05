#!/bin/bash
#SBATCH --job-name=pa_gen
#SBATCH --partition=h100-mig
#SBATCH --gres=gpu:nvidia_h100_nvl_3g.47gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
# Generic generation job. Submit from the project root with --export=ALL,RUNNER=<script.py>,<runner env...>
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
source slurm/local.env
export HF_HUB_OFFLINE=1
if [ -n "${HOME_OVERRIDE:-}" ]; then
  export HOME="$HOME_OVERRIDE" HF_HOME="$HOME_OVERRIDE/.cache/huggingface" XDG_CACHE_HOME="$HOME_OVERRIDE/.cache"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "runner: ${RUNNER:?set RUNNER} | output: ${PROJECT_A_OUTPUT:-default}"
PY_VAR=${PY_ENV:-PYTHON_LINGSHU}; PY="${!PY_VAR}"
echo "python: $PY_VAR -> $PY"
"$PY" "$RUNNER"
