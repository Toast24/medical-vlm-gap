#!/bin/bash
#SBATCH --job-name=project-a
#SBATCH --partition=h100-mig
#SBATCH --gres=gpu:nvidia_h100_nvl_3g.47gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/project_a_%j.out
#SBATCH --error=logs/project_a_%j.err

set -u

PROJECT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Override this if the cluster provides Python through a module/conda environment.
PYTHON="${PYTHON:-python3}"

cd "$PROJECT" || exit 1

mkdir -p logs

echo "=== PROJECT A LINGSHU PILOT ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Project: $PROJECT"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi

echo "=== PYTHON ==="
"$PYTHON" --version

echo "=== RUNNING PROJECT A: 100 CONDITIONS ==="
"$PYTHON" run_project_a_lingshu.py

STATUS=$?

echo "=== PROJECT A EXIT CODE: $STATUS ==="
exit $STATUS
