#!/bin/bash

#SBATCH --job-name=lingshu-batch
#SBATCH --partition=h100-mig
#SBATCH --gres=gpu:nvidia_h100_nvl_3g.47gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/lingshu-%j.out
#SBATCH --error=logs/lingshu-%j.err

set -e

PROJECT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python3}"

cd "$PROJECT" || exit 1

mkdir -p logs results

echo "=== LINGSHU BATCH JOB ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Project: $PROJECT"
echo "CUDA: ${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi

echo "=== PYTHON ==="
"$PYTHON" --version

echo "=== RUNNING LINGSHU BATCH ==="
"$PYTHON" run_lingshu_batch.py

echo "=== DONE ==="
