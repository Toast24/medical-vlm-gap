#!/bin/bash

#SBATCH --job-name=maira2-batch
#SBATCH --partition=h100-mig
#SBATCH --gres=gpu:nvidia_h100_nvl_3g.47gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/maira2-%j.out
#SBATCH --error=logs/maira2-%j.err

set -e

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

cd "$PROJECT" || exit 1

mkdir -p logs results

echo "=== MAIRA-2 BATCH JOB ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Project: $PROJECT"
echo "CUDA: ${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi

echo "=== PYTHON ==="
"$PYTHON" --version

echo "=== RUNNING MAIRA-2 BATCH ==="
"$PYTHON" run_maira2_batch.py

echo "=== DONE ==="
