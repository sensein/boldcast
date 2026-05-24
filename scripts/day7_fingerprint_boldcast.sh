#!/usr/bin/env bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=day7_bc
#SBATCH --output=logs/day7_bc_%j.out
#SBATCH --error=logs/day7_bc_%j.err

set +u
source ~/.bashrc
micromamba activate $BOLDCAST_ENV
set -u

REPO=$REPO
cd "$REPO"
mkdir -p logs results/day7_fingerprint

export PYTHONUNBUFFERED=1
python scripts/day7_fingerprint_boldcast.py \
    --config configs/demo.yaml \
    --ckpt results/day5_train/ckpt_final.pt \
    --out-json results/day7_fingerprint/boldcast_metrics.json \
    --ci 0.95
