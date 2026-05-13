#!/usr/bin/env bash
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=day4_overfit
#SBATCH --output=logs/day4_overfit_%j.out
#SBATCH --error=logs/day4_overfit_%j.err

set +u
source ~/.bashrc
micromamba activate $BOLDCAST_ENV
set -u

REPO=$REPO
cd "$REPO"
mkdir -p logs results

python -u scripts/day4_overfit.py \
    --config configs/demo.yaml \
    --max-steps 1000 \
    --out-dir results/day4_overfit
