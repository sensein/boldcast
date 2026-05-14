#!/usr/bin/env bash
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=day4_sanity
#SBATCH --output=logs/day4_sanity_%j.out
#SBATCH --error=logs/day4_sanity_%j.err

# Day-4 ADR-0005-D5 retest: probe the empirical MSE floor at higher LR
# and longer run, to inform spec-acceptance update (Path A).
#
# Historical: this script reproduces the 2026-05-13 sanity-check run
# (job 13888490) that informed ADR 0005 D8. After the spec revision the
# canonical Day-4 run lives in day4_overfit.sh; this script is preserved
# for reproducibility, hence the original pi_satra partition.

set +u
source ~/.bashrc
micromamba activate $BOLDCAST_ENV
set -u

REPO=$REPO
cd "$REPO"
mkdir -p logs results

python -u scripts/day4_overfit.py \
    --config configs/demo.yaml \
    --lr 1e-3 \
    --max-steps 3000 \
    --out-dir results/day4_sanity_lr1e3
