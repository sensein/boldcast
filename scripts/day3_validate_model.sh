#!/usr/bin/env bash
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --job-name=day3_validate
#SBATCH --output=logs/day3_validate_%j.out
#SBATCH --error=logs/day3_validate_%j.err

# Partition: pi_satra (original Day-3 validation venue, 2026-05-13).
# If pi_satra is congested, swap to mit_normal_gpu (H200s) — see
# day4_overfit.sh for the canonical pattern.

set +u
source ~/.bashrc
micromamba activate $BOLDCAST_ENV
set -u

REPO=$REPO
cd "$REPO"
mkdir -p logs

python -u scripts/day3_validate_model.py --config configs/demo.yaml
