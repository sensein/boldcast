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

# Resolve the checkout without hardcoding it: BOLDCAST_REPO wins, then the
# sbatch submission directory, then this script's own location.
REPO="${BOLDCAST_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
[ -f "$REPO/configs/demo.yaml" ] || { echo "not a boldcast checkout: $REPO" >&2; exit 1; }
cd "$REPO"
set -a; [ -f .env ] && . ./.env; set +a

set +u
source ~/.bashrc
micromamba activate "${BOLDCAST_ENV:?set BOLDCAST_ENV in .env to the micromamba env prefix}"
set -u

mkdir -p logs results

python -u scripts/day4_overfit.py \
    --config configs/demo.yaml \
    --lr 1e-3 \
    --max-steps 3000 \
    --out-dir results/day4_sanity_lr1e3
