#!/usr/bin/env bash
#SBATCH --partition=b200-batch
#SBATCH --account=all-sails_mit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --job-name=day4_overfit
#SBATCH --output=logs/day4_overfit_%j.out
#SBATCH --error=logs/day4_overfit_%j.err

# Partition: b200-batch on AICR (B200, 180 GB). The 2026-05 runs used
# H200 nodes on ORCD Engaging.

# Resolve the checkout without hardcoding it: BOLDCAST_REPO wins, then the
# sbatch submission directory, then this script's own location.
REPO="${BOLDCAST_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
[ -f "$REPO/configs/demo.yaml" ] || { echo "not a boldcast checkout: $REPO" >&2; exit 1; }
cd "$REPO"
set -a; [ -f .env ] && . ./.env; set +a

source scripts/_activate_env.sh

mkdir -p logs results

python -u scripts/day4_overfit.py \
    --config configs/demo.yaml \
    --lr 1e-3 \
    --max-steps 3000 \
    --out-dir results/day4_overfit
