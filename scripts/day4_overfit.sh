#!/usr/bin/env bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --job-name=day4_overfit
#SBATCH --output=logs/day4_overfit_%j.out
#SBATCH --error=logs/day4_overfit_%j.err

# Partition: mit_normal_gpu (104 H200s across 13 nodes). pi_satra was
# the original Day-4 venue but had no GPU availability on 2026-05-13;
# mit_normal_gpu is the canonical Day-4 partition going forward.

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
    --out-dir results/day4_overfit
