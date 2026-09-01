#!/usr/bin/env bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --job-name=day6_train
#SBATCH --output=logs/day6_train_%j.out
#SBATCH --error=logs/day6_train_%j.err

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

export PYTHONUNBUFFERED=1
torchrun --standalone --nproc-per-node=2 --nnodes=1 \
    scripts/day6_train_schaefer.py \
    --config configs/demo.yaml \
    --out-dir results/day6_baseline
