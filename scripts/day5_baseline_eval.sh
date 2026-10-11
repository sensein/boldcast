#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=day5_baseline
#SBATCH --output=logs/day5_baseline_%j.out
#SBATCH --error=logs/day5_baseline_%j.err

# Resolve the checkout without hardcoding it: BOLDCAST_REPO wins, then the
# sbatch submission directory, then this script's own location.
REPO="${BOLDCAST_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
[ -f "$REPO/configs/demo.yaml" ] || { echo "not a boldcast checkout: $REPO" >&2; exit 1; }
cd "$REPO"
set -a; [ -f .env ] && . ./.env; set +a

source scripts/_activate_env.sh

mkdir -p logs results/day5_train

export PYTHONUNBUFFERED=1
python scripts/day5_baseline_eval.py \
    --config configs/demo.yaml \
    --ckpt results/day5_train/ckpt_final.pt \
    --out-json results/day5_train/baseline_eval.json
