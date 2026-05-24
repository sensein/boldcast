#!/usr/bin/env bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=day5_baseline
#SBATCH --output=logs/day5_baseline_%j.out
#SBATCH --error=logs/day5_baseline_%j.err

set +u
source ~/.bashrc
micromamba activate $BOLDCAST_ENV
set -u

REPO=$REPO
cd "$REPO"
mkdir -p logs results/day5_train

export PYTHONUNBUFFERED=1
python scripts/day5_baseline_eval.py \
    --config configs/demo.yaml \
    --ckpt results/day5_train/ckpt_final.pt \
    --out-json results/day5_train/baseline_eval.json
