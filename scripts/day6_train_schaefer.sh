#!/usr/bin/env bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=day6_train
#SBATCH --output=logs/day6_train_%j.out
#SBATCH --error=logs/day6_train_%j.err

set +u
source ~/.bashrc
micromamba activate $BOLDCAST_ENV
set -u

REPO=$REPO
cd "$REPO"
mkdir -p logs results

export PYTHONUNBUFFERED=1
torchrun --standalone --nproc-per-node=2 --nnodes=1 \
    scripts/day6_train_schaefer.py \
    --config configs/demo.yaml \
    --out-dir results/day6_baseline
