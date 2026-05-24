#!/usr/bin/env bash
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=day6_cache
#SBATCH --output=logs/day6_cache_%j.out
#SBATCH --error=logs/day6_cache_%j.err

set +u
source ~/.bashrc
micromamba activate $BOLDCAST_ENV
set -u

REPO=$REPO
cd "$REPO"
mkdir -p logs

export PYTHONUNBUFFERED=1
python scripts/day6_build_schaefer_cache.py --config configs/demo.yaml
