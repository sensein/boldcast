#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=day6_cache
#SBATCH --output=logs/day6_cache_%j.out
#SBATCH --error=logs/day6_cache_%j.err

# Resolve the checkout without hardcoding it: BOLDCAST_REPO wins, then the
# sbatch submission directory, then this script's own location.
REPO="${BOLDCAST_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
[ -f "$REPO/configs/demo.yaml" ] || { echo "not a boldcast checkout: $REPO" >&2; exit 1; }
cd "$REPO"
set -a; [ -f .env ] && . ./.env; set +a

source scripts/_activate_env.sh

mkdir -p logs

export PYTHONUNBUFFERED=1
python scripts/day6_build_schaefer_cache.py --config configs/demo.yaml
