#!/usr/bin/env bash
#SBATCH --partition=b200-batch
#SBATCH --account=all-sails_mit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --job-name=day3_validate
#SBATCH --output=logs/day3_validate_%j.out
#SBATCH --error=logs/day3_validate_%j.err

# Partition: b200-batch on AICR. For a quick interactive check,
# b200-devel (4 h, up to 2 GPUs) usually starts sooner.

# Resolve the checkout without hardcoding it: BOLDCAST_REPO wins, then the
# sbatch submission directory, then this script's own location.
REPO="${BOLDCAST_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
[ -f "$REPO/configs/demo.yaml" ] || { echo "not a boldcast checkout: $REPO" >&2; exit 1; }
cd "$REPO"
set -a; [ -f .env ] && . ./.env; set +a

source scripts/_activate_env.sh

mkdir -p logs

python -u scripts/day3_validate_model.py --config configs/demo.yaml
