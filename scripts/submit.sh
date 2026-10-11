#!/usr/bin/env bash
# Submit an sbatch wrapper with this machine's partition and account.
#
# Tracked wrappers request resources only (CPUs, memory, time, GPU count).
# Partition and account are site settings, read from the gitignored .env:
#
#   BOLDCAST_SLURM_ACCOUNT   account to charge (optional)
#   BOLDCAST_GPU_PARTITION   partition for wrappers that request a GPU
#   BOLDCAST_CPU_PARTITION   partition for the others
#
# They reach sbatch as SBATCH_ACCOUNT / SBATCH_PARTITION, which override the
# script's own directives.
#
# Usage: scripts/submit.sh scripts/day5_train_boldcast.sh [args...]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script="${1:?usage: scripts/submit.sh <sbatch-script> [args...]}"
env_file="${BOLDCAST_ENV_FILE:-$REPO/.env}"
set -a
if [ -f "$env_file" ]; then . "$env_file"; fi
set +a

if grep -q '^#SBATCH --gres=gpu' "$script"; then
    partition="${BOLDCAST_GPU_PARTITION:-}"
else
    partition="${BOLDCAST_CPU_PARTITION:-}"
fi
if [ -n "$partition" ]; then export SBATCH_PARTITION="$partition"; fi
if [ -n "${BOLDCAST_SLURM_ACCOUNT:-}" ]; then export SBATCH_ACCOUNT="$BOLDCAST_SLURM_ACCOUNT"; fi

cd "$REPO"
exec sbatch "$@"
