#!/bin/bash
#SBATCH --job-name=boldcast-scaling
#SBATCH --partition=ou_bcs_low,pi_satra
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --partition=ou_bcs_low
#SBATCH --output=logs/slurm-scaling-%j.out
#SBATCH --error=logs/slurm-scaling-%j.err

# Run multi-GPU scaling sweep for ORCD proposal benchmarks.
#
# Submit:
#   mkdir -p logs
#   sbatch benchmarks/run_scaling_sweep.sh
#
# Outputs:
#   $SCRATCH/output/scaling_results.jsonl  — one JSON line per GPU config
#   Use benchmarks/plot_scaling.py to generate figures.

set -euo pipefail

OUTPUT_DIR="${SCRATCH}/output"
OUTPUT_FILE="${OUTPUT_DIR}/scaling_results.jsonl"
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Activate micromamba environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate $BOLDCAST_ENV

echo "============================================"
echo "BOLDcast Multi-GPU Scaling Sweep"
echo "Node: $(hostname)"
echo "GPUs available: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "============================================"

# Clear previous results
> "$OUTPUT_FILE"

SEQ_LEN=256
N_TOKENS=1792
HIDDEN_DIM=256
BATCH_SIZE=2
N_STEPS=20

for N_GPUS in 1 2 4 8; do
    echo ""
    echo "--- Running with ${N_GPUS} GPU(s) ---"

    if [ "$N_GPUS" -eq 1 ]; then
        python benchmarks/scaling_throughput.py \
            --n_gpus "$N_GPUS" \
            --seq_len "$SEQ_LEN" \
            --n_spatial_tokens "$N_TOKENS" \
            --hidden_dim "$HIDDEN_DIM" \
            --batch_size "$BATCH_SIZE" \
            --n_steps "$N_STEPS" \
            --output "$OUTPUT_FILE"
    else
        torchrun --nproc_per_node="$N_GPUS" \
            benchmarks/scaling_throughput.py \
            --n_gpus "$N_GPUS" \
            --seq_len "$SEQ_LEN" \
            --n_spatial_tokens "$N_TOKENS" \
            --hidden_dim "$HIDDEN_DIM" \
            --batch_size "$BATCH_SIZE" \
            --n_steps "$N_STEPS" \
            --output "$OUTPUT_FILE"
    fi

    echo "--- ${N_GPUS} GPU(s) complete ---"
done

echo ""
echo "============================================"
echo "Sweep complete. Results in: $OUTPUT_FILE"
echo "To plot: python benchmarks/plot_scaling.py --input $OUTPUT_FILE --output ${SCRATCH}/output/scaling.pdf"
echo "============================================"
