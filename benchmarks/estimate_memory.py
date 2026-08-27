"""
GPU memory estimation for BOLDcast model configurations.

Estimates activation memory, parameter memory, and optimizer state memory
for different architecture configurations and hardware targets (H200, B200).

Usage:
    python benchmarks/estimate_memory.py
    python benchmarks/estimate_memory.py --hidden_dim 512 --n_layers 12
    python benchmarks/estimate_memory.py --output "$SCRATCH/output/memory_estimates.json"
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)
_scratch_dir = os.getenv("SCRATCH_DIR")
if not _scratch_dir or not Path(_scratch_dir).is_absolute():
    raise RuntimeError(
        f"SCRATCH_DIR must be set to an absolute path (got {_scratch_dir!r}). "
        "Check .env or the shell environment."
    )
output_dir = Path(_scratch_dir) / "output/benchmarks/"


@dataclass
class ModelConfig:
    """Architecture configuration."""

    n_spatial_tokens: int = 1792  # ~1024 cortical + ~768 subcortical
    hidden_dim: int = 256
    n_temporal_layers: int = 8  # Mamba layers
    n_spatial_layers: int = 4  # Spatial mixing layers (interleaved)
    mamba_state_dim: int = 16  # SSM state dimension (N in Mamba)
    mamba_expand_factor: int = 2  # Inner dimension = expand * hidden_dim
    spatial_knn_k: int = 16  # kNN neighbors for spatial mixing
    seq_len: int = 256  # TRs per training sequence
    dtype_bytes: int = 2  # BF16 = 2 bytes


@dataclass
class HardwareConfig:
    """GPU hardware specs."""

    name: str
    memory_gb: float
    fp8_support: bool = False


# Hardware targets
# H100 = HardwareConfig(name="H100", memory_gb=141.0, fp8_support=False)
H200 = HardwareConfig(name="H200", memory_gb=141.0, fp8_support=False)
B200 = HardwareConfig(name="B200", memory_gb=192.0, fp8_support=True)


@dataclass
class MemoryEstimate:
    """Memory breakdown for a single training sequence."""

    config_name: str
    # Parameter memory
    param_memory_gb: float = 0.0
    optimizer_memory_gb: float = 0.0  # Adam: 2x param memory (m, v)
    gradient_memory_gb: float = 0.0
    # Activation memory
    spatial_activation_gb: float = 0.0
    temporal_activation_gb: float = 0.0
    input_data_gb: float = 0.0
    # Totals
    total_per_sequence_gb: float = 0.0
    total_per_sequence_checkpointed_gb: float = 0.0
    # Fixed overhead (params + optimizer + gradients)
    fixed_overhead_gb: float = 0.0
    # Batch size estimates
    batch_sizes: dict = field(default_factory=dict)


def count_parameters(cfg: ModelConfig) -> int:
    """Estimate total trainable parameters."""
    d = cfg.hidden_dim
    d_inner = d * cfg.mamba_expand_factor
    n_ssm = cfg.mamba_state_dim

    params = 0

    # --- Mamba temporal layers ---
    # Each Mamba layer: in_proj (d -> 2*d_inner), conv1d (d_inner, kernel=4),
    # x_proj (d_inner -> dt_rank + 2*n_ssm), dt_proj (dt_rank -> d_inner),
    # out_proj (d_inner -> d), plus A, D parameters
    dt_rank = max(d // 16, 1)  # typical default
    per_mamba_layer = (
        d * 2 * d_inner  # in_proj
        + d_inner * 4  # conv1d (kernel=4)
        + d_inner * (dt_rank + 2 * n_ssm)  # x_proj
        + dt_rank * d_inner  # dt_proj
        + d_inner * d  # out_proj
        + d_inner * n_ssm  # A
        + d_inner  # D
        + 2 * d  # LayerNorm
    )
    params += cfg.n_temporal_layers * per_mamba_layer

    # --- Spatial mixing layers (kNN message passing) ---
    # Each: linear projections for Q, K, V (d -> d), output proj (d -> d), LayerNorm
    per_spatial_layer = (
        3 * d * d  # Q, K, V projections
        + d * d  # output projection
        + 2 * d  # LayerNorm
    )
    params += cfg.n_spatial_layers * per_spatial_layer

    # --- Embedding and head ---
    # Input projection: n_spatial_tokens features -> d (per patch)
    # We assume patches have ~50 vertices each on average, projected to d
    avg_patch_size = 50
    params += avg_patch_size * d  # input embedding
    params += d * avg_patch_size  # output decoding head
    params += d * d  # contrastive projection head

    # --- Stimulus encoder projection ---
    clip_dim = 768  # CLIP ViT-L/14
    params += clip_dim * d + d  # stimulus projection
    params += d * 4 * d + d * d  # alignment MLP (small)

    # --- Structural conditioning ---
    n_morph_features = 3  # thickness, curvature, sulcal depth
    params += n_morph_features * d + d  # morphometric projection

    return params


def estimate_activation_memory(cfg: ModelConfig) -> tuple[float, float, float]:
    """
    Estimate activation memory in GB for a single sequence.

    Returns:
        (spatial_gb, temporal_gb, input_gb)
    """
    P = cfg.n_spatial_tokens
    T = cfg.seq_len
    d = cfg.hidden_dim
    b = cfg.dtype_bytes

    # --- Input data ---
    # Tokenized input: T × P × d × dtype_bytes
    input_gb = (T * P * d * b) / (1024**3)

    # --- Spatial mixing activations ---
    # Per spatial layer: store Q, K, V (P × d each), attention scores (P × k for kNN),
    # output (P × d), plus intermediates. Across T timepoints.
    # With kNN (k neighbors), attention is sparse: P × k instead of P × P
    k = cfg.spatial_knn_k
    per_spatial_layer_bytes = T * (
        3 * P * d * b  # Q, K, V
        + P * k * b  # sparse attention scores
        + P * d * b  # output
        + P * d * b  # residual / intermediate
    )
    spatial_gb = (cfg.n_spatial_layers * per_spatial_layer_bytes) / (1024**3)

    # If using full attention instead of kNN (worst case):
    # per_spatial_layer_bytes_full = T * (3 * P * d * b + P * P * b + P * d * b)
    # spatial_gb_full = (cfg.n_spatial_layers * per_spatial_layer_bytes_full) / (1024**3)

    # --- Mamba temporal activations ---
    # Per Mamba layer: SSM state (P × d_inner × n_ssm), conv state (P × d_inner × kernel),
    # scan intermediates (T × P × d_inner), gate values (T × P × d_inner)
    d_inner = d * cfg.mamba_expand_factor
    n_ssm = cfg.mamba_state_dim
    per_temporal_layer_bytes = (
        T * P * d_inner * b  # scan output
        + T * P * d_inner * b  # gate values (z)
        + T * P * d_inner * b  # x after conv
        + P * d_inner * n_ssm * b  # SSM state (persistent)
        + P * d_inner * 4 * b  # conv1d state (kernel=4)
    )
    temporal_gb = (cfg.n_temporal_layers * per_temporal_layer_bytes) / (1024**3)

    return spatial_gb, temporal_gb, input_gb


def estimate_memory(
    cfg: ModelConfig,
    hardware_targets: list[HardwareConfig] | None = None,
    checkpointing_factor: float = 0.45,
) -> MemoryEstimate:
    """
    Full memory estimation for a given model configuration.

    Args:
        cfg: Model configuration
        hardware_targets: List of GPU configs to estimate batch sizes for
        checkpointing_factor: Fraction of activation memory retained with
            gradient checkpointing (typically 0.4-0.5 for sqrt(n) checkpointing)
    """
    if hardware_targets is None:
        hardware_targets = [H200, B200]

    est = MemoryEstimate(config_name=f"T{cfg.seq_len}_P{cfg.n_spatial_tokens}_d{cfg.hidden_dim}")

    # --- Parameter memory ---
    n_params = count_parameters(cfg)
    est.param_memory_gb = (n_params * cfg.dtype_bytes) / (1024**3)
    est.optimizer_memory_gb = (n_params * 4 * 2) / (1024**3)  # Adam m,v in FP32
    est.gradient_memory_gb = (n_params * cfg.dtype_bytes) / (1024**3)
    est.fixed_overhead_gb = est.param_memory_gb + est.optimizer_memory_gb + est.gradient_memory_gb

    # --- Activation memory ---
    spatial_gb, temporal_gb, input_gb = estimate_activation_memory(cfg)
    est.spatial_activation_gb = spatial_gb
    est.temporal_activation_gb = temporal_gb
    est.input_data_gb = input_gb

    activation_total = spatial_gb + temporal_gb + input_gb
    est.total_per_sequence_gb = activation_total
    est.total_per_sequence_checkpointed_gb = activation_total * checkpointing_factor

    # --- Batch size estimates per hardware ---
    for hw in hardware_targets:
        available = hw.memory_gb - est.fixed_overhead_gb
        # Reserve 10% for fragmentation and CUDA overhead
        usable = available * 0.90

        batch_no_ckpt = max(1, int(usable / est.total_per_sequence_gb))
        batch_ckpt = max(1, int(usable / est.total_per_sequence_checkpointed_gb))

        est.batch_sizes[hw.name] = {
            "no_checkpointing": batch_no_ckpt,
            "with_checkpointing": batch_ckpt,
            "effective_8gpu_no_ckpt": batch_no_ckpt * 8,
            "effective_8gpu_ckpt": batch_ckpt * 8,
        }

    return est


def format_report(est: MemoryEstimate, cfg: ModelConfig) -> str:
    """Format a human-readable memory report."""
    n_params = count_parameters(cfg)
    lines = [
        f"{'=' * 70}",
        f"BOLDcast Memory Estimation: {est.config_name}",
        f"{'=' * 70}",
        "",
        f"Model: {n_params:,} parameters ({n_params / 1e6:.1f}M)",
        f"Sequence: {cfg.seq_len} TRs × {cfg.n_spatial_tokens} spatial tokens "
        f"× dim {cfg.hidden_dim}",
        f"Temporal: {cfg.n_temporal_layers} Mamba layers (state_dim={cfg.mamba_state_dim})",
        f"Spatial: {cfg.n_spatial_layers} kNN mixing layers (k={cfg.spatial_knn_k})",
        "",
        "--- Fixed Memory (per GPU) ---",
        f"  Parameters:       {est.param_memory_gb:.3f} GB",
        f"  Optimizer (Adam): {est.optimizer_memory_gb:.3f} GB",
        f"  Gradients:        {est.gradient_memory_gb:.3f} GB",
        f"  Total fixed:      {est.fixed_overhead_gb:.3f} GB",
        "",
        "--- Activation Memory (per sequence) ---",
        f"  Input data:              {est.input_data_gb:.3f} GB",
        f"  Spatial mixing:          {est.spatial_activation_gb:.3f} GB",
        f"  Temporal (Mamba):        {est.temporal_activation_gb:.3f} GB",
        f"  Total (no checkpointing):    {est.total_per_sequence_gb:.2f} GB",
        f"  Total (with checkpointing):  {est.total_per_sequence_checkpointed_gb:.2f} GB",
        "",
        "--- Batch Size Estimates ---",
    ]

    for hw_name, sizes in est.batch_sizes.items():
        lines.extend(
            [
                f"  {hw_name}:",
                f"    Per GPU (no ckpt):     {sizes['no_checkpointing']}",
                f"    Per GPU (with ckpt):   {sizes['with_checkpointing']}",
                f"    8-GPU effective (no):  {sizes['effective_8gpu_no_ckpt']}",
                f"    8-GPU effective (ckpt):{sizes['effective_8gpu_ckpt']}",
            ]
        )

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="BOLDcast GPU memory estimation")
    parser.add_argument("--seq_len", type=int, default=256, help="Training sequence length (TRs)")
    parser.add_argument("--n_spatial_tokens", type=int, default=1792)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_temporal_layers", type=int, default=8)
    parser.add_argument("--n_spatial_layers", type=int, default=4)
    parser.add_argument("--mamba_state_dim", type=int, default=16)
    parser.add_argument("--output", type=str, default=None, help="Save JSON results to file")
    parser.add_argument("--sweep", action="store_true", help="Run sweep over key configurations")
    args = parser.parse_args()

    if args.sweep:
        configs = [
            ("default", ModelConfig()),
            ("short_seq", ModelConfig(seq_len=128)),
            ("long_seq", ModelConfig(seq_len=512)),
            ("large_hidden", ModelConfig(hidden_dim=512, n_temporal_layers=12)),
            ("fewer_patches", ModelConfig(n_spatial_tokens=1024)),
            (
                "roi_baseline",
                ModelConfig(n_spatial_tokens=400, hidden_dim=128, n_temporal_layers=6),
            ),
        ]
        results = {}
        for name, cfg in configs:
            est = estimate_memory(cfg)
            print(format_report(est, cfg))
            print()
            results[name] = asdict(est)

        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            output_path = Path(output_dir / "memory_estimates.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_path}")
    else:
        cfg = ModelConfig(
            seq_len=args.seq_len,
            n_spatial_tokens=args.n_spatial_tokens,
            hidden_dim=args.hidden_dim,
            n_temporal_layers=args.n_temporal_layers,
            n_spatial_layers=args.n_spatial_layers,
            mamba_state_dim=args.mamba_state_dim,
        )
        est = estimate_memory(cfg)
        print(format_report(est, cfg))

        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(asdict(est), f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            output_path = Path(output_dir / "memory_estimates.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(asdict(est), f, indent=2)
            print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
