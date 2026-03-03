"""
Multi-GPU throughput scaling benchmark for BOLDcast.

Measures forward + backward pass throughput across 1, 2, 4, 8 GPUs
using PyTorch DDP. Produces data for scaling plots required by ORCD proposal.

Usage:
    # Single GPU baseline
    python benchmarks/scaling_throughput.py --n_gpus 1

    # Multi-GPU (launched via torchrun)
    torchrun --nproc_per_node=2 benchmarks/scaling_throughput.py --n_gpus 2
    torchrun --nproc_per_node=4 benchmarks/scaling_throughput.py --n_gpus 4
    torchrun --nproc_per_node=8 benchmarks/scaling_throughput.py --n_gpus 8

    # Full sweep (submit via SLURM)
    sbatch benchmarks/run_scaling_sweep.sh

    # With custom config
    torchrun --nproc_per_node=4 benchmarks/scaling_throughput.py \\
        --n_gpus 4 --seq_len 256 --hidden_dim 256 --batch_size 4
        --output $SCRATCH/output/scaling_results.jsonl
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class BenchmarkResult:
    n_gpus: int
    seq_len: int
    n_spatial_tokens: int
    hidden_dim: int
    batch_size_per_gpu: int
    effective_batch_size: int
    avg_step_time_ms: float
    std_step_time_ms: float
    throughput_sequences_per_sec: float
    throughput_trs_per_sec: float
    peak_memory_gb: float
    gpu_name: str


class SimpleMambaBlock(nn.Module):
    """
    Simplified Mamba-like temporal block for benchmarking.

    Uses a causal conv1d + gated linear unit pattern that approximates
    Mamba's compute profile without requiring the mamba-ssm package.
    Replace with actual Mamba block for real training.
    """

    def __init__(self, d_model: int, expand: int = 2, d_conv: int = 4):
        super().__init__()
        d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True
        )
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, D)
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # Causal conv
        x = x.transpose(1, 2)  # (B, D_inner, T)
        x = self.conv1d(x)[:, :, :residual.shape[1]]
        x = x.transpose(1, 2)

        x = torch.silu(x) * torch.silu(z)
        x = self.out_proj(x)
        return x + residual


class SpatialMixingBlock(nn.Module):
    """
    Simplified spatial mixing via linear attention over spatial tokens.
    Approximates kNN message passing compute cost.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B*T, P, D)
        residual = x
        x = self.norm(x)
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Linear attention approximation (avoids O(P^2) for benchmarking)
        # Real implementation uses kNN sparse attention
        k = torch.softmax(k, dim=-1)
        kv = torch.einsum("bpd,bpe->bde", k, v)
        x = torch.einsum("bpd,bde->bpe", q, kv)
        x = self.out_proj(x)
        return x + residual


class BOLDcastBenchmarkModel(nn.Module):
    """
    Benchmark-ready model approximating BOLDcast architecture.

    Spatial tokens → interleaved Mamba temporal + spatial mixing blocks.
    """

    def __init__(
        self,
        n_spatial_tokens: int = 1792,
        hidden_dim: int = 256,
        n_temporal_layers: int = 8,
        n_spatial_layers: int = 4,
    ):
        super().__init__()
        self.n_spatial_tokens = n_spatial_tokens
        self.hidden_dim = hidden_dim

        # Input projection (simulate patch embedding)
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)

        # Interleaved temporal and spatial blocks.
        # nn.ModuleList only accepts nn.Module instances, so store temporal
        # and spatial blocks in separate lists and track order separately.
        self.temporal_blocks = nn.ModuleList()
        self.spatial_blocks = nn.ModuleList()
        self._block_order: list[tuple[str, int]] = []  # ("temporal"|"spatial", list_idx)

        spatial_interval = max(1, n_temporal_layers // n_spatial_layers)
        for i in range(n_temporal_layers):
            self.temporal_blocks.append(SimpleMambaBlock(hidden_dim))
            self._block_order.append(("temporal", len(self.temporal_blocks) - 1))
            if (i + 1) % spatial_interval == 0:
                self.spatial_blocks.append(SpatialMixingBlock(hidden_dim))
                self._block_order.append(("spatial", len(self.spatial_blocks) - 1))

        # Output head (forecasting)
        self.head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, P, D) — batch, time, spatial tokens, hidden dim
        """
        B, T, P, D = x.shape

        x = self.input_proj(x)

        for block_type, idx in self._block_order:
            if block_type == "temporal":
                block = self.temporal_blocks[idx]
                # Reshape to (B*P, T, D) for temporal processing
                x = x.permute(0, 2, 1, 3).reshape(B * P, T, D)
                x = block(x)
                x = x.reshape(B, P, T, D).permute(0, 2, 1, 3)
            else:
                block = self.spatial_blocks[idx]
                # Reshape to (B*T, P, D) for spatial processing
                x = x.reshape(B * T, P, D)
                x = block(x)
                x = x.reshape(B, T, P, D)

        return self.head(x)


def setup_distributed():
    """Initialize DDP if running multi-GPU."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, True
    else:
        torch.cuda.set_device(0)
        return 0, 0, False


def run_benchmark(
    n_gpus: int = 1,
    seq_len: int = 256,
    n_spatial_tokens: int = 1792,
    hidden_dim: int = 256,
    batch_size_per_gpu: int = 2,
    n_warmup: int = 5,
    n_steps: int = 20,
) -> BenchmarkResult:
    """Run throughput benchmark."""
    rank, local_rank, is_distributed = setup_distributed()

    device = torch.device(f"cuda:{local_rank}")
    gpu_name = torch.cuda.get_device_name(device)

    model = BOLDcastBenchmarkModel(
        n_spatial_tokens=n_spatial_tokens,
        hidden_dim=hidden_dim,
    ).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Synthetic data
    def make_batch():
        return torch.randn(
            batch_size_per_gpu, seq_len, n_spatial_tokens, hidden_dim,
            device=device, dtype=torch.bfloat16
        )

    model = model.to(torch.bfloat16)

    # Warmup
    for _ in range(n_warmup):
        x = make_batch()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
            loss = out.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    # Benchmark
    step_times = []
    for _ in range(n_steps):
        x = make_batch()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
            loss = out.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_times.append((t1 - t0) * 1000)  # ms

    peak_mem = torch.cuda.max_memory_allocated(device) / (1024**3)

    avg_time = sum(step_times) / len(step_times)
    std_time = (sum((t - avg_time) ** 2 for t in step_times) / len(step_times)) ** 0.5
    effective_batch = batch_size_per_gpu * n_gpus
    throughput_seq = effective_batch / (avg_time / 1000)
    throughput_trs = throughput_seq * seq_len

    result = BenchmarkResult(
        n_gpus=n_gpus,
        seq_len=seq_len,
        n_spatial_tokens=n_spatial_tokens,
        hidden_dim=hidden_dim,
        batch_size_per_gpu=batch_size_per_gpu,
        effective_batch_size=effective_batch,
        avg_step_time_ms=round(avg_time, 2),
        std_step_time_ms=round(std_time, 2),
        throughput_sequences_per_sec=round(throughput_seq, 2),
        throughput_trs_per_sec=round(throughput_trs, 2),
        peak_memory_gb=round(peak_mem, 2),
        gpu_name=gpu_name,
    )

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"BOLDcast Throughput Benchmark ({n_gpus} GPU{'s' if n_gpus > 1 else ''})")
        print(f"{'='*60}")
        print(f"GPU: {gpu_name}")
        print(f"Config: {seq_len} TRs × {n_spatial_tokens} tokens × dim {hidden_dim}")
        print(f"Batch: {batch_size_per_gpu}/GPU × {n_gpus} GPUs = {effective_batch}")
        print(f"Step time: {avg_time:.1f} ± {std_time:.1f} ms")
        print(f"Throughput: {throughput_seq:.2f} seq/s | {throughput_trs:.0f} TRs/s")
        print(f"Peak memory: {peak_mem:.2f} GB")
        print(f"{'='*60}\n")

    if is_distributed:
        dist.destroy_process_group()

    return result


def main():
    parser = argparse.ArgumentParser(description="BOLDcast throughput benchmark")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--n_spatial_tokens", type=int, default=1792)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--n_warmup", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    result = run_benchmark(
        n_gpus=args.n_gpus,
        seq_len=args.seq_len,
        n_spatial_tokens=args.n_spatial_tokens,
        hidden_dim=args.hidden_dim,
        batch_size_per_gpu=args.batch_size,
        n_warmup=args.n_warmup,
        n_steps=args.n_steps,
    )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        # Append to JSONL for collecting multi-GPU sweep results
        with open(args.output, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")
        print(f"Result appended to {args.output}")


if __name__ == "__main__":
    main()