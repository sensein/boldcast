"""Training subpackage: loss, optimizer/scheduler factories, Trainer, utils.

Re-exports the public API surface so Day-4 / Day-5 scripts can write
``from boldcast.training import Trainer, build_optimizer, ...`` rather
than importing each submodule by name.
"""

from boldcast.training.ddp import (
    get_local_rank,
    get_rank,
    get_world_size,
    is_distributed_run,
    is_rank_zero,
)
from boldcast.training.loss import build_forecast_targets, forecasting_loss
from boldcast.training.optim import build_optimizer, build_scheduler
from boldcast.training.trainer import Trainer
from boldcast.training.utils import JsonlLogger, save_checkpoint, seed_everything

__all__ = [
    "JsonlLogger",
    "Trainer",
    "build_forecast_targets",
    "build_optimizer",
    "build_scheduler",
    "forecasting_loss",
    "get_local_rank",
    "get_rank",
    "get_world_size",
    "is_distributed_run",
    "is_rank_zero",
    "save_checkpoint",
    "seed_everything",
]
