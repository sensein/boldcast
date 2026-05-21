"""Training subpackage: loss, optimizer/scheduler factories, Trainer, utils.

Re-exports the public API surface so Day-4 / Day-5 scripts can write
``from boldcast.training import Trainer, build_optimizer, ...`` rather
than importing each submodule by name.
"""

from boldcast.training.ddp import (
    cleanup_distributed,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed_run,
    is_rank_zero,
    setup_model_for_ddp,
)
from boldcast.training.loss import build_forecast_targets, forecasting_loss
from boldcast.training.optim import build_optimizer, build_scheduler
from boldcast.training.trainer import Trainer
from boldcast.training.utils import (
    JsonlLogger,
    heldout_decreased_by,
    save_checkpoint,
    seed_everything,
)

__all__ = [
    "JsonlLogger",
    "Trainer",
    "build_forecast_targets",
    "build_optimizer",
    "build_scheduler",
    "cleanup_distributed",
    "forecasting_loss",
    "get_local_rank",
    "get_rank",
    "get_world_size",
    "heldout_decreased_by",
    "init_distributed",
    "is_distributed_run",
    "is_rank_zero",
    "save_checkpoint",
    "seed_everything",
    "setup_model_for_ddp",
]
