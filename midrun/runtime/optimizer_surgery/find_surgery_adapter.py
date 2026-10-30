"""Finds the momentum-reset adapter for an optimizer."""

import torch

from midrun.runtime.optimizer_surgery.adam_surgery_adapter import (
    AdamSurgeryAdapter,
)
from midrun.runtime.optimizer_surgery.optimizer_surgery_adapter import (
    OptimizerSurgeryAdapter,
)
from midrun.runtime.optimizer_surgery.sgd_surgery_adapter import SGDSurgeryAdapter

_ADAPTERS: tuple[OptimizerSurgeryAdapter, ...] = (
    SGDSurgeryAdapter(),
    AdamSurgeryAdapter(),
)


def find_surgery_adapter(
    optimizer: torch.optim.Optimizer,
) -> OptimizerSurgeryAdapter | None:
    """Returns the first registered adapter supporting `optimizer`'s type, or `None`."""
    for adapter in _ADAPTERS:
        if adapter.supports(optimizer):
            return adapter
    return None
