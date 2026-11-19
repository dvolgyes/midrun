"""The torchinfo summary of a model, traced with a live batch."""

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

from midrun.runtime.torchinfo_inspector.parameter_totals import (
    parameter_totals,
)

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule


def torchinfo_rich_summary(
    pl_module: "LightningModule",
    batch: torch.Tensor | Sequence[object] | Mapping[str, object],
) -> dict[str, str]:
    """A forward-traced summary using the live in-flight `batch` as input.

    Args:
        pl_module: The composed `LightningModule`.
        batch: The batch to trace shapes with, e.g. `runtime.current_step.batch`.

    Returns:
        `{"summary.txt": <text>, "summary.json": <json>}`.

    Raises:
        ImportError: torchinfo is not installed.
    """
    import torchinfo

    stats = torchinfo.summary(pl_module, input_data=batch, verbose=0)
    stats.total_params, stats.trainable_params, stats.total_param_bytes = (
        parameter_totals(pl_module.parameters())
    )
    payload = {
        "total_params": stats.total_params,
        "trainable_params": stats.trainable_params,
        "total_mult_adds": stats.total_mult_adds,
    }
    return {"summary.txt": str(stats), "summary.json": json.dumps(payload, indent=2)}
