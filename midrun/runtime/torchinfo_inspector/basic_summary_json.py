"""The parameter counts of a model."""

from typing import TYPE_CHECKING, Any

from midrun.runtime.torchinfo_inspector.parameter_totals import (
    parameter_totals,
)
from midrun.runtime.torchinfo_inspector.torchinfo_available import (
    torchinfo_available,
)

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule


def basic_summary_json(pl_module: "LightningModule") -> dict[str, Any]:
    """Return the known counts and mark them partial if any sizes are unknown."""
    counts = parameter_totals(pl_module.parameters())
    payload = {
        "total_params": counts.total,
        "trainable_params": counts.trainable,
        "non_trainable_params": counts.total - counts.trainable,
        "torchinfo_used": torchinfo_available(),
    }
    if counts.partial:
        payload["partial"] = True
    return payload
