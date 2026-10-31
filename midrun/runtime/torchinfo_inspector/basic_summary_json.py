"""The parameter counts of a model."""

from typing import TYPE_CHECKING, Any

from midrun.runtime.torchinfo_inspector.torchinfo_available import (
    torchinfo_available,
)

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule


def basic_summary_json(pl_module: "LightningModule") -> dict[str, Any]:
    """Always-available parameter counts, independent of torchinfo's presence."""
    total = sum(p.numel() for p in pl_module.parameters())
    trainable = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": total - trainable,
        "torchinfo_used": torchinfo_available(),
    }
