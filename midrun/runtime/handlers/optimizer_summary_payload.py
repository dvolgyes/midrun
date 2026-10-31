"""Describes the optimizers of a trainer."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lightning.pytorch import Trainer


def optimizer_summary_payload(trainer: "Trainer") -> dict[str, Any]:
    """Returns the type and the param group learning rates of each optimizer, by index."""
    return {
        str(index): {
            "type": type(optimizer).__name__,
            "param_groups": [
                {"lr": float(group["lr"])} for group in optimizer.param_groups
            ],
        }
        for index, optimizer in enumerate(trainer.optimizers)
    }
