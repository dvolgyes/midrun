"""Runs the `set_learning_rate` command."""

from typing import TYPE_CHECKING, Any

import torch

from midrun.runtime.command_specs.command_runtime import CommandRuntime
from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint
from midrun.runtime.handlers.optimizer_at import optimizer_at

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer


def _non_target_lr_storages(
    trainer: "Trainer", optimizer_index: int, param_group: int | str
) -> set[torch.UntypedStorage]:
    """Collect tensor storage that this command must not modify."""
    return {
        lr.untyped_storage()
        for other_index, optimizer in enumerate(trainer.optimizers)
        for index, group in enumerate(optimizer.param_groups)
        if (other_index != optimizer_index or param_group not in ("*", index))
        and isinstance(lr := group["lr"], torch.Tensor)
    }


def handle_set_learning_rate(
    runtime: CommandRuntime,
    pl_module: "LightningModule",
    trainer: "Trainer",
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Sets the learning rate of all param groups, or of one, of one optimizer.

    One param group needs `optimizer_surgery_enabled`. The result has a warning when an
    LR scheduler can overwrite the value.

    Raises:
        ValueError: The optimizer or the param group does not exist, or one param group
            is selected without optimizer surgery.
    """
    del pl_module, safe_point
    value = command.args["value"]
    optimizer_index = command.args["optimizer"]
    optimizer = optimizer_at(trainer.optimizers, optimizer_index)
    param_group = command.args["param_group"]
    if param_group != "*" and not runtime.config.optimizer_surgery_enabled:
        raise ValueError(
            "per-group learning-rate selection requires optimizer_surgery_enabled"
        )
    if param_group == "*":
        groups = list(enumerate(optimizer.param_groups))
    else:
        if param_group >= len(optimizer.param_groups):
            raise ValueError(
                f"optimizer {optimizer_index} has no param_group {param_group}"
            )
        groups = [(param_group, optimizer.param_groups[param_group])]
    changes = {
        str(index): {"before": float(group["lr"]), "after": value}
        for index, group in groups
    }
    other_storages = (
        _non_target_lr_storages(trainer, optimizer_index, param_group)
        if any(isinstance(group["lr"], torch.Tensor) for _, group in groups)
        else set()
    )
    for _, group in groups:
        lr = group["lr"]
        if isinstance(lr, torch.Tensor):
            if other_storages and lr.untyped_storage() in other_storages:
                lr = lr.clone()
                group["lr"] = lr
            lr.fill_(value)
        else:
            group["lr"] = value
    result: dict[str, Any] = {"optimizer": optimizer_index, "groups": changes}
    if getattr(trainer, "lr_scheduler_configs", None):
        result["warning"] = (
            "optimizer has an active LR scheduler; this value may be overwritten "
            "on its next scheduler step"
        )
    return result
