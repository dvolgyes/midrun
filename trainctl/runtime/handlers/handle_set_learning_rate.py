"""Runs the `set_learning_rate` command."""

from typing import TYPE_CHECKING, Any

from trainctl.runtime.commands.command import Command
from trainctl.runtime.events import SafePoint
from trainctl.runtime.handlers.optimizer_at import optimizer_at

if TYPE_CHECKING:
    from trainctl.runtime.runtime import TrainctlRuntime


def handle_set_learning_rate(
    runtime: "TrainctlRuntime",
    pl_module: Any,
    trainer: Any,
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
        index = param_group
        if index >= len(optimizer.param_groups):
            raise ValueError(f"optimizer {optimizer_index} has no param_group {index}")
        groups = [(index, optimizer.param_groups[index])]
    changes: dict[str, Any] = {}
    for index, group in groups:
        changes[str(index)] = {"before": group["lr"], "after": value}
        group["lr"] = value
    result: dict[str, Any] = {"optimizer": optimizer_index, "groups": changes}
    if getattr(trainer, "lr_scheduler_configs", None):
        result["warning"] = (
            "optimizer has an active LR scheduler; this value may be overwritten "
            "on its next scheduler step"
        )
    return result
