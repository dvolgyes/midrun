"""Runs the `reset_momentum` command."""

from typing import TYPE_CHECKING, Any

from midrun.runtime.command_specs.command_runtime import CommandRuntime
from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint
from midrun.runtime.handlers.optimizer_at import optimizer_at
from midrun.runtime.optimizer_surgery.find_surgery_adapter import find_surgery_adapter
from midrun.runtime.optimizer_surgery.resolve_optimizer_parameters import (
    resolve_optimizer_parameters,
)

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer


def handle_reset_momentum(
    runtime: CommandRuntime,
    pl_module: "LightningModule",
    trainer: "Trainer",
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Resets the momentum state of the selected parameters of one optimizer.

    Raises:
        ValueError: Optimizer surgery is disabled, the optimizer does not exist, or no
            surgery adapter supports the optimizer.
    """
    del safe_point
    if not runtime.config.optimizer_surgery_enabled:
        raise ValueError("reset_momentum requires optimizer_surgery_enabled")
    optimizer_index = command.args["optimizer"]
    optimizer = optimizer_at(trainer.optimizers, optimizer_index)
    adapter = find_surgery_adapter(optimizer)
    if adapter is None:
        raise ValueError(
            "optimizer_surgery_unsupported: no reset_momentum adapter for "
            f"{type(optimizer).__module__}.{type(optimizer).__name__}"
        )
    parameters = resolve_optimizer_parameters(
        pl_module, optimizer, command.args["selector"]
    )
    result = adapter.reset_momentum(optimizer, parameters)
    return {"optimizer": optimizer_index, **result}
