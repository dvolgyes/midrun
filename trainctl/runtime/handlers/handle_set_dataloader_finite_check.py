"""Runs the `set_dataloader_finite_check` command."""

from typing import TYPE_CHECKING, Any

from trainctl.runtime.commands.command import Command
from trainctl.runtime.events import SafePoint

if TYPE_CHECKING:
    from trainctl.runtime.runtime import TrainctlRuntime


def handle_set_dataloader_finite_check(
    runtime: "TrainctlRuntime",
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Turns the dataloader finite guard on or off."""
    del pl_module, trainer, safe_point
    enabled = command.args["enabled"]
    runtime.finite_guard.enabled = enabled
    return {"enabled": enabled}
