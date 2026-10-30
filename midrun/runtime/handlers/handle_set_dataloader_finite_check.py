"""Runs the `set_dataloader_finite_check` command."""

from typing import Any

from midrun.runtime.command_specs.command_runtime import CommandRuntime
from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint


def handle_set_dataloader_finite_check(
    runtime: CommandRuntime,
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
