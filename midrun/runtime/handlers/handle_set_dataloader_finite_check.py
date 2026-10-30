"""Runs the `set_dataloader_finite_check` command."""

from typing import TYPE_CHECKING, Any

from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint

if TYPE_CHECKING:
    from midrun.runtime.runtime import MidrunRuntime


def handle_set_dataloader_finite_check(
    runtime: "MidrunRuntime",
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
