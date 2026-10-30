"""Runs the `release_exception` command."""

from typing import Any

from midrun.runtime.command_specs.command_runtime import CommandRuntime
from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint


def handle_release_exception(
    runtime: CommandRuntime,
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Ends the active exception hold.

    Raises:
        ValueError: No exception hold is active.
    """
    del pl_module, trainer, safe_point
    if runtime.state.read().exception is None:
        raise ValueError("release_exception: no exception hold is active")
    runtime.exception_hold = None
    return {"released": command.id}
