"""Runs the `release_exception` command."""

from typing import TYPE_CHECKING, Any

from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint

if TYPE_CHECKING:
    from midrun.runtime.runtime import MidrunRuntime


def handle_release_exception(
    runtime: "MidrunRuntime",
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
    runtime._exception_hold = None
    return {"released": command.id}
