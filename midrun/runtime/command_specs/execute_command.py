"""Runs one due command and records its result in the command queue."""

from typing import Any

from loguru import logger

from midrun.runtime.command_specs.command_runtime import CommandRuntime
from midrun.runtime.command_specs.command_specs import COMMAND_SPECS
from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint


def execute_command(
    runtime: CommandRuntime,
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> None:
    """Runs `command`'s handler and marks the command succeeded or failed.

    A handler's `Exception` fails the command and does not stop training. Any other
    `BaseException` fails the command and propagates.
    """
    # Skips a hold cancelled after `_due_commands` returned it. Only holds can be
    # cancelled, and only rank 0 runs them, so no rank skips what the others run.
    if not runtime.command_queue.mark_executing(command):
        return
    # The queue validates the kind at submit, so every queued kind has a handler.
    handler = COMMAND_SPECS[command.kind].handler
    try:
        result = handler(runtime, pl_module, trainer, command, safe_point)
    except Exception as exc:  # noqa: BLE001 -- command boundary: untrusted REST input must not crash training
        logger.warning(
            "midrun command {} ({}) failed: {}", command.id, command.kind, exc
        )
        runtime.command_queue.mark_result(command, succeeded=False, error=str(exc))
        return
    except BaseException as exc:
        # An interrupt or exit ends training, but not as a command still executing.
        error = f"interrupted by {type(exc).__name__}"
        runtime.command_queue.mark_result(command, succeeded=False, error=error)
        raise
    runtime.command_queue.mark_result(command, succeeded=True, result=result)
