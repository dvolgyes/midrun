"""Runs one due command and records its result in the command queue."""

from contextlib import nullcontext
from typing import TYPE_CHECKING

from loguru import logger

from midrun.runtime.command_specs.command_runtime import CommandRuntime
from midrun.runtime.command_specs.command_specs import COMMAND_SPECS
from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint
from midrun.runtime.rank_agreement import RankAgreement

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer


def execute_command(
    runtime: CommandRuntime,
    pl_module: "LightningModule",
    trainer: "Trainer",
    command: Command,
    safe_point: SafePoint | None,
    *,
    collective: bool = False,
) -> None:
    """Runs `command`'s handler and marks the command succeeded or failed.

    A handler's `Exception` fails the command and does not stop training. Any other
    `BaseException` fails the command and propagates.
    Collective invocations wait for every handler before publishing its local result.
    """
    # The queue validates the kind at submit, so every queued kind has a handler.
    handler = COMMAND_SPECS[command.kind].handler
    try:
        with (
            RankAgreement(runtime.rank_sync, command.kind)
            if collective
            else nullcontext(),
            runtime.gate.hold_transition() if command.kind == "hold" else nullcontext(),
        ):
            if not runtime.command_queue.mark_executing(command):
                return
            result = handler(runtime, pl_module, trainer, command, safe_point)
    except Exception as exc:  # noqa: BLE001 -- command boundary: untrusted REST input must not crash training
        error = str(exc)
        logger.warning(
            "midrun command {} ({}) failed: {}", command.id, command.kind, error
        )
        runtime.command_queue.mark_result(command, succeeded=False, error=error)
        return
    except BaseException as exc:
        # An interrupt or exit ends training, but not as a command still executing.
        error = (
            RankAgreement.agreed_error(exc) or f"interrupted by {type(exc).__name__}"
        )
        runtime.command_queue.mark_result(command, succeeded=False, error=error)
        raise
    runtime.command_queue.mark_result(command, succeeded=True, result=result)
