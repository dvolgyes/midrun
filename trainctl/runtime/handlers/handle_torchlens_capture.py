"""Runs the `torchlens_capture` command."""

from typing import TYPE_CHECKING, Any

from trainctl.runtime import forward_input
from trainctl.runtime.commands.command import Command
from trainctl.runtime.events import SafePoint
from trainctl.runtime.torchlens_available import torchlens_available
from trainctl.runtime.torchlens_capture import torchlens_capture

if TYPE_CHECKING:
    from trainctl.runtime.runtime import TrainctlRuntime


def handle_torchlens_capture(
    runtime: "TrainctlRuntime",
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict:
    """Writes a torchlens trace of the model for the in-flight batch as a debug capture.

    Enters a hold first if the run is not held.

    Raises:
        ValueError: No batch is in flight.
        ImportError: torchlens is not installed.
    """
    if runtime.current_step.batch is None:
        raise ValueError("torchlens_capture requires an in-flight batch")
    if not torchlens_available():
        raise ImportError("torchlens_capture requires the torchlens package")
    if not runtime.gate.is_held():
        # torchlens instruments the model with hooks; auto-engage a hold so a human
        # can inspect the result (or a failed capture's aftermath) before choosing to
        # resume, rather than requiring a separate hold command submitted first.
        runtime.gate.enter(
            command.id, str(safe_point), trainer.current_epoch, trainer.global_step
        )
    target = forward_input.resolve(pl_module, runtime.current_step.batch)
    files = torchlens_capture(pl_module, target)
    return runtime.artifacts.write_debug_capture("torchlens-trace", files)
