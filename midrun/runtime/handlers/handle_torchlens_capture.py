"""Runs the `torchlens_capture` command."""

from typing import TYPE_CHECKING, Any

from midrun.runtime.command_specs.command_runtime import CommandRuntime
from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint
from midrun.runtime.run_on_in_flight_batch import run_on_in_flight_batch
from midrun.runtime.torchlens_available import torchlens_available
from midrun.runtime.torchlens_capture import torchlens_capture

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer


def handle_torchlens_capture(
    runtime: CommandRuntime,
    pl_module: "LightningModule",
    trainer: "Trainer",
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Writes a torchlens trace of the model for the in-flight batch as a debug capture.

    Enters a hold first if the run is not held.

    Raises:
        ValueError: No batch is in flight, or the parameters are sharded.
        ImportError: torchlens is not installed.
    """
    if not torchlens_available():
        raise ImportError("torchlens_capture requires the torchlens package")

    def capture(target: object) -> dict[str, str | bytes]:
        """Enters a hold, then traces `target`.

        The hold is entered inside the inspector guard, so that a refused capture
        engages no hold.
        """
        if not runtime.gate.is_held():
            # torchlens instruments the model with hooks; auto-engage a hold so a
            # human can inspect the result (or a failed capture's aftermath) before
            # choosing to resume, rather than requiring a separate hold command
            # submitted first.
            runtime.gate.enter(
                command.id, str(safe_point), trainer.current_epoch, trainer.global_step
            )
        return torchlens_capture(pl_module, target)

    files = run_on_in_flight_batch(
        capture, runtime, trainer, pl_module, "torchlens_capture"
    )
    return runtime.artifact_store.write_debug_capture("torchlens-trace", files)
