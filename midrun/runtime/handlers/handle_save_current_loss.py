"""Runs the `save_current_loss` command."""

from typing import TYPE_CHECKING, Any

from midrun.runtime.command_specs.command_runtime import CommandRuntime
from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer


def handle_save_current_loss(
    runtime: CommandRuntime,
    pl_module: "LightningModule",
    trainer: "Trainer",
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Writes the in-flight loss with `torch.save` as a debug capture.

    The loss is saved detached, so the file holds its value without the autograd graph.

    Raises:
        ValueError: No loss is in flight.
    """
    del pl_module, trainer, command, safe_point
    if runtime.current_step.loss is None:
        raise ValueError("save_current_loss requires an in-flight loss")
    import io

    import torch

    buffer = io.BytesIO()
    torch.save(runtime.current_step.loss.detach(), buffer)
    return runtime.artifact_store.write_debug_capture(
        "current-loss", {"loss.pt": buffer.getvalue()}
    )
