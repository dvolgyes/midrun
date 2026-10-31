"""Runs the `save_current_batch` command."""

from typing import TYPE_CHECKING, Any

from lightning_utilities.core.apply_func import apply_to_collection

from midrun.runtime.command_specs.command_runtime import CommandRuntime
from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer


def handle_save_current_batch(
    runtime: CommandRuntime,
    pl_module: "LightningModule",
    trainer: "Trainer",
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Writes the in-flight batch with `torch.save` as a debug capture.

    Each tensor in the batch's collections is saved as a detached copy on its own
    device. `torch.save` writes a tensor's whole storage, so saving a view of a
    larger tensor directly would write the whole base tensor.

    Raises:
        ValueError: No batch is in flight.
    """
    del pl_module, trainer, command, safe_point
    if runtime.current_step.batch is None:
        raise ValueError("save_current_batch requires an in-flight batch")
    import io

    import torch

    batch = apply_to_collection(
        runtime.current_step.batch, torch.Tensor, lambda t: t.detach().clone()
    )
    buffer = io.BytesIO()
    torch.save(batch, buffer)
    return runtime.artifact_store.write_debug_capture(
        "current-batch", {"batch.pt": buffer.getvalue()}
    )
