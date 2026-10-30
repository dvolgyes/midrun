"""Runs the `save_current_loss` command."""

from typing import TYPE_CHECKING, Any

from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint

if TYPE_CHECKING:
    from midrun.runtime.runtime import MidrunRuntime


def handle_save_current_loss(
    runtime: "MidrunRuntime",
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Writes the in-flight loss with `torch.save` as a debug capture."""
    del pl_module, trainer, command, safe_point
    import io

    import torch

    buffer = io.BytesIO()
    torch.save(runtime.current_step.loss, buffer)
    return runtime.artifact_store.write_debug_capture(
        "current-loss", {"loss.pt": buffer.getvalue()}
    )
