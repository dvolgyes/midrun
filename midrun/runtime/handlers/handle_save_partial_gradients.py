"""Runs the `save_partial_gradients` command."""

import json
from typing import TYPE_CHECKING, Any

from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint

if TYPE_CHECKING:
    from midrun.runtime.runtime import MidrunRuntime


def handle_save_partial_gradients(
    runtime: "MidrunRuntime",
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Writes whether each parameter has a gradient, and its norm and shape, as a debug capture."""
    del trainer, command, safe_point
    summary = {
        name: {
            "grad_is_none": parameter.grad is None,
            "grad_norm": (
                float(parameter.grad.norm()) if parameter.grad is not None else None
            ),
            "grad_shape": (
                list(parameter.grad.shape) if parameter.grad is not None else None
            ),
        }
        for name, parameter in pl_module.named_parameters()
    }
    return runtime.artifact_store.write_debug_capture(
        "partial-gradients", {"gradients.json": json.dumps(summary, indent=2)}
    )
