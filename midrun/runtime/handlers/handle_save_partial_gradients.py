"""Runs the `save_partial_gradients` command."""

import json
import math
from typing import Any

import torch

from midrun.runtime.command_specs.command_runtime import CommandRuntime
from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint


def handle_save_partial_gradients(
    runtime: CommandRuntime,
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Writes whether each parameter has a gradient, and its norm and shape, as a debug capture.

    The norm is computed in at least float32, so that the norm of a float16 or
    bfloat16 gradient does not overflow. A NaN or infinite norm is written as
    `{"kind": "float", "value": "nan"}` (or `"inf"`), as in the hooks' `params.json`.
    """
    del trainer, command, safe_point
    summary: dict[str, dict[str, Any]] = {}
    for name, parameter in pl_module.named_parameters():
        grad = parameter.grad
        norm: float | dict[str, str] | None = None
        if grad is not None:
            norm = float(
                torch.linalg.vector_norm(
                    grad, dtype=torch.promote_types(grad.dtype, torch.float32)
                )
            )
            if not math.isfinite(norm):
                # Standard JSON has no NaN/Infinity literal.
                norm = {"kind": "float", "value": repr(norm)}
        summary[name] = {
            "grad_is_none": grad is None,
            "grad_norm": norm,
            "grad_shape": list(grad.shape) if grad is not None else None,
        }
    return runtime.artifact_store.write_debug_capture(
        "partial-gradients",
        {"gradients.json": json.dumps(summary, indent=2, allow_nan=False)},
    )
