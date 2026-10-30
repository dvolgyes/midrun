"""Runs an inspector on the forward input of the in-flight batch."""

from collections.abc import Callable
from typing import Any

import torch

from midrun.runtime import forward_input
from midrun.runtime.command_specs.command_runtime import CommandRuntime
from midrun.runtime.inspector_forward import inspector_forward


def run_on_in_flight_batch[T](
    inspect: Callable[[Any], T],
    runtime: CommandRuntime,
    trainer: object,
    pl_module: torch.nn.Module,
    what: str,
    *,
    needs_grad: bool = False,
) -> T:
    """Returns `inspect` of the forward input of the current step's batch.

    `inspect` runs inside `inspector_forward`, on the input that
    `forward_input.resolve` derives from the batch. Call it on the training thread.

    Args:
        inspect: Gets the forward input and returns the inspector's result.
        runtime: The runtime whose current step holds the batch.
        trainer: The trainer of `pl_module`.
        pl_module: The model that `inspect` runs a forward pass on.
        what: The name of the inspector, for the error messages.
        needs_grad: Whether `inspect` needs autograd to record the graph.

    Raises:
        ValueError: No batch is in flight, or the parameters are sharded.
    """
    batch = runtime.current_step.batch
    if batch is None:
        raise ValueError(f"{what} requires an in-flight batch")
    with inspector_forward(
        runtime.backend, trainer, pl_module, what, needs_grad=needs_grad
    ):
        return inspect(forward_input.resolve(pl_module, batch))
