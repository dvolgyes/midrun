"""Runs the `torchviz_capture` command."""

from typing import TYPE_CHECKING, Any

from midrun.runtime.command_specs.command_runtime import CommandRuntime
from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint
from midrun.runtime.graph_inspectors.torchviz_capture import torchviz_capture
from midrun.runtime.graph_inspectors.torchviz_capture_live import (
    torchviz_capture_live,
)
from midrun.runtime.run_on_in_flight_batch import run_on_in_flight_batch

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer


def handle_torchviz_capture(
    runtime: CommandRuntime,
    pl_module: "LightningModule",
    trainer: "Trainer",
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Writes a torchviz autograd graph as a debug capture.

    In an exception hold, it draws the graph of the failed loss. Otherwise it draws the
    graph of a new forward pass on the in-flight batch.

    Raises:
        ValueError: The loss or the batch that the capture needs is missing, or
            the parameters are sharded.
    """
    del safe_point
    # Both options read saved tensors, which the failed backward of an exception hold
    # has already freed, so they default to on only in a live capture.
    live = runtime.exception_hold is None
    show_attrs = command.args["show_attrs"]
    show_attrs = live if show_attrs is None else show_attrs
    show_saved = command.args["show_saved"]
    show_saved = live if show_saved is None else show_saved
    if not live:
        # Diagnosing an actual backward failure: visualize the loss exactly as
        # autograd left it when backward() raised, not an unrelated fresh graph.
        if runtime.current_step.loss is None:
            raise ValueError("torchviz_capture requires an in-flight loss")
        files = torchviz_capture(
            pl_module, runtime.current_step.loss, show_attrs, show_saved
        )
    else:
        # A normal live capture: current_step.loss was already backwarded by
        # Lightning itself by the time any command can run, so its graph is gone
        # ("Trying to backward through the graph a second time") -- run a fresh,
        # not-yet-backwarded forward pass instead.
        files = run_on_in_flight_batch(
            lambda target: torchviz_capture_live(
                pl_module, target, show_attrs, show_saved
            ),
            runtime,
            trainer,
            pl_module,
            "torchviz_capture",
            needs_grad=True,
        )
    return runtime.artifact_store.write_debug_capture("torchviz-graph", files)
