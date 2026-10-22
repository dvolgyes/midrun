"""Runs the `snapshot` command: writes a state snapshot artifact."""

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from trainctl.runtime.commands.command import Command
from trainctl.runtime.events import SafePoint
from trainctl.runtime.handlers.optimizer_summary_payload import (
    optimizer_summary_payload,
)

if TYPE_CHECKING:
    from trainctl.runtime.runtime import TrainctlRuntime


def handle_snapshot(
    runtime: "TrainctlRuntime",
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict:
    """Writes a snapshot of the runtime state, the model, the optimizers or the schedulers.

    Raises:
        ValueError: The snapshot kind is unknown.
    """
    del safe_point
    kind = command.args["kind"]
    if kind == "runtime":
        payload = asdict(runtime.state.read())
    elif kind == "model_summary":
        payload = {
            "class": type(pl_module).__name__,
            "hparams": dict(getattr(pl_module, "hparams", {})),
        }
    elif kind == "optimizer_summary":
        payload = optimizer_summary_payload(trainer)
    elif kind == "scheduler_summary":
        configs = getattr(trainer, "lr_scheduler_configs", [])
        payload = {
            str(index): {"type": type(getattr(config, "scheduler", config)).__name__}
            for index, config in enumerate(configs)
        }
    else:
        raise ValueError(f"unknown snapshot kind {kind!r}")
    return runtime.artifacts.write_snapshot(kind, payload)
