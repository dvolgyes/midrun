"""Runs the `set_lightning_profiler` command."""

from typing import TYPE_CHECKING, Any

from trainctl.runtime.commands.command import Command
from trainctl.runtime.events import SafePoint

if TYPE_CHECKING:
    from trainctl.runtime.runtime import TrainctlRuntime


def handle_set_lightning_profiler(
    runtime: "TrainctlRuntime",
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Installs (or removes) one of Lightning's own profilers as `trainer.profiler`.

    Unlike `profile` (a bounded `torch.profiler` capture over N batches, finalized
    into a debug-capture artifact), Lightning's profilers wrap Lightning's own named
    hooks (`training_step`, `backward`, `optimizer_step`, dataloader-next, callbacks,
    ...) and accumulate continuously until swapped out or read. `trainer.profiler` is
    a plain attribute Lightning re-reads on every hook call, so swapping it here takes
    effect immediately, without restarting the Trainer.

    `command.args["level"]` selects `off`, `simple`, `advanced`, or `pytorch`.
    Selecting a level other than the current one -- or reselecting the current one --
    always installs a fresh profiler instance, discarding any previously accumulated
    durations (an implicit reset).
    """
    del pl_module, safe_point
    level = command.args["level"]
    profilers = runtime.backend.PL.profilers
    if level == "off":
        trainer.profiler = profilers.PassThroughProfiler()
        runtime.lightning_profiler = None
    else:
        profiler_cls = {
            "simple": profilers.SimpleProfiler,
            "advanced": profilers.AdvancedProfiler,
            "pytorch": profilers.PyTorchProfiler,
        }[level]
        profiler = profiler_cls()
        trainer.profiler = profiler
        runtime.lightning_profiler = profiler
    runtime.lightning_profiler_level = level
    return {"level": level}
