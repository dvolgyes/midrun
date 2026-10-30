"""Runs the `set_lightning_profiler` command."""

import weakref
from typing import TYPE_CHECKING, Any

import torch

from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint

if TYPE_CHECKING:
    from midrun.runtime.runtime import MidrunRuntime


def handle_set_lightning_profiler(
    runtime: "MidrunRuntime",
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

    A replaced `PyTorchProfiler` is torn down, so its `torch.profiler` stops. The
    other profilers are not: the hook that runs this command is inside one of their
    actions, and its exit still stops that action on the old profiler. The new
    profiler is set up as the Trainer sets up its own. Its log dir is the old
    profiler's `dirpath`, which the Trainer set from `trainer.log_dir`: this command
    runs on rank 0 only, and `trainer.log_dir` is a collective.
    """
    del safe_point
    level = command.args["level"]
    profilers = runtime.backend.PL.profilers
    old = trainer.profiler
    stage = trainer.state.fn
    if isinstance(old, profilers.PyTorchProfiler):
        torch_profiler = old.profiler
        if (
            isinstance(torch_profiler, torch.profiler.profile)
            and torch_profiler.profiler is None
        ):
            # It never recorded, and Lightning's teardown fails to read its events.
            torch_profiler.__exit__(None, None, None)
            old.profiler = None
        old.teardown(stage=stage)
    if level == "off":
        profiler = profilers.PassThroughProfiler()
        runtime.lightning_profiler = None
    else:
        profiler_cls = {
            "simple": profilers.SimpleProfiler,
            "advanced": profilers.AdvancedProfiler,
            "pytorch": profilers.PyTorchProfiler,
        }[level]
        profiler = profiler_cls()
        runtime.lightning_profiler = profiler
    # The Trainer sets this attribute too; `PyTorchProfiler` reads it.
    # pylint: disable-next=protected-access
    profiler._lightning_module = weakref.proxy(pl_module)
    profiler.setup(
        stage=stage,
        local_rank=trainer.local_rank if trainer.world_size > 1 else None,
        log_dir=old.dirpath,
    )
    trainer.profiler = profiler
    runtime.lightning_profiler_level = level
    return {"level": level}
