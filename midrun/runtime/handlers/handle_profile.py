"""Runs the `profile` command: arms a bounded torch.profiler run."""

from typing import TYPE_CHECKING, Any

from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint
from midrun.runtime.profiler import ProfilerRun

if TYPE_CHECKING:
    from midrun.runtime.runtime import MidrunRuntime


def handle_profile(
    runtime: "MidrunRuntime",
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Starts a bounded torch.profiler run with the step counts and the level of `command`.

    The steps count full batches from the next batch on. The rest of the current
    batch, with a hold in which this command runs, is not recorded.

    Raises:
        ValueError: A profiler run is already armed.
    """
    del pl_module, trainer, safe_point
    if runtime.profiler_run is not None:
        raise ValueError("a profiler run is already armed")
    warmup_steps = command.args["warmup_steps"]
    active_steps = command.args["active_steps"]
    level = command.args["level"]
    then_hold = command.args["then_hold"]
    run = ProfilerRun(command.id, warmup_steps, active_steps, level, then_hold)
    run.start()
    runtime.profiler_run = run
    return {
        "armed": True,
        "warmup_steps": warmup_steps,
        "active_steps": active_steps,
        "level": level,
        "then_hold": then_hold,
    }
