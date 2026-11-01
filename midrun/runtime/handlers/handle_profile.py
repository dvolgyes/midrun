"""Runs the `profile` command: arms a bounded torch.profiler run."""

from typing import TYPE_CHECKING, Any

from midrun.runtime.command_specs.command_runtime import CommandRuntime
from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint
from midrun.runtime.profiler import ProfilerRun
from midrun.runtime.require_steady_cpu_clock import require_steady_cpu_clock

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer


def handle_profile(
    runtime: CommandRuntime,
    pl_module: "LightningModule",
    trainer: "Trainer",
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Starts a bounded torch.profiler run with the step counts and the level of `command`.

    The steps count full batches from the next batch on. The rest of the current
    batch, with a hold in which this command runs, is not recorded.

    Raises:
        ValueError: A profiler run is already armed.
        RuntimeError: The CPU clock that torch.profiler reads differs between the
            CPUs of this thread.
    """
    del pl_module, trainer, safe_point
    if runtime.profiler_run is not None:
        raise ValueError("a profiler run is already armed")
    warmup_steps = command.args["warmup_steps"]
    active_steps = command.args["active_steps"]
    level = command.args["level"]
    then_hold = command.args["then_hold"]
    run = ProfilerRun(command.id, warmup_steps, active_steps, level, then_hold)
    require_steady_cpu_clock("profile")
    run.start(runtime.rank_info.device)
    runtime.profiler_run = run
    return {
        "armed": True,
        "warmup_steps": warmup_steps,
        "active_steps": active_steps,
        "level": level,
        "then_hold": then_hold,
    }
