"""Runs the `set_hparam` command."""

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from midrun.runtime.command_specs.command_runtime import CommandRuntime
from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint
from midrun.runtime.hparam_registry.check_and_assign import check_and_assign
from midrun.runtime.hparam_registry.resolve_hparam_path import resolve_hparam_path

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer


def handle_set_hparam(
    runtime: CommandRuntime,
    pl_module: "LightningModule",
    trainer: "Trainer",
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Assigns a new value to one tunable hparam.

    Raises:
        ValueError: The hparam is unknown or not tunable, or the value is not valid.
    """
    del safe_point
    name = command.args["name"]
    spec = runtime.tunable_hparam_specs.get(name)
    if spec is None:
        raise ValueError(f"unknown or non-tunable hparam: {name!r}")
    parent, leaf = resolve_hparam_path(pl_module, trainer, spec.path)
    result = check_and_assign(spec, parent, leaf, command.args["value"])
    return {"name": name, **asdict(result)}
