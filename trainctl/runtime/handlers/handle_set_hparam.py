"""Runs the `set_hparam` command."""

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from trainctl.runtime.commands.command import Command
from trainctl.runtime.events import SafePoint
from trainctl.runtime.hparam_registry.check_and_assign import check_and_assign
from trainctl.runtime.hparam_registry.resolve_hparam_path import resolve_hparam_path

if TYPE_CHECKING:
    from trainctl.runtime.runtime import TrainctlRuntime


def handle_set_hparam(
    runtime: "TrainctlRuntime",
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Assigns a new value to one tunable hparam.

    Raises:
        ValueError: The hparam is unknown or not tunable, or the value is not valid.
    """
    del safe_point
    name = command.args["name"]
    spec = runtime._tunable_hparam_specs.get(name)
    if spec is None:
        raise ValueError(f"unknown or non-tunable hparam: {name!r}")
    parent, leaf = resolve_hparam_path(pl_module, trainer, spec.path)
    result = check_and_assign(spec, parent, leaf, command.args["value"])
    return {"name": name, **asdict(result)}
