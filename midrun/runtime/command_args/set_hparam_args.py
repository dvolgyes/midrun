"""The argument model of the set_hparam command."""

from typing import Any

from midrun.runtime.command_args.command_args_model import CommandArgsModel


class SetHparamArgs(CommandArgsModel):
    """Args of `set_hparam`."""

    # The knob's current value decides the valid type of `value`; see
    # `hparam_registry.check_and_assign`.
    name: str
    value: Any
