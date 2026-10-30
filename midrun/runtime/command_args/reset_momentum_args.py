"""The argument model of the reset_momentum command, with its selectors."""

from typing import Literal

from midrun.runtime.command_args.command_args_model import CommandArgsModel
from midrun.runtime.command_args.non_negative_index import NonNegativeIndex


class _ParameterSelector(CommandArgsModel):
    """A `reset_momentum` selector for one parameter by name."""

    parameter: str


class _ParameterPrefixSelector(CommandArgsModel):
    """A `reset_momentum` selector for the parameters whose name has a prefix."""

    parameter_prefix: str


class _ParamGroupSelector(CommandArgsModel):
    """A `reset_momentum` selector for one param group."""

    param_group: NonNegativeIndex


class ResetMomentumArgs(CommandArgsModel):
    """Args of `reset_momentum`."""

    optimizer: NonNegativeIndex = 0
    selector: (
        Literal["all"]
        | _ParameterSelector
        | _ParameterPrefixSelector
        | _ParamGroupSelector
    ) = "all"
