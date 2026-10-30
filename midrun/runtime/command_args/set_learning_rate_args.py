"""The argument model of the set_learning_rate command."""

from typing import Annotated, Literal

from pydantic import Field

from midrun.runtime.command_args.command_args_model import CommandArgsModel
from midrun.runtime.command_args.non_negative_index import NonNegativeIndex


class SetLearningRateArgs(CommandArgsModel):
    """Args of `set_learning_rate`."""

    value: Annotated[float, Field(ge=0)]
    optimizer: NonNegativeIndex = 0
    param_group: Literal["*"] | NonNegativeIndex = "*"
