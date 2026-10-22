"""The argument model of the profile command."""

from typing import Annotated, Literal

from pydantic import Field

from trainctl.runtime.command_args.command_args_model import CommandArgsModel
from trainctl.runtime.command_args.non_negative_index import NonNegativeIndex


class ProfileArgs(CommandArgsModel):
    """Args of `profile`."""

    warmup_steps: NonNegativeIndex = 0
    active_steps: Annotated[int, Field(ge=1)] = 1
    level: Literal["basic", "shapes", "memory", "full"] = "basic"
    then_hold: bool = False
