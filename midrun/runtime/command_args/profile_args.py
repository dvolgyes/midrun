"""The argument model of the profile command."""

from typing import Annotated, Literal

from pydantic import Field

from midrun.runtime.command_args.command_args_model import CommandArgsModel
from midrun.runtime.command_args.non_negative_index import NonNegativeIndex


class ProfileArgs(CommandArgsModel):
    """Args of `profile`.

    The default warmup is one batch, as torch.profiler warns that a run without
    warmup can skew its results.
    """

    warmup_steps: NonNegativeIndex = 1
    active_steps: Annotated[int, Field(ge=1)] = 1
    level: Literal["basic", "shapes", "memory", "full"] = "basic"
    then_hold: bool = False
