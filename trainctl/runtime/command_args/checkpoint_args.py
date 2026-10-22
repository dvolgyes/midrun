"""The argument model of the checkpoint command."""

from typing import Annotated

from pydantic import Field

from trainctl.runtime.artifacts import ARTIFACT_NAME_PATTERN
from trainctl.runtime.command_args.command_args_model import CommandArgsModel


class CheckpointArgs(CommandArgsModel):
    """Args of `checkpoint`."""

    label: Annotated[str, Field(pattern=ARTIFACT_NAME_PATTERN)] = "manual"
