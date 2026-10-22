"""The argument model of the set_lightning_profiler command."""

from typing import Literal

from trainctl.runtime.command_args.command_args_model import CommandArgsModel


class SetLightningProfilerArgs(CommandArgsModel):
    """Args of `set_lightning_profiler`."""

    level: Literal["off", "simple", "advanced", "pytorch"] = "off"
