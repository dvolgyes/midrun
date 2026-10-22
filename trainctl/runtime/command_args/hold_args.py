"""The argument model of the hold command."""

from trainctl.runtime.command_args.command_args_model import CommandArgsModel


class HoldArgs(CommandArgsModel):
    """Args of `hold`."""

    reason: str = ""
