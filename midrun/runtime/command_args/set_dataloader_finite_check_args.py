"""The argument model of the set_dataloader_finite_check command."""

from midrun.runtime.command_args.command_args_model import CommandArgsModel


class SetDataloaderFiniteCheckArgs(CommandArgsModel):
    """Args of `set_dataloader_finite_check`."""

    enabled: bool
