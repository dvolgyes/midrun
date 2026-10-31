"""The argument model, handler, and rank and hold policy of one command kind."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from midrun.runtime.command_args.command_args_model import CommandArgsModel
from midrun.runtime.command_specs.command_runtime import CommandRuntime
from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint


@dataclass(frozen=True, kw_only=True)
class CommandSpec:
    """Everything that the runtime and REST need to know about one command kind.

    Every field is required, so a new kind states its rank and hold policy.

    Attributes:
        args: The model that checks the arguments at submission.
        handler: Runs the command on the training thread at a safe point.
        all_ranks: Every rank runs the command, because it changes state that each
            rank holds a copy of. Otherwise it acts on rank 0's services and runs
            only there.
        during_hold: A gate hold runs the command; otherwise it waits for the
            hold's end.
        during_exception: The exception hold runs the command.
        needs_batch: The command reads the in-flight batch, so it stays queued
            while no batch is in flight, as at an epoch end.
    """

    args: type[CommandArgsModel]
    handler: Callable[
        [CommandRuntime, Any, Any, Command, SafePoint | None], dict[str, Any]
    ]
    all_ranks: bool
    during_hold: bool
    during_exception: bool
    needs_batch: bool
