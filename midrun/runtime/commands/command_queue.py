"""The thread-safe queue of submitted commands."""

import threading
import time
from collections.abc import Callable
from typing import Any

from midrun.runtime.commands.command import Command
from midrun.runtime.commands.command_refusals import CommandRefusals
from midrun.runtime.commands.command_store import CommandStore
from midrun.runtime.commands.command_when import CommandWhen
from midrun.runtime.commands.next_id import next_id
from midrun.runtime.commands.refused_command_error import RefusedCommandError
from midrun.runtime.events import SafePoint


class CommandQueue:
    """Thread-safe registry of submitted commands and their due-ness evaluation.

    `validate_args` checks each submission's kind and args, and returns the args with
    defaults filled in. It raises ValueError for an invalid submission.

    The queue keeps every queued or executing command, and the last
    `CommandStore.finished_limit` finished ones; it forgets older finished commands.

    Attributes:
        refusals: The command kinds that `submit` and `submit_once` refuse.
    """

    _validate_args: Callable[[str, dict[str, Any]], dict[str, Any]]
    _lock: threading.Lock
    _store: CommandStore
    refusals: CommandRefusals

    def __init__(
        self, validate_args: Callable[[str, dict[str, Any]], dict[str, Any]]
    ) -> None:
        self._validate_args = validate_args
        self._lock = threading.Lock()
        self._store = CommandStore()
        self.refusals = CommandRefusals()

    def submit(self, kind: str, args: dict[str, Any], when: CommandWhen) -> Command:
        """Registers a new queued command.

        Raises:
            ValueError: `kind` or `args` is invalid.
            RefusedCommandError: `kind` was refused.
        """
        args = self._validate_args(kind, args)
        command = Command(
            id=next_id("cmd"), kind=kind, args=args, when=when, created_at=time.time()
        )
        with self._lock:
            if (reason := self.refusals.reason(kind)) is not None:
                raise RefusedCommandError(f"{kind} refused: {reason}")
            self._store.add(command)
        return command

    def end_queued(
        self, status: str, *, kind: str | None = None, error: str | None = None
    ) -> list[str]:
        """Ends every still-queued command, or every one of `kind`, with `status`.

        `status` is `cancelled` or `expired`; `error` becomes each command's error.
        To keep a kind out, refuse it in `refusals` first: a submission that passed
        the refusal check before then is already queued, and is ended here.

        Returns:
            The ids of the ended commands.
        """
        with self._lock:
            ended = [
                command
                for command in self._store.queued()
                if kind is None or command.kind == kind
            ]
            for command in ended:
                command.status = status
                command.error = error
                self._store.finish(command)
            return [command.id for command in ended]

    def submit_once(
        self, kind: str, args: dict[str, Any], when: CommandWhen
    ) -> tuple[Command, bool]:
        """Returns an equal command that is still queued or executing, else submits one.

        Returns:
            The command, and whether it was newly submitted.

        Raises:
            ValueError: `kind` or `args` is invalid.
            RefusedCommandError: `kind` was refused.
        """
        args = self._validate_args(kind, args)
        with self._lock:
            if (reason := self.refusals.reason(kind)) is not None:
                raise RefusedCommandError(f"{kind} refused: {reason}")
            for command in self._store.all():
                if command.status in ("queued", "executing") and (
                    command.kind,
                    command.args,
                    command.when,
                ) == (kind, args, when):
                    return command, False
            command = Command(
                id=next_id("cmd"),
                kind=kind,
                args=args,
                when=when,
                created_at=time.time(),
            )
            self._store.add(command)
        return command, True

    def get(self, command_id: str) -> Command | None:
        """Returns a command by id, or None if unknown or forgotten."""
        with self._lock:
            return self._store.get(command_id)

    def list_all(self, *, status: str | None = None) -> list[Command]:
        """Returns the known commands, in submission order.

        A `status` limits the result to the commands with that status.
        """
        with self._lock:
            if status == "queued":
                return self._store.queued()
            return [
                command
                for command in self._store.all()
                if status is None or command.status == status
            ]

    def due(
        self,
        safe_point: SafePoint | None,
        epoch: int,
        global_step: int,
        allowed: frozenset[str] | None,
    ) -> list[Command]:
        """Returns queued commands whose `when` condition is satisfied now.

        `allowed` limits the result to these kinds; None allows every kind.
        """
        with self._lock:
            return [
                command
                for command in self._store.queued()
                if (allowed is None or command.kind in allowed)
                and command.when.is_due(safe_point, epoch, global_step)
            ]

    def cancel(self, command_id: str) -> bool:
        """Cancels a still-queued command.

        Returns:
            True if cancelled, False if unknown or already past `queued`.
        """
        with self._lock:
            command = self._store.get(command_id)
            if command is None or command.status != "queued":
                return False
            command.status = "cancelled"
            self._store.finish(command)
            return True

    def mark_executing(self, command: Command) -> bool:
        """Transitions a command from `queued` to `executing`.

        A command that `due` returned can be cancelled before it runs, so the
        transition happens only if the command is still queued.

        Returns:
            True if the command is now executing, False if it is no longer queued.
        """
        with self._lock:
            if command.status != "queued":
                return False
            command.status = "executing"
            self._store.start(command)
            return True

    def mark_result(
        self,
        command: Command,
        *,
        succeeded: bool,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Records a command's terminal outcome."""
        with self._lock:
            command.status = "succeeded" if succeeded else "failed"
            command.result = result
            command.error = error
            self._store.finish(command)
