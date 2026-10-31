"""Behavioral tests for `wait_for_command`."""

import asyncio
from types import SimpleNamespace

import pytest

from midrun.rest.wait_for_command import wait_for_command
from midrun.runtime.commands.command import Command
from midrun.runtime.commands.parse_when import parse_when


class _ForgettingQueue:
    """Returns the queued command once, then forgets it."""

    def __init__(self, command: Command) -> None:
        self._command: Command | None = command

    def get(self, command_id: str) -> Command | None:
        command, self._command = self._command, None
        return command


def test_a_command_that_leaves_the_queue_while_waiting_raises() -> None:
    command = Command(
        id="c1", kind="hold", args={}, when=parse_when(None), created_at=0.0
    )
    runtime = SimpleNamespace(command_queue=_ForgettingQueue(command))
    with pytest.raises(RuntimeError, match="vanished from the queue"):
        asyncio.run(wait_for_command(runtime, "c1", timeout_s=1.0, poll_interval_s=0.0))
