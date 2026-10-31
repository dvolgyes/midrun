"""Waits until a queued command starts or ends."""

import asyncio
import time
from typing import TYPE_CHECKING

from midrun.runtime.commands.command import Command

if TYPE_CHECKING:
    from midrun.runtime.runtime import MidrunRuntime


async def wait_for_command(
    runtime: "MidrunRuntime",
    command_id: str,
    timeout_s: float,
    poll_interval_s: float = 0.05,
) -> Command:
    """Polls until `command_id` leaves `queued`, or `timeout_s` elapses.

    A hold command moves straight from `queued` to `succeeded`/`failed` synchronously
    on the training thread the instant `RunGate.enter` returns (see
    `execute_command`), so this doubles as "wait for the hold to actually
    engage." A command still `queued` when this returns hasn't reached its safe point
    yet -- e.g. a `train_epoch_end` hold armed mid-epoch -- and stays pollable via
    `GET /commands/{id}` or `GET /holds`.

    Raises:
        RuntimeError: The queue forgot `command_id` while this waited.
    """
    deadline = time.monotonic() + timeout_s
    command = runtime.command_queue.get(command_id)
    while (
        command is not None
        and command.status == "queued"
        and time.monotonic() < deadline
    ):
        await asyncio.sleep(poll_interval_s)
        command = runtime.command_queue.get(command_id)
    if command is None:
        raise RuntimeError(
            f"command {command_id!r} vanished from the queue while waiting"
        )
    return command
