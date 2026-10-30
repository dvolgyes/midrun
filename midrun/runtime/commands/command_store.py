"""The live commands and a bounded history of the finished ones."""

from collections import deque

from midrun.runtime.commands.command import Command


class CommandStore:
    """Holds the commands that `CommandQueue` knows, in submission order.

    The queued commands also have their own index, so a scan for due commands reads
    only them. A finished command stays readable until `finished_limit` newer
    commands have finished; then the store forgets it. Not thread-safe:
    `CommandQueue` holds its lock around every call.

    Attributes:
        finished_limit: How many finished commands the store keeps.
    """

    finished_limit: int
    _commands: dict[str, Command]
    _queued: dict[str, Command]
    _finished: deque[str]

    def __init__(self, *, finished_limit: int = 1000) -> None:
        self.finished_limit = finished_limit
        self._commands = {}
        self._queued = {}
        self._finished = deque()

    def add(self, command: Command) -> None:
        """Stores a new queued command."""
        self._commands[command.id] = command
        self._queued[command.id] = command

    def get(self, command_id: str) -> Command | None:
        """Returns a stored command by id, or None."""
        return self._commands.get(command_id)

    def all(self) -> list[Command]:
        """Returns every stored command, in submission order."""
        return list(self._commands.values())

    def queued(self) -> list[Command]:
        """Returns the commands that are still queued, in submission order."""
        return list(self._queued.values())

    def start(self, command: Command) -> None:
        """Removes a command that started executing from the queued index."""
        del self._queued[command.id]

    def finish(self, command: Command) -> None:
        """Records that a command finished, and forgets the oldest finished ones."""
        self._queued.pop(command.id, None)
        self._finished.append(command.id)
        while len(self._finished) > self.finished_limit:
            del self._commands[self._finished.popleft()]
