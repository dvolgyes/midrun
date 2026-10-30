"""The error for a command kind that the queue no longer accepts."""


class RefusedCommandError(RuntimeError):
    """A command was submitted after its kind was refused, as holds are after a stop."""
