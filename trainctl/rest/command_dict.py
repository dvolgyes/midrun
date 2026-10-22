"""The JSON form of a command in REST responses."""

from typing import Any

from trainctl.runtime.commands.command import Command


def command_dict(command: Command) -> dict[str, Any]:
    """Returns the fields of `command` that REST responses show."""
    return {
        "id": command.id,
        "kind": command.kind,
        "args": command.args,
        "status": command.status,
        "result": command.result,
        "error": command.error,
        "created_at": command.created_at,
    }
