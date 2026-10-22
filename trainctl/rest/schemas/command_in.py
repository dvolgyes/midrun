"""The body of `POST /commands`."""

from typing import Any

from pydantic import BaseModel


class ExecutionSpec(BaseModel):
    """When a `/commands` submission should become due. See `trainctl.runtime.commands.parse_when`."""

    when: Any = "now"


class CommandIn(BaseModel):
    """Body for `POST /commands`."""

    kind: str
    args: dict[str, Any] = {}
    execution: ExecutionSpec = ExecutionSpec()
