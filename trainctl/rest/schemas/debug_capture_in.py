"""The body of `POST /debug/captures`."""

from typing import Any

from pydantic import BaseModel


class DebugCaptureIn(BaseModel):
    """Body for `POST /debug/captures`. Persists a capture taken by an external debugger tool."""

    kind: str
    files: dict[str, str] = {}
    metadata: dict[str, Any] = {}
