"""Persists a debugger capture result on the target's own midrun process.

Pushes through `POST /debug/captures` rather than writing directly into the
target's `ArtifactStore` from this (external) process: `ArtifactStore` mints ids
from a process-local counter (`midrun.runtime.commands.next_id`), so a second
process instantiating its own `ArtifactStore` could mint colliding ids.
"""

import json
from typing import Any

from midrun.debugger.hold_client import HoldClient


def push_capture(
    hold_client: HoldClient,
    kind: str,
    result: dict[str, Any],
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persists a capture `result` as a debug artifact.

    `result` comes from `pyspy_capture.capture` or `gdb_capture.gdb_capture`.

    Returns:
        The artifact manifest (id, kind, created_at, metadata, files, path).
    """
    files = {"result.json": json.dumps(result, indent=2, default=str)}
    return hold_client.push_capture(kind, files, extra_metadata or {})
