"""The body of `POST /snapshots`."""

from typing import Any

from pydantic import BaseModel


class SnapshotIn(BaseModel):
    """Body for `POST /snapshots`."""

    when: Any = "now"
    kind: str = "runtime"
