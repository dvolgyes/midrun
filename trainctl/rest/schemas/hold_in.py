"""The body of `POST /holds`."""

from typing import Any

from pydantic import BaseModel, Field


class HoldIn(BaseModel):
    """Body for `POST /holds`.

    Attributes:
        timeout_s: How long the request waits for the hold to actually engage
            before falling back to a `202` pollable response, at most 60 seconds.
            `0` skips the wait and returns the async response immediately.
    """

    when: Any = "now"
    reason: str = ""
    timeout_s: float = Field(default=5.0, ge=0, le=60)
