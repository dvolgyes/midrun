"""Parses the execution time of a REST request."""

from typing import Any

from fastapi import HTTPException

from trainctl.runtime.commands.parse_when import parse_when


def parse_when_or_400(raw: Any) -> Any:
    """Returns `parse_when(raw)`.

    Raises:
        HTTPException: `raw` is not a valid execution time (status 400).
    """
    try:
        return parse_when(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
