"""Parses the execution time of a REST request."""

from typing import Any

from midrun.rest.argument_error import argument_error
from midrun.runtime.commands.parse_when import parse_when


def parse_when_or_422(raw: Any, *, loc: tuple[str, ...] = ("body", "when")) -> Any:
    """Returns `parse_when(raw)`.

    Raises:
        RequestValidationError: `raw`, the argument at `loc`, is not a valid
            execution time (status 422).
    """
    try:
        return parse_when(raw)
    except ValueError as exc:
        raise argument_error(loc, str(exc)) from exc
