"""Checks the arguments of a submitted command."""

from typing import Any

from pydantic import ValidationError

from midrun.runtime.command_specs.command_specs import COMMAND_SPECS


class InvalidCommandError(ValueError):
    """A submitted command has an unknown kind or invalid args."""


def validate_command_args(kind: str, args: dict[str, Any]) -> dict[str, Any]:
    """Checks `args` for a `kind` command and returns them with defaults filled in.

    Raises:
        InvalidCommandError: `kind` is unknown, or `args` has a missing, unknown, or invalid
            entry. The message names each failed argument.
    """
    spec = COMMAND_SPECS.get(kind)
    if spec is None:
        raise InvalidCommandError(
            f"unknown command kind {kind!r}; known kinds: {', '.join(sorted(COMMAND_SPECS))}"
        )
    try:
        checked: dict[str, Any] = spec.args.model_validate(args).model_dump()
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'args'}: {error['msg']}"
            for error in exc.errors()
        )
        raise InvalidCommandError(f"invalid args for {kind!r}: {problems}") from None
    return checked
