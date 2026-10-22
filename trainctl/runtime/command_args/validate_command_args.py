"""Checks the arguments of a submitted command."""

from typing import Any

from pydantic import ValidationError

from trainctl.runtime.command_args.command_args_by_kind import COMMAND_ARGS


class InvalidCommandError(ValueError):
    """A submitted command has an unknown kind or invalid args."""


def validate_command_args(kind: str, args: dict[str, Any]) -> dict[str, Any]:
    """Checks `args` for a `kind` command and returns them with defaults filled in.

    Raises:
        InvalidCommandError: `kind` is unknown, or `args` has a missing, unknown, or invalid
            entry. The message names each failed argument.
    """
    model = COMMAND_ARGS.get(kind)
    if model is None:
        raise InvalidCommandError(
            f"unknown command kind {kind!r}; known kinds: {', '.join(sorted(COMMAND_ARGS))}"
        )
    try:
        checked: dict[str, Any] = model.model_validate(args).model_dump()
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'args'}: {error['msg']}"
            for error in exc.errors()
        )
        raise InvalidCommandError(f"invalid args for {kind!r}: {problems}") from None
    return checked
