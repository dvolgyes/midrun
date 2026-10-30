"""Builds the error for a request argument that a route checks itself."""

from fastapi.exceptions import RequestValidationError


def argument_error(loc: tuple[str, ...], message: str) -> RequestValidationError:
    """Returns the error that rejects the argument at `loc` with `message`.

    The app answers it as it answers a request that fails its schema: status 422
    with `{"detail": [{"loc": [...], "msg": ..., "type": "value_error"}]}`.
    """
    return RequestValidationError([{"loc": loc, "msg": message, "type": "value_error"}])
