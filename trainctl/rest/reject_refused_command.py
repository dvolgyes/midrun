"""Maps a command whose kind the run no longer accepts to status 409."""

from fastapi import Request
from fastapi.responses import JSONResponse

from trainctl.runtime.commands.refused_command_error import RefusedCommandError


async def reject_refused_command(
    request: Request, exc: RefusedCommandError
) -> JSONResponse:
    """Returns status 409 with the reason for the refusal."""
    del request
    return JSONResponse(status_code=409, content={"detail": str(exc)})
