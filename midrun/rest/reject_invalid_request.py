"""Maps a request that fails validation to status 422."""

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


async def reject_invalid_request(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Returns status 422 with the validation errors, without the rejected input.

    The input can be a value that JSON cannot encode, such as NaN, which would turn
    FastAPI's default response into status 500.
    """
    del request
    errors = [
        {key: value for key, value in error.items() if key != "input"}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})
