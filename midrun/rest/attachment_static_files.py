"""Serves stored artifact files so a browser never renders them in Midrun's origin."""

from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from starlette.types import Scope


class AttachmentStaticFiles(StaticFiles):
    """`StaticFiles` that marks every response as a sandboxed download.

    A debug capture can hold any uploaded file, e.g. HTML or an SVG with script.
    Opened directly, it downloads, and a sandboxed document cannot run script with
    access to this origin. `<img>` and `fetch()` ignore these headers, so the
    dashboard previews still work.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        """Returns the file response with the isolation headers added."""
        response = await super().get_response(path, scope)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "sandbox"
        response.headers["Content-Disposition"] = "attachment"
        return response
