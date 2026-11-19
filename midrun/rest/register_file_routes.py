"""Register the read-only virtual filesystem endpoint."""

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from loguru import logger

from midrun.rest.prefers_html import prefers_html
from midrun.rest.virtual_dir_page import virtual_dir_page
from midrun.runtime.runtime import MidrunRuntime
from midrun.runtime.virtual_fs.is_virtual_dir import is_virtual_dir
from midrun.runtime.virtual_fs.list_virtual_dir import list_virtual_dir
from midrun.runtime.virtual_fs.virtual_file_content import virtual_file_content


def register_file_routes(app: FastAPI, runtime: MidrunRuntime) -> None:
    """Register the read-only virtual filesystem endpoint."""

    @app.get("/files/{path:path}")
    def get_file(path: str, request: Request) -> Response:
        """Serves virtual files and negotiates directory listings."""
        virtual_path = "/" + path
        if is_virtual_dir(runtime, virtual_path):
            shown = virtual_path.rstrip("/") or "/"
            entries = list_virtual_dir(runtime, virtual_path)
            # A browser gets a page of links; curl and scripts get JSON.
            if prefers_html(request.headers.get("accept", "")):
                children = [
                    (name, is_virtual_dir(runtime, f"{shown.rstrip('/')}/{name}"))
                    for name in entries
                ]
                return HTMLResponse(
                    virtual_dir_page(shown, children),
                    headers={
                        "Vary": "Accept",
                        "Content-Security-Policy": "default-src 'none'",
                    },
                )
            return JSONResponse(
                {"path": shown, "entries": entries}, headers={"Vary": "Accept"}
            )
        try:
            content = virtual_file_content(runtime, virtual_path)
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except Exception as exc:
            logger.warning("midrun GET /files{} failed: {}", virtual_path, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if content is None:
            raise HTTPException(status_code=404, detail="unknown virtual path")
        media_type = (
            "application/json" if Path(virtual_path).suffix == ".json" else "text/plain"
        )
        return Response(content=content, media_type=media_type)
