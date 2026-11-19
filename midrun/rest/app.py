"""FastAPI app exposing Midrun's REST control/debug plane on global rank 0.

Every mutating route only enqueues a `Command` or calls `RunGate.release` (safe from any
thread); the Lightning training thread executes commands from
`MidrunRuntime._run_due_commands`. REST must never touch the Trainer/model directly —
see the module docstring in `midrun.runtime.runtime`.
"""

import os
import socket
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from midrun.rest.argument_error import argument_error
from midrun.rest.attachment_static_files import AttachmentStaticFiles
from midrun.rest.command_dict import command_dict
from midrun.rest.parse_when_or_422 import parse_when_or_422
from midrun.rest.register_artifact_routes import register_artifact_routes
from midrun.rest.register_debug_routes import register_debug_routes
from midrun.rest.register_file_routes import register_file_routes
from midrun.rest.register_hold_routes import register_hold_routes
from midrun.rest.register_status_routes import register_status_routes
from midrun.rest.reject_invalid_request import reject_invalid_request
from midrun.rest.reject_refused_command import reject_refused_command
from midrun.rest.schemas.command_in import CommandIn
from midrun.rest.trusted_hosts import trusted_hosts
from midrun.runtime.command_args.validate_command_args import InvalidCommandError
from midrun.runtime.command_specs.needs_batch import NEEDS_BATCH
from midrun.runtime.commands.refused_command_error import RefusedCommandError
from midrun.runtime.events import SafePoint
from midrun.runtime.runtime import MidrunRuntime

_STATIC_DIR = Path(__file__).parent / "static"


def build_app(runtime: MidrunRuntime) -> FastAPI:
    """Builds the REST app bound to one `MidrunRuntime` instance."""
    app = FastAPI(
        title="midrun",
        exception_handlers={
            RequestValidationError: reject_invalid_request,
            RefusedCommandError: reject_refused_command,
        },
    )
    # Host-header allowlist: a page that rebinds its domain to this machine
    # must not read the API as same-origin.
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=trusted_hosts(
            runtime.config.rest_host,
            runtime.config.rest_allowed_hosts,
            os.environ,
            (socket.gethostname(), socket.getfqdn()),
        ),
        www_redirect=False,
    )

    @app.exception_handler(InvalidCommandError)
    async def reject_invalid_command(
        request: Request, exc: InvalidCommandError
    ) -> JSONResponse:
        """Maps a command with an unknown kind or invalid args to status 422."""
        return await reject_invalid_request(
            request, argument_error(("body",), str(exc))
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        """Serves the bundled control interface."""
        return FileResponse(_STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    if runtime.artifacts is not None:
        for category in ("checkpoints", "snapshots", "debug"):
            app.mount(
                f"/artifact-files/{category}",
                AttachmentStaticFiles(
                    directory=str(runtime.artifact_store.root / category)
                ),
                name=f"artifact-files-{category}",
            )

    register_status_routes(app, runtime)
    _register_command_routes(app, runtime)
    register_hold_routes(app, runtime)
    register_artifact_routes(app, runtime)
    register_debug_routes(app, runtime)
    register_file_routes(app, runtime)

    return app


def _register_command_routes(app: FastAPI, runtime: MidrunRuntime) -> None:
    """Register command submission and inspection endpoints."""

    @app.get("/commands")
    def list_commands() -> list[dict[str, Any]]:
        """Lists commands in queue order."""
        return [command_dict(command) for command in runtime.command_queue.list_all()]

    @app.get("/commands/{command_id}")
    def get_command(command_id: str) -> dict[str, Any]:
        """Returns one command by ID."""
        command = runtime.command_queue.get(command_id)
        if command is None:
            raise HTTPException(status_code=404, detail="unknown command id")
        return command_dict(command)

    @app.post("/commands")
    def submit_command(body: CommandIn) -> dict[str, Any]:
        """Validates and enqueues a runtime command."""
        if body.kind == "release_exception" and runtime.state.read().exception is None:
            raise HTTPException(status_code=409, detail="no exception hold is active")
        when = parse_when_or_422(body.execution.when, loc=("body", "execution", "when"))
        if body.kind in NEEDS_BATCH and when.safe_point not in (
            None,
            SafePoint.TRAIN_BATCH_END,
            SafePoint.BEFORE_OPTIMIZER_STEP,
        ):
            # Such a command would wait at this safe point forever.
            raise argument_error(
                ("body", "execution", "when"),
                f"{body.kind} requires an in-flight batch, and "
                f"{when.safe_point} has none",
            )
        command = runtime.command_queue.submit(body.kind, body.args, when)
        runtime.gate.notify()
        return command_dict(command)
