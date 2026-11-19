"""Register hold management and training control endpoints."""

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from midrun.rest.command_dict import command_dict
from midrun.rest.parse_when_or_422 import parse_when_or_422
from midrun.rest.schemas.hold_in import HoldIn
from midrun.rest.wait_for_command import wait_for_command
from midrun.runtime.command_specs.allowed_during_exception import (
    ALLOWED_DURING_EXCEPTION,
)
from midrun.runtime.command_specs.allowed_during_hold import ALLOWED_DURING_HOLD
from midrun.runtime.command_specs.runnable_kinds import runnable_kinds
from midrun.runtime.commands.parse_when import parse_when
from midrun.runtime.runtime import MidrunRuntime


def register_hold_routes(app: FastAPI, runtime: MidrunRuntime) -> None:
    """Register hold management and training control endpoints."""

    @app.get("/holds")
    def list_holds() -> dict[str, Any]:
        """Reports active holds and commands waiting for release."""
        commands = runtime.command_queue.list_all()
        active = runtime.gate.active_hold()
        state = runtime.state.read()
        exception = state.exception
        # While an exception is held, the training thread runs only the exception
        # hold loop, so that loop's policy applies.
        allowed: frozenset[str] | None = None
        if exception is not None:
            allowed = ALLOWED_DURING_EXCEPTION
        elif active is not None:
            allowed = ALLOWED_DURING_HOLD
        if allowed is not None:
            allowed = runnable_kinds(allowed, runtime.current_step.batch is not None)
        return {
            "holds": [command_dict(c) for c in commands if c.kind == "hold"],
            "active": asdict(active) if active else None,
            "exception": asdict(exception) if exception else None,
            "waiting": [
                command_dict(c)
                for c in commands
                if allowed is not None
                and c.kind != "hold"
                and c.status == "queued"
                and (
                    c.kind not in allowed
                    or not c.when.is_due(None, state.epoch, state.global_step)
                )
            ],
        }

    # Async, so a request that waits for its hold uses no worker thread.
    @app.post("/holds", response_model=None)
    async def arm_hold(body: HoldIn) -> dict[str, Any] | JSONResponse:
        """Arms a hold and waits up to the requested timeout."""
        when = parse_when_or_422(body.when)
        command = runtime.command_queue.submit("hold", {"reason": body.reason}, when)
        runtime.gate.notify()
        command = await wait_for_command(runtime, command.id, timeout_s=body.timeout_s)
        if command.status == "queued":
            return JSONResponse(status_code=202, content=command_dict(command))
        return command_dict(command)

    @app.post("/holds/{command_id}/cancel")
    def cancel_hold(command_id: str) -> dict[str, Any]:
        """Cancels an unresolved hold."""
        command = runtime.command_queue.get(command_id)
        if (
            command is None
            or command.kind != "hold"
            or not runtime.command_queue.cancel(command_id)
        ):
            raise HTTPException(
                status_code=404,
                detail="hold not cancellable (unknown id, not a hold, or already resolved)",
            )
        return {"cancelled": command_id}

    @app.post("/holds/{command_id}/release")
    def release_hold(command_id: str) -> dict[str, Any]:
        """Releases the active hold by its own ID or a joined ID."""
        if not runtime.gate.release(command_id=command_id):
            raise HTTPException(status_code=404, detail="no active hold with that id")
        return {"released": command_id}

    @app.post("/control/pause")
    def control_pause() -> dict[str, Any]:
        """Requests a pause unless one is already active or queued."""
        # A repeated pause queues nothing, so one continue always resumes the run.
        active = runtime.gate.active_hold()
        if active is not None:
            return {"hold_id": active.command_id, "queued": False}
        command, queued = runtime.command_queue.submit_once(
            "hold", {"reason": "control/pause"}, parse_when("now")
        )
        runtime.gate.notify()
        return {"hold_id": command.id, "queued": queued}

    @app.post("/control/continue")
    def control_continue() -> dict[str, Any]:
        """Cancels pending pause requests and releases the active hold."""
        # Cancels a pause that is not due yet before the release, so it cannot hold
        # the run at the next safe point. Holds from /holds stay queued.
        cancelled = [
            command.id
            for command in runtime.command_queue.list_all()
            if (command.kind, command.args, command.when)
            == ("hold", {"reason": "control/pause"}, parse_when("now"))
            and runtime.command_queue.cancel(command.id)
        ]
        return {"released": runtime.gate.release(), "cancelled_holds": cancelled}

    @app.post("/control/stop")
    def control_stop() -> dict[str, Any]:
        """Queues a stop at the next safe point."""
        command = runtime.command_queue.submit("stop", {}, parse_when("now"))
        runtime.gate.notify()
        return command_dict(command)
