"""FastAPI app exposing Trainctl's REST control/debug plane on global rank 0.

Every mutating route only enqueues a `Command` or calls `RunGate.release` (safe from any
thread); the Lightning training thread executes commands from
`TrainctlRuntime._run_due_commands`. REST must never touch the Trainer/model directly —
see the module docstring in `trainctl.runtime.runtime`.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from loguru import logger

from trainctl.rest.command_dict import command_dict
from trainctl.rest.parse_when_or_400 import parse_when_or_400
from trainctl.rest.reject_invalid_request import reject_invalid_request
from trainctl.rest.reject_refused_command import reject_refused_command
from trainctl.rest.schemas.checkpoint_in import CheckpointIn
from trainctl.rest.schemas.command_in import CommandIn
from trainctl.rest.schemas.debug_capture_in import DebugCaptureIn
from trainctl.rest.schemas.hold_in import HoldIn
from trainctl.rest.schemas.snapshot_in import SnapshotIn
from trainctl.rest.wait_for_command import wait_for_command
from trainctl.runtime import inspectors_registry
from trainctl.runtime.command_args.validate_command_args import InvalidCommandError
from trainctl.runtime.commands.parse_when import parse_when
from trainctl.runtime.commands.refused_command_error import RefusedCommandError
from trainctl.runtime.cuda_memory.cheap_cuda_memory_stats import (
    cheap_cuda_memory_stats,
)
from trainctl.runtime.cuda_memory.dump_cuda_memory_snapshot import (
    dump_cuda_memory_snapshot,
)
from trainctl.runtime.cuda_memory.set_cuda_memory_history import (
    set_cuda_memory_history,
)
from trainctl.runtime.virtual_fs.is_virtual_dir import is_virtual_dir
from trainctl.runtime.virtual_fs.list_virtual_dir import list_virtual_dir
from trainctl.runtime.virtual_fs.virtual_file_content import virtual_file_content

_STATIC_DIR = Path(__file__).parent / "static"
_MEDIA_TYPES = {".json": "application/json"}

if TYPE_CHECKING:
    from trainctl.runtime.runtime import TrainctlRuntime


def build_app(runtime: "TrainctlRuntime") -> FastAPI:
    """Builds the REST app bound to one `TrainctlRuntime` instance."""
    app = FastAPI(
        title="trainctl",
        exception_handlers={
            RequestValidationError: reject_invalid_request,
            RefusedCommandError: reject_refused_command,
        },
    )

    @app.exception_handler(InvalidCommandError)
    def reject_invalid_command(
        request: Request, exc: InvalidCommandError
    ) -> JSONResponse:
        """Maps a command with an unknown kind or invalid args to status 400."""
        del request
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    if runtime.artifacts is not None:
        app.mount(
            "/artifact-files",
            StaticFiles(directory=str(runtime.artifact_store.root)),
            name="artifact-files",
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "run_id": runtime.run_id}

    @app.get("/run")
    def get_run() -> dict[str, Any]:
        return asdict(runtime.state.read())

    @app.get("/metrics")
    def get_metrics() -> dict[str, Any]:
        return {
            key: asdict(metric) for key, metric in runtime.state.read().metrics.items()
        }

    @app.get("/optimizers")
    def get_optimizers() -> dict[str, Any]:
        return {"learning_rates": list(runtime.state.read().learning_rates)}

    @app.get("/commands")
    def list_commands() -> list[dict[str, Any]]:
        return [command_dict(command) for command in runtime.command_queue.list_all()]

    @app.get("/commands/{command_id}")
    def get_command(command_id: str) -> dict[str, Any]:
        command = runtime.command_queue.get(command_id)
        if command is None:
            raise HTTPException(status_code=404, detail="unknown command id")
        return command_dict(command)

    @app.post("/commands")
    def submit_command(body: CommandIn) -> dict[str, Any]:
        if body.kind == "release_exception" and runtime.state.read().exception is None:
            raise HTTPException(status_code=409, detail="no exception hold is active")
        when = parse_when_or_400(body.execution.when)
        command = runtime.command_queue.submit(body.kind, body.args, when)
        runtime.gate.notify()
        return command_dict(command)

    @app.get("/holds")
    def list_holds() -> dict[str, Any]:
        commands = runtime.command_queue.list_all()
        active = runtime.gate.active_hold()
        held = active is not None or runtime.state.read().exception is not None
        return {
            "holds": [command_dict(c) for c in commands if c.kind == "hold"],
            "active": asdict(active) if active else None,
            # A hold runs only some kinds; the other queued commands wait for its end.
            "waiting": [
                command_dict(c)
                for c in commands
                if held and c.kind != "hold" and c.status == "queued"
            ],
        }

    # Async, so a request that waits for its hold uses no worker thread.
    @app.post("/holds", response_model=None)
    async def arm_hold(body: HoldIn) -> dict[str, Any] | JSONResponse:
        when = parse_when_or_400(body.when)
        command = runtime.command_queue.submit("hold", {"reason": body.reason}, when)
        runtime.gate.notify()
        command = await wait_for_command(runtime, command.id, timeout_s=body.timeout_s)
        if command.status == "queued":
            return JSONResponse(status_code=202, content=command_dict(command))
        return command_dict(command)

    @app.post("/holds/{command_id}/cancel")
    def cancel_hold(command_id: str) -> dict[str, Any]:
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
        active = runtime.gate.active_hold()
        if active is None or active.command_id != command_id:
            raise HTTPException(status_code=404, detail="no active hold with that id")
        runtime.gate.release()
        return {"released": command_id}

    @app.post("/control/pause")
    def control_pause() -> dict[str, Any]:
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
        return {"released": runtime.gate.release()}

    @app.post("/control/stop")
    def control_stop() -> dict[str, Any]:
        command = runtime.command_queue.submit("stop", {}, parse_when("now"))
        runtime.gate.notify()
        return command_dict(command)

    @app.post("/checkpoints")
    def create_checkpoint(body: CheckpointIn) -> dict[str, Any]:
        when = parse_when_or_400(body.when)
        command = runtime.command_queue.submit(
            "checkpoint", {"label": body.label}, when
        )
        runtime.gate.notify()
        return command_dict(command)

    @app.post("/snapshots")
    def create_snapshot(body: SnapshotIn) -> dict[str, Any]:
        when = parse_when_or_400(body.when)
        command = runtime.command_queue.submit("snapshot", {"kind": body.kind}, when)
        runtime.gate.notify()
        return command_dict(command)

    @app.get("/artifacts")
    def list_artifacts() -> dict[str, Any]:
        root = runtime.artifact_store.root
        return {
            "checkpoints": sorted(
                entry.name for entry in (root / "checkpoints").glob("*.ckpt")
            ),
            "snapshots": sorted(entry.name for entry in (root / "snapshots").iterdir()),
            "debug": sorted(entry.name for entry in (root / "debug").iterdir()),
        }

    @app.get("/artifacts/{artifact_id}")
    def get_artifact(artifact_id: str) -> dict[str, Any]:
        for subdir in ("snapshots", "debug"):
            manifest_path = (
                runtime.artifact_store.root / subdir / artifact_id / "manifest.json"
            )
            if manifest_path.exists():
                manifest: dict[str, Any] = json.loads(manifest_path.read_text())
                manifest["files_url"] = f"/artifact-files/{subdir}/{artifact_id}"
                return manifest
        raise HTTPException(status_code=404, detail="unknown artifact id")

    @app.post("/debug/captures")
    def create_debug_capture(body: DebugCaptureIn) -> dict[str, Any]:
        try:
            return runtime.artifact_store.write_debug_capture(
                body.kind, body.files, body.metadata
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/debug/pytorch/status")
    def debug_pytorch_status() -> dict[str, Any]:
        return {
            "enabled": runtime.torch_debug._started,
            "url": runtime.state.read().torch_debug_url,
        }

    @app.get("/debug/cuda-memory/snapshot")
    def cuda_memory_snapshot() -> dict[str, Any]:
        try:
            snapshot_bytes = dump_cuda_memory_snapshot()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        stats = cheap_cuda_memory_stats()
        manifest = runtime.artifact_store.write_debug_capture(
            "cuda-memory",
            {
                "snapshot.pickle": snapshot_bytes,
                "summary.json": json.dumps(stats, indent=2),
            },
        )
        manifest["files_url"] = f"/artifact-files/debug/{manifest['id']}"
        return manifest

    @app.get("/debug/cuda-memory/history")
    def cuda_memory_history(
        enabled: bool, max_entries: Annotated[int | None, Query(ge=1)] = None
    ) -> dict[str, Any]:
        try:
            state = set_cuda_memory_history(enabled, max_entries)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        runtime.cuda_memory_history = state
        return asdict(state)

    @app.get("/telemetry/pipeline")
    def telemetry_pipeline() -> dict[str, Any]:
        latest = runtime.pipeline.latest()
        return {
            "latest": asdict(latest) if latest else None,
            "summary": runtime.pipeline.summary(),
        }

    @app.get("/inspectors")
    def get_inspectors() -> dict[str, Any]:
        return inspectors_registry.availability_snapshot()

    @app.get("/files/{path:path}")
    def get_file(path: str) -> Response:
        virtual_path = "/" + path
        if is_virtual_dir(runtime, virtual_path):
            return JSONResponse(
                {
                    "path": virtual_path,
                    "entries": list_virtual_dir(runtime, virtual_path),
                }
            )
        try:
            content = virtual_file_content(runtime, virtual_path)
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except Exception as exc:
            logger.warning("trainctl GET /files{} failed: {}", virtual_path, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if content is None:
            raise HTTPException(status_code=404, detail="unknown virtual path")
        media_type = _MEDIA_TYPES.get(Path(virtual_path).suffix, "text/plain")
        return Response(content=content, media_type=media_type)

    return app
