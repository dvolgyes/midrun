"""Register artifact publication and discovery endpoints."""

import json
from typing import Any

from fastapi import FastAPI, HTTPException

from midrun.rest.argument_error import argument_error
from midrun.rest.checkpoint_entries import checkpoint_entries
from midrun.rest.command_dict import command_dict
from midrun.rest.parse_when_or_422 import parse_when_or_422
from midrun.rest.require_artifact_store import require_artifact_store
from midrun.rest.schemas.checkpoint_in import CheckpointIn
from midrun.rest.schemas.debug_capture_in import DebugCaptureIn
from midrun.rest.schemas.snapshot_in import SnapshotIn
from midrun.runtime.runtime import MidrunRuntime


def register_artifact_routes(app: FastAPI, runtime: MidrunRuntime) -> None:
    """Register artifact publication and discovery endpoints."""

    @app.post("/checkpoints")
    def create_checkpoint(body: CheckpointIn) -> dict[str, Any]:
        """Queues a checkpoint with the requested label."""
        when = parse_when_or_422(body.when)
        command = runtime.command_queue.submit(
            "checkpoint", {"label": body.label}, when
        )
        runtime.gate.notify()
        return command_dict(command)

    @app.get("/checkpoints")
    def list_checkpoints() -> dict[str, Any]:
        """Lists the saved checkpoints with their download URLs."""
        root = require_artifact_store(runtime.artifacts).root
        return {"checkpoints": checkpoint_entries(root / "checkpoints")}

    @app.post("/snapshots")
    def create_snapshot(body: SnapshotIn) -> dict[str, Any]:
        """Queues a snapshot of the requested kind."""
        when = parse_when_or_422(body.when)
        command = runtime.command_queue.submit("snapshot", {"kind": body.kind}, when)
        runtime.gate.notify()
        return command_dict(command)

    @app.get("/artifacts")
    def list_artifacts() -> dict[str, Any]:
        """Lists published artifacts by category."""
        root = require_artifact_store(runtime.artifacts).root
        # An operator can delete a store directory while the run continues.
        return {
            subdir: sorted(entry.name for entry in (root / subdir).glob(pattern))
            if (root / subdir).is_dir()
            else []
            for subdir, pattern in (
                ("checkpoints", "*.ckpt"),
                ("snapshots", "*"),
                ("debug", "*"),
            )
        }

    @app.get("/artifacts/{artifact_id}")
    def get_artifact(artifact_id: str) -> dict[str, Any]:
        """Returns a published capture or snapshot manifest."""
        root = require_artifact_store(runtime.artifacts).root
        for subdir in ("snapshots", "debug"):
            manifest_path = root / subdir / artifact_id / "manifest.json"
            if manifest_path.exists():
                manifest: dict[str, Any] = json.loads(manifest_path.read_text())
                manifest["files_url"] = f"/artifact-files/{subdir}/{artifact_id}"
                return manifest
        raise HTTPException(status_code=404, detail="unknown artifact id")

    @app.post("/debug/captures")
    def create_debug_capture(body: DebugCaptureIn) -> dict[str, Any]:
        """Publishes a capture with validated filenames."""
        store = require_artifact_store(runtime.artifacts)
        try:
            return store.write_debug_capture(body.kind, body.files, body.metadata)
        except ValueError as exc:
            raise argument_error(("body", "files"), str(exc)) from exc
