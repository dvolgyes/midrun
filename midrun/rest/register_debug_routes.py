"""Register PyTorch debug and CUDA allocator endpoints."""

import json
from dataclasses import asdict
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query

from midrun.rest.require_artifact_store import require_artifact_store
from midrun.runtime.cuda_memory.cheap_cuda_memory_stats import (
    cheap_cuda_memory_stats,
)
from midrun.runtime.cuda_memory.dump_cuda_memory_snapshot import (
    dump_cuda_memory_snapshot,
)
from midrun.runtime.cuda_memory.set_cuda_memory_history import (
    set_cuda_memory_history,
)
from midrun.runtime.runtime import MidrunRuntime


def register_debug_routes(app: FastAPI, runtime: MidrunRuntime) -> None:
    """Register PyTorch debug and CUDA allocator endpoints."""

    @app.get("/debug/pytorch/status")
    def debug_pytorch_status() -> dict[str, Any]:
        """Reports whether the PyTorch debug service is running."""
        return {
            "enabled": runtime.torch_debug.started,
            "url": runtime.state.read().torch_debug_url,
        }

    @app.get("/debug/cuda-memory/snapshot")
    def cuda_memory_snapshot() -> dict[str, Any]:
        """Captures the CUDA allocator snapshot and its summary."""
        store = require_artifact_store(runtime.artifacts)
        try:
            snapshot_bytes = dump_cuda_memory_snapshot()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        stats = cheap_cuda_memory_stats(runtime.rank_info.device)
        manifest = store.write_debug_capture(
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
        enabled: bool, *, max_entries: Annotated[int | None, Query(ge=1)] = None
    ) -> dict[str, Any]:
        """Enables or disables CUDA memory history."""
        try:
            state = set_cuda_memory_history(enabled, max_entries)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        runtime.cuda_memory_history = state
        return asdict(state)
