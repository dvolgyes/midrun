"""Behavioral tests for POST /holds and the hold routes.

A request that waits for its hold to engage must not stop other requests from being
served.
"""

import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from trainctl.config import TrainctlConfig
from trainctl.lightning_backend import load_backend
from trainctl.rest.app import build_app
from trainctl.runtime.artifacts import ArtifactStore
from trainctl.runtime.runtime import TrainctlRuntime


def _runtime(tmp_path: Path) -> TrainctlRuntime:
    config = TrainctlConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = TrainctlRuntime(config, load_backend("lightning.pytorch"))
    runtime.artifacts = ArtifactStore(tmp_path)
    return runtime


def test_waiting_hold_requests_do_not_block_health(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    # More waiting requests than the server's default worker-thread limit (40).
    waiting_requests = 45
    statuses: list[int] = []
    with TestClient(build_app(runtime)) as client:

        def arm() -> None:
            response = client.post("/holds", json={"timeout_s": 5})
            statuses.append(response.status_code)

        threads = [threading.Thread(target=arm) for _ in range(waiting_requests)]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + 5
        while (
            len(runtime.command_queue.list_all()) < 40 and time.monotonic() < deadline
        ):
            time.sleep(0.01)

        started = time.monotonic()
        health = client.get("/health")
        elapsed = time.monotonic() - started

        for thread in threads:
            thread.join()

    assert health.status_code == 200
    assert elapsed < 2, f"/health took {elapsed:.1f} s while holds waited"
    # No training thread runs, so no hold engages and every request times out.
    assert statuses == [202] * waiting_requests


def test_hold_timeout_above_the_limit_is_rejected(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    client = TestClient(build_app(runtime))

    response = client.post("/holds", json={"timeout_s": 61})

    assert response.status_code == 422, response.text
    assert runtime.command_queue.list_all() == []


def test_negative_hold_timeout_is_rejected(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    client = TestClient(build_app(runtime))

    response = client.post("/holds", json={"timeout_s": -1})

    assert response.status_code == 422, response.text
    assert runtime.command_queue.list_all() == []
