"""Behavioral tests for POST /holds and the hold routes.

A request that waits for its hold to engage must not stop other requests from being
served.
"""

import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.rest.app import build_app
from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.commands.parse_when import parse_when
from midrun.runtime.runtime import MidrunRuntime


def _runtime(tmp_path: Path) -> MidrunRuntime:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = MidrunRuntime(config, load_backend("lightning.pytorch"))
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


def test_cancel_of_a_command_that_is_not_a_hold_is_rejected(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    client = TestClient(build_app(runtime))
    command = runtime.command_queue.submit(
        "snapshot", {"kind": "runtime"}, parse_when("now")
    )

    response = client.post(f"/holds/{command.id}/cancel")

    assert response.status_code == 404, response.text
    assert runtime.command_queue.get(command.id).status == "queued"


def test_holds_lists_the_commands_that_wait_while_held(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    client = TestClient(build_app(runtime))
    command = runtime.command_queue.submit("save_current_batch", {}, parse_when("now"))
    assert client.get("/holds").json()["waiting"] == []

    runtime.gate.enter("hold_1", "train_batch_end", epoch=0, step=0)

    waiting = client.get("/holds").json()["waiting"]
    assert [entry["id"] for entry in waiting] == [command.id]


def test_release_exception_without_an_exception_hold_is_rejected(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    client = TestClient(build_app(runtime))

    response = client.post("/commands", json={"kind": "release_exception"})

    assert response.status_code == 409, response.text
    assert runtime.command_queue.list_all() == []


def test_a_non_finite_hold_timeout_is_rejected_as_invalid(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    client = TestClient(build_app(runtime), raise_server_exceptions=False)

    response = client.post(
        "/holds",
        content='{"timeout_s": NaN}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422, response.text
    assert response.json()["detail"][0]["loc"] == ["body", "timeout_s"]
    assert runtime.command_queue.list_all() == []
