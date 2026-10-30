"""Behavioral tests for POST /holds and the hold routes.

A request that waits for its hold to engage must not stop other requests from being
served.
"""

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.rest.app import build_app
from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.commands.parse_when import parse_when
from midrun.runtime.events import SafePoint
from midrun.runtime.runtime import MidrunRuntime
from midrun.runtime.state.exception_breakpoint_info import ExceptionBreakpointInfo


class _StepTrainer:
    current_epoch = 0
    global_step = 0


def _runtime(tmp_path: Path) -> MidrunRuntime:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = build_runtime(config, load_backend("lightning.pytorch"))
    runtime.artifacts = ArtifactStore(tmp_path)
    return runtime


def test_waiting_hold_requests_do_not_block_health(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    # More waiting requests than the server's default worker-thread limit (40).
    waiting_requests = 45
    statuses: list[int] = []
    with TestClient(build_app(runtime), base_url="http://localhost") as client:

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
    client = TestClient(build_app(runtime), base_url="http://localhost")

    response = client.post("/holds", json={"timeout_s": 61})

    assert response.status_code == 422, response.text
    assert runtime.command_queue.list_all() == []


def test_negative_hold_timeout_is_rejected(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    client = TestClient(build_app(runtime), base_url="http://localhost")

    response = client.post("/holds", json={"timeout_s": -1})

    assert response.status_code == 422, response.text
    assert runtime.command_queue.list_all() == []


def test_cancel_of_a_command_that_is_not_a_hold_is_rejected(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    client = TestClient(build_app(runtime), base_url="http://localhost")
    command = runtime.command_queue.submit(
        "snapshot", {"kind": "runtime"}, parse_when("now")
    )

    response = client.post(f"/holds/{command.id}/cancel")

    assert response.status_code == 404, response.text
    assert runtime.command_queue.get(command.id).status == "queued"


def test_holds_lists_the_commands_that_wait_while_held(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    client = TestClient(build_app(runtime), base_url="http://localhost")
    command = runtime.command_queue.submit("save_current_batch", {}, parse_when("now"))
    assert client.get("/holds").json()["waiting"] == []

    runtime.gate.enter("hold_1", "train_batch_end", epoch=0, step=0)

    waiting = client.get("/holds").json()["waiting"]
    assert [entry["id"] for entry in waiting] == [command.id]


def test_holds_does_not_list_a_command_that_runs_during_the_hold(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    client = TestClient(build_app(runtime), base_url="http://localhost")
    runtime.command_queue.submit("snapshot", {}, parse_when("now"))
    runtime.gate.enter("hold_1", "train_batch_end", epoch=0, step=0)

    assert client.get("/holds").json()["waiting"] == []


def test_holds_applies_the_exception_hold_policy_during_an_exception(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    client = TestClient(build_app(runtime), base_url="http://localhost")
    runtime.command_queue.submit("save_current_batch", {}, parse_when("now"))
    snapshot = runtime.command_queue.submit("snapshot", {}, parse_when("now"))
    exception = ExceptionBreakpointInfo(
        exception_type="RuntimeError",
        exception_message="boom",
        epoch=0,
        global_step=0,
        batch_idx=0,
        rank=0,
        started_at=0.0,
    )
    runtime.state.update(exception=exception)

    waiting = client.get("/holds").json()["waiting"]

    assert [entry["id"] for entry in waiting] == [snapshot.id]


def test_holds_shows_the_active_exception_hold(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    client = TestClient(build_app(runtime), base_url="http://localhost")
    assert client.get("/holds").json()["exception"] is None
    exception = ExceptionBreakpointInfo(
        exception_type="RuntimeError",
        exception_message="boom",
        epoch=0,
        global_step=3,
        batch_idx=3,
        rank=0,
        started_at=0.0,
    )

    runtime.state.update(exception=exception)

    shown = client.get("/holds").json()["exception"]
    assert shown["exception_type"] == "RuntimeError"
    assert shown["exception_message"] == "boom"
    assert shown["global_step"] == 3


def test_release_exception_without_an_exception_hold_is_rejected(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    client = TestClient(build_app(runtime), base_url="http://localhost")

    response = client.post("/commands", json={"kind": "release_exception"})

    assert response.status_code == 409, response.text
    assert runtime.command_queue.list_all() == []


def test_a_non_finite_hold_timeout_is_rejected_as_invalid(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    client = TestClient(
        build_app(runtime), base_url="http://localhost", raise_server_exceptions=False
    )

    response = client.post(
        "/holds",
        content='{"timeout_s": NaN}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422, response.text
    assert response.json()["detail"][0]["loc"] == ["body", "timeout_s"]
    assert runtime.command_queue.list_all() == []


@pytest.mark.parametrize("released", ["first", "second"])
def test_a_hold_due_while_another_is_active_joins_it(
    tmp_path: Path, released: str
) -> None:
    runtime = _runtime(tmp_path)
    client = TestClient(build_app(runtime), base_url="http://localhost")
    first = client.post("/holds", json={"reason": "A", "timeout_s": 0}).json()["id"]
    second = client.post("/holds", json={"reason": "B", "timeout_s": 0}).json()["id"]

    runtime._run_due_commands(
        None, _StepTrainer(), SafePoint.TRAIN_BATCH_END, allowed=None
    )

    assert runtime.gate.active_hold().command_id == first
    assert client.get(f"/commands/{second}").json()["result"] == {
        "hold": "joined",
        "active_hold_id": first,
        "reason": "B",
    }
    response = client.post(f"/holds/{first if released == 'first' else second}/release")
    assert response.status_code == 200, response.text
    assert not runtime.gate.is_held()
