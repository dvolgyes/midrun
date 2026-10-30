"""Behavioral tests for HoldClient against a real midrun REST server."""

import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import uvicorn

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.debugger.hold_client import HoldClient, HoldNotActiveError
from midrun.load_backend import load_backend
from midrun.rest.app import build_app
from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.events import SafePoint
from midrun.runtime.runtime import MidrunRuntime


class _StepTrainer:
    current_epoch = 0
    global_step = 0


@pytest.fixture
def served_runtime(tmp_path: Path) -> Iterator[tuple[MidrunRuntime, HoldClient]]:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = build_runtime(config, load_backend("lightning.pytorch"))
    runtime.artifacts = ArtifactStore(tmp_path)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            build_app(runtime), host="127.0.0.1", port=port, log_level="warning"
        )
    )
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [sock]}, daemon=True
    )
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        yield runtime, HoldClient(f"http://127.0.0.1:{port}")
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()


def test_require_hold_active_accepts_a_hold_that_joined_the_active_one(
    served_runtime: tuple[MidrunRuntime, HoldClient],
) -> None:
    runtime, client = served_runtime
    first = client.arm_hold(reason="A", timeout_s=0)["id"]
    second = client.arm_hold(reason="B", timeout_s=0)["id"]
    runtime._run_due_commands(
        None, _StepTrainer(), SafePoint.TRAIN_BATCH_END, allowed=None
    )

    client.require_hold_active(first)
    client.require_hold_active(second)


def test_require_hold_active_refuses_a_released_hold(
    served_runtime: tuple[MidrunRuntime, HoldClient],
) -> None:
    runtime, client = served_runtime
    hold_id = client.arm_hold(reason="A", timeout_s=0)["id"]
    runtime._run_due_commands(
        None, _StepTrainer(), SafePoint.TRAIN_BATCH_END, allowed=None
    )
    client.release_hold(hold_id)

    with pytest.raises(HoldNotActiveError):
        client.require_hold_active(hold_id)
