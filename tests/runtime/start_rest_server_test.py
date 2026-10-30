"""Behavioral tests: a REST server stops after a failed readiness probe or a stop."""

import asyncio
import socket
import threading

import pytest
from fastapi import FastAPI

from midrun.runtime import start_rest_server as start_rest_server_module
from midrun.runtime.start_rest_server import start_rest_server


def test_a_server_that_misses_its_readiness_probe_is_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        start_rest_server_module,
        "wait_own_rest_ready",
        lambda url, run_id, timeout: False,
    )
    sock = socket.create_server(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    rest_threads_before = {
        thread for thread in threading.enumerate() if thread.name == "midrun-rest"
    }

    with pytest.raises(RuntimeError, match="did not become ready"):
        start_rest_server(FastAPI(), sock, "127.0.0.1", port, "run-own")

    assert {
        thread for thread in threading.enumerate() if thread.name == "midrun-rest"
    } == rest_threads_before
    assert sock.fileno() == -1
    with socket.socket() as client:
        assert client.connect_ex(("127.0.0.1", port)) != 0


def test_a_stopped_server_exits_before_its_stop_join_timeout_with_a_hung_request() -> (
    None
):
    entered = threading.Event()
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"run_id": "run-own"}

    @app.get("/slow")
    async def slow() -> None:
        entered.set()
        await asyncio.sleep(60)

    sock = socket.create_server(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server, thread = start_rest_server(app, sock, "127.0.0.1", port, "run-own")
    with socket.create_connection(("127.0.0.1", port)) as client:
        client.sendall(b"GET /slow HTTP/1.1\r\nHost: localhost\r\n\r\n")
        assert entered.wait(timeout=5)

        server.should_exit = True
        # The same join timeout as `MidrunRuntime._stop_rest`.
        thread.join(timeout=5)

        assert not thread.is_alive()
