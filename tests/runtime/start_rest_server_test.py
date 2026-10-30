"""Behavioral tests: a REST server that fails its readiness probe does not keep running."""

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
