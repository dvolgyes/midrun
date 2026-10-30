"""Starts this run's REST server in a background thread."""

import socket
import threading

import uvicorn
from fastapi import FastAPI

from midrun.runtime.rest_base_url import rest_base_url
from midrun.runtime.wait_own_rest_ready import wait_own_rest_ready


def start_rest_server(
    app: FastAPI, sock: socket.socket, host: str, port: int, run_id: str
) -> tuple[uvicorn.Server, threading.Thread]:
    """Serves `app` on `sock` and waits until the server reports `run_id`.

    Once `should_exit` is set, the server cancels requests still running after 3 s,
    so that it stops within the 5 s join of `MidrunRuntime._stop_rest`.

    Returns:
        The server and the daemon thread that runs it.

    Raises:
        RuntimeError: The server did not report `run_id` in time. The server is
            stopped and `sock` is closed first.
    """
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
            timeout_graceful_shutdown=3,
        )
    )
    thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [sock]},
        name="midrun-rest",
        daemon=True,
    )
    thread.start()
    if not wait_own_rest_ready(
        f"{rest_base_url(host, port)}/health", run_id, timeout=10
    ):
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()
        raise RuntimeError("REST server did not become ready in time")
    return server, thread
