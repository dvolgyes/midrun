"""Behavioral tests for the REST app's Host-header check against DNS rebinding."""

import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import uvicorn
from fastapi.testclient import TestClient

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.rest.app import build_app
from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.runtime import MidrunRuntime


def _runtime(tmp_path: Path, **options: object) -> MidrunRuntime:
    config = MidrunConfig(
        rest_enabled=False,
        fuse_enabled=False,
        torch_debug_enabled=False,
        **options,  # type: ignore[arg-type]
    )
    runtime = build_runtime(config, load_backend("lightning.pytorch"))
    runtime.artifacts = ArtifactStore(tmp_path)
    return runtime


@pytest.mark.parametrize(
    "path", ["/run", "/health", "/files/state", "/static/app.js", "/", "/openapi.json"]
)
@pytest.mark.parametrize(
    "host",
    [
        "evil.example:8090",
        "evil.example",
        "localhost.evil.example:8090",
        "127.0.0.1.nip.io:8090",
        "evil.example:8090@localhost",
        "[::ffff:127.0.0.1]:8090",
        "",
    ],
)
def test_foreign_host_is_rejected_on_read_routes(
    tmp_path: Path, path: str, host: str
) -> None:
    client = TestClient(build_app(_runtime(tmp_path)))
    response = client.get(path, headers={"host": host})
    assert response.status_code == 400


def test_foreign_host_is_rejected_on_artifact_files(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    (tmp_path / "debug").mkdir(exist_ok=True)
    (tmp_path / "debug" / "x.txt").write_text("secret")
    client = TestClient(build_app(runtime))
    response = client.get(
        "/artifact-files/debug/x.txt", headers={"host": "evil.example:8090"}
    )
    assert response.status_code == 400
    assert "secret" not in response.text


@pytest.mark.parametrize(
    "host", ["localhost:8090", "127.0.0.1:8090", "[::1]:8090", "LOCALHOST:8090"]
)
def test_loopback_host_is_accepted(tmp_path: Path, host: str) -> None:
    client = TestClient(build_app(_runtime(tmp_path)))
    assert client.get("/run", headers={"host": host}).status_code == 200


def test_ssh_connection_server_ip_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SSH_CONNECTION", "192.0.2.10 51234 198.51.100.7 22")
    client = TestClient(build_app(_runtime(tmp_path)))
    assert client.get("/run", headers={"host": "198.51.100.7:8090"}).status_code == 200
    assert client.get("/run", headers={"host": "192.0.2.10:8090"}).status_code == 400


def test_rest_allowed_hosts_entry_is_accepted(tmp_path: Path) -> None:
    client = TestClient(
        build_app(_runtime(tmp_path, rest_allowed_hosts=("proxy.example",)))
    )
    assert client.get("/run", headers={"host": "proxy.example"}).status_code == 200
    assert client.get("/run", headers={"host": "evil.example"}).status_code == 400


def test_machine_hostname_is_accepted(tmp_path: Path) -> None:
    client = TestClient(build_app(_runtime(tmp_path)))
    host = f"{socket.gethostname()}:8090"
    assert client.get("/run", headers={"host": host}).status_code == 200


@pytest.fixture
def served_port(tmp_path: Path) -> Iterator[int]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port: int = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            build_app(_runtime(tmp_path)),
            host="127.0.0.1",
            port=port,
            log_level="warning",
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
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()


def _raw_status(port: int, request: bytes) -> int | None:
    with socket.create_connection(("127.0.0.1", port), timeout=5) as conn:
        conn.sendall(request)
        reply = b""
        while b"\r\n" not in reply:
            chunk = conn.recv(4096)
            if not chunk:
                break
            reply += chunk
    if not reply.startswith(b"HTTP/"):
        return None
    return int(reply.split(b" ", 2)[1])


@pytest.mark.parametrize(
    "request_bytes",
    [
        b"GET /run HTTP/1.1\r\nHost: evil.example:8090\r\n\r\n",
        # An absolute-form target must not override the Host header.
        b"GET http://localhost:8090/run HTTP/1.1\r\nHost: evil.example\r\n\r\n",
        b"GET /run HTTP/1.0\r\n\r\n",
        b"GET /run HTTP/1.1\r\nHost: evil.example\r\nHost: localhost\r\n\r\n",
        b"GET /run HTTP/1.1\r\nHost: localhost\r\nHost: evil.example\r\n\r\n",
        b"GET /run HTTP/1.1\r\nHost: evil.example\r\nX-Forwarded-Host: localhost\r\n\r\n",
    ],
)
def test_real_server_rejects_bypass_attempts(
    served_port: int, request_bytes: bytes
) -> None:
    status = _raw_status(served_port, request_bytes)
    assert status is None or status >= 400


def test_real_server_accepts_loopback_host(served_port: int) -> None:
    request = f"GET /run HTTP/1.1\r\nHost: 127.0.0.1:{served_port}\r\n\r\n"
    assert _raw_status(served_port, request.encode()) == 200


def test_midrun_whitelist_ip_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MIDRUN_WHITELIST_IP", "203.0.113.9")
    client = TestClient(build_app(_runtime(tmp_path)))
    assert client.get("/run", headers={"host": "203.0.113.9:8090"}).status_code == 200
    assert client.get("/run", headers={"host": "203.0.113.8:8090"}).status_code == 400


def test_unset_midrun_whitelist_ip_adds_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MIDRUN_WHITELIST_IP", raising=False)
    client = TestClient(build_app(_runtime(tmp_path)))
    assert client.get("/run", headers={"host": "203.0.113.9:8090"}).status_code == 400
