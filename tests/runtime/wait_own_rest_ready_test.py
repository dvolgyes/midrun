"""Behavioral tests: the REST readiness probe reaches the local server directly."""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from midrun.runtime.wait_own_rest_ready import wait_own_rest_ready


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = json.dumps({"status": "ok", "run_id": "run-own"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


@pytest.fixture
def health_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/health"
    server.shutdown()
    server.server_close()
    thread.join()


def test_probe_ignores_a_configured_http_proxy(
    health_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Port 9 (discard) has no listener, so a probe sent through the proxy fails.
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)

    assert wait_own_rest_ready(health_url, "run-own", timeout=3)
