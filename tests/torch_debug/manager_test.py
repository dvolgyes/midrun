"""Behavioral tests: the debug frontend readiness probe reaches the frontend directly."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from midrun.torch_debug.manager import TorchDebugManager


class _FrontendHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def test_wait_ready_ignores_a_configured_http_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Port 9 (discard) has no listener, so a probe sent through the proxy fails.
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FrontendHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    manager = TorchDebugManager(port=server.server_address[1], port_search=1)
    manager._selected_port = server.server_address[1]
    try:
        assert manager.wait_ready(timeout=3)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
