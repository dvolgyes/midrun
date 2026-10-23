"""Behavioral tests for the gdb call-injection capture mode.

Retry-on-signal-collision, the hold-confirmation refusal path, and the
no-dangling-process invariant are tested against real gdb attaches to real
target processes -- this technique's whole value proposition is what it does to
a live OS process, which cannot be meaningfully mocked.
"""

import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from trainctl.debugger.gdb_capture.check_gdb_available import (
    GdbUnavailableError,
    check_gdb_available,
)
from trainctl.debugger.gdb_capture.gdb_attach_error import GdbAttachError
from trainctl.debugger.gdb_capture.gdb_capture import gdb_capture
from trainctl.debugger.hold_client import HoldClient, HoldNotActiveError

pytestmark = pytest.mark.skipif(shutil.which("gdb") is None, reason="gdb not installed")

_PTRACE_SCOPE = Path("/proc/sys/kernel/yama/ptrace_scope")
_YAMA_SCOPE = int(_PTRACE_SCOPE.read_text()) if _PTRACE_SCOPE.exists() else 0
_IS_ROOT = os.geteuid() == 0
# Yama scope 1 lets gdb attach to a sibling only when the target opts in with
# prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY), as the targets below do; scope 2 and
# 3 ignore that opt-in.
_attach_permitted = pytest.mark.skipif(
    _YAMA_SCOPE >= 2 and not _IS_ROOT,
    reason="kernel.yama.ptrace_scope >= 2 permits ptrace only to root",
)
_ALLOW_ANY_TRACER = (
    "import ctypes\nctypes.CDLL(None).prctl(0x59616D61, ctypes.c_ulong(-1), 0, 0, 0)\n"
)


@pytest.fixture
def target_process() -> Iterator[subprocess.Popen[bytes]]:
    proc = subprocess.Popen(  # noqa: S603 -- fixed, non-shell argv; test-only target process
        [
            sys.executable,
            "-c",
            _ALLOW_ANY_TRACER
            + "import time\ncounter = 0\nwhile True:\n    counter += 1\n    time.sleep(0.01)\n",
        ]
    )
    time.sleep(0.3)
    try:
        yield proc
    finally:
        proc.kill()
        proc.wait(timeout=5)


@pytest.fixture
def churning_target(tmp_path: Path) -> Iterator[subprocess.Popen[bytes]]:
    """A target that continuously forks and reaps short-lived children.

    Reproduces the real "signal delivered mid-call" scenario that motivates
    `gdb_capture`'s retry loop (originally observed with churning DataLoader
    workers) without needing torch/Lightning.
    """
    script = tmp_path / "churn.py"
    script.write_text(
        _ALLOW_ANY_TRACER + "import multiprocessing, time\n"
        "def _short():\n"
        "    time.sleep(0.02)\n"
        "if __name__ == '__main__':\n"
        "    while True:\n"
        "        p = multiprocessing.Process(target=_short)\n"
        "        p.start()\n"
        "        p.join()\n"
    )
    proc = subprocess.Popen([sys.executable, str(script)])  # noqa: S603 -- fixed, non-shell argv; test-only target
    time.sleep(0.5)
    try:
        yield proc
    finally:
        proc.kill()
        proc.wait(timeout=5)


def _no_gdb_processes_targeting(pid: int) -> bool:
    result = subprocess.run(
        ["pgrep", "-f", f"gdb.*-p {pid}"], capture_output=True, text=True, check=False
    )  # noqa: S603, S607 -- fixed argv, test-only assertion
    return result.stdout.strip() == ""


def test_check_gdb_available_ok() -> None:
    check_gdb_available()


def test_check_gdb_available_missing_binary_raises() -> None:
    with pytest.raises(GdbUnavailableError):
        check_gdb_available("definitely-not-a-real-gdb-binary")


@_attach_permitted
def test_capture_reads_live_variable(target_process: subprocess.Popen[bytes]) -> None:
    snippet = (
        "import sys, os, json\n"
        "m = sys.modules['__main__']\n"
        "report = {'pid': os.getpid(), 'counter': getattr(m, 'counter', None)}\n"
        "with open({out_path}, 'w') as f:\n"
        "    json.dump(report, f)\n"
    )
    result = gdb_capture(target_process.pid, snippet, poll_timeout_s=15)
    assert result["pid"] == target_process.pid
    assert isinstance(result["counter"], int)


@_attach_permitted
def test_capture_snippet_with_literal_braces_survives_placeholder_substitution(
    target_process: subprocess.Popen[bytes],
) -> None:
    snippet = (
        "import json\n"
        "report = {'a': {'nested': 1}, 'b': [1, 2, 3]}\n"
        "with open({out_path}, 'w') as f:\n"
        "    json.dump(report, f)\n"
    )
    result = gdb_capture(target_process.pid, snippet, poll_timeout_s=15)
    assert result == {"a": {"nested": 1}, "b": [1, 2, 3]}


@_attach_permitted
def test_capture_survives_fork_exit_churn(
    churning_target: subprocess.Popen[bytes],
) -> None:
    snippet = "import json\nwith open({out_path}, 'w') as f:\n    json.dump({'ok': True}, f)\n"
    result = gdb_capture(
        churning_target.pid, snippet, poll_timeout_s=20, max_attempts=30
    )
    assert result == {"ok": True}
    assert churning_target.poll() is None


@_attach_permitted
def test_no_dangling_gdb_process_after_capture(
    target_process: subprocess.Popen[bytes],
) -> None:
    snippet = "import json\nwith open({out_path}, 'w') as f:\n    json.dump({'ok': True}, f)\n"
    gdb_capture(target_process.pid, snippet, poll_timeout_s=15)
    time.sleep(0.5)
    assert _no_gdb_processes_targeting(target_process.pid)


@pytest.mark.skipif(
    _YAMA_SCOPE == 0 or _IS_ROOT, reason="the kernel permits this attach"
)
def test_capture_refused_attach_names_the_cause() -> None:
    proc = subprocess.Popen(  # noqa: S603 -- fixed, non-shell argv; test-only target process
        [sys.executable, "-c", "import time\nwhile True:\n    time.sleep(0.01)\n"]
    )
    time.sleep(0.3)
    try:
        with pytest.raises(
            GdbAttachError, match=r"could not attach to pid \d+.*ptrace"
        ):
            gdb_capture(proc.pid, "pass", poll_timeout_s=5)
    finally:
        proc.kill()
        proc.wait(timeout=5)


class _FakeTrainctlHandler(http.server.BaseHTTPRequestHandler):
    """Serves canned /commands/{id} and /holds responses for the refusal-path test."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 -- overrides BaseHTTPRequestHandler's own parameter name
        pass

    def do_GET(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler's required method name
        if self.path.startswith("/commands/"):
            body = json.dumps({"status": "queued"}).encode()
        elif self.path == "/holds":
            body = json.dumps({"holds": [], "active": None}).encode()
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def fake_trainctl_server() -> Iterator[str]:
    server = http.server.HTTPServer(("127.0.0.1", 0), _FakeTrainctlHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_capture_refuses_without_active_hold(
    target_process: subprocess.Popen[bytes], fake_trainctl_server: str
) -> None:
    client = HoldClient(fake_trainctl_server)
    with pytest.raises(HoldNotActiveError):
        gdb_capture(
            target_process.pid, "pass", hold_client=client, hold_command_id="cmd_x"
        )
