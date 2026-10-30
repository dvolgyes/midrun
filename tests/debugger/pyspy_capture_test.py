"""Behavioral tests for the py-spy capture mode.

The specific claim that repeated dumps never stall a live CUDA target running
real `.item()`/`backward()` calls was validated by a manual live run against a
real GPU training process this session (py-spy structurally never resumes the
target through injected code, unlike `gdb_capture`); that scenario needs a real
GPU and a real Lightning training loop, so it stays a manual/live check rather
than a CI-run pytest case. This file covers the invariant that CI *can* check
without a GPU: py-spy's read-only dump never stalls or kills a plain target
process, for both a busy-looping and an idle target.
"""

import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from midrun.debugger import pyspy_capture

pytestmark = pytest.mark.skipif(
    shutil.which("py-spy") is None, reason="py-spy not installed"
)

_PTRACE_SCOPE = Path("/proc/sys/kernel/yama/ptrace_scope")
_YAMA_SCOPE = int(_PTRACE_SCOPE.read_text()) if _PTRACE_SCOPE.exists() else 0
# Yama scope 1 lets py-spy attach to a sibling only when the target opts in with
# prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY), as the target below does; scope 2 and
# 3 ignore that opt-in.
_needs_ptrace = pytest.mark.skipif(
    _YAMA_SCOPE >= 2 and os.geteuid() != 0,
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
            _ALLOW_ANY_TRACER + "import time\nwhile True:\n    time.sleep(0.01)\n",
        ]
    )
    time.sleep(0.3)
    try:
        yield proc
    finally:
        proc.kill()
        proc.wait(timeout=5)


@_needs_ptrace
def test_capture_returns_pid_and_thread_frames(
    target_process: subprocess.Popen[bytes],
) -> None:
    result = pyspy_capture.capture(target_process.pid)
    assert result["pid"] == target_process.pid
    assert len(result["threads"]) >= 1
    assert "frames" in result["threads"][0]


@_needs_ptrace
def test_capture_with_locals_includes_locals_field(
    target_process: subprocess.Popen[bytes],
) -> None:
    result = pyspy_capture.capture(target_process.pid, locals_verbosity=1)
    frame = result["threads"][0]["frames"][0]
    assert "locals" in frame


@_needs_ptrace
def test_repeated_captures_never_stall_or_kill_target(
    target_process: subprocess.Popen[bytes],
) -> None:
    for _ in range(5):
        pyspy_capture.capture(target_process.pid, locals_verbosity=1)
    assert target_process.poll() is None


@pytest.fixture
def target_with_children() -> Iterator[subprocess.Popen[bytes]]:
    python_child = (
        _ALLOW_ANY_TRACER + "import time\nwhile True:\n    time.sleep(0.01)\n"
    )
    code = (
        _ALLOW_ANY_TRACER
        + "import subprocess, sys, time\n"
        + f"subprocess.Popen([sys.executable, '-c', {python_child!r}])\n"
        + "subprocess.Popen(['sleep', '60'])\n"
        + "while True:\n    time.sleep(0.01)\n"
    )
    proc = subprocess.Popen(  # noqa: S603 -- fixed, non-shell argv; test-only target process
        [sys.executable, "-c", code], start_new_session=True
    )
    children = Path(f"/proc/{proc.pid}/task/{proc.pid}/children")
    deadline = time.monotonic() + 10
    while len(children.read_text().split()) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(0.3)
    try:
        yield proc
    finally:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


@_needs_ptrace
def test_capture_with_subprocesses_dumps_every_python_descendant(
    target_with_children: subprocess.Popen[bytes],
) -> None:
    children = [
        int(pid)
        for pid in Path(
            f"/proc/{target_with_children.pid}/task/{target_with_children.pid}/children"
        )
        .read_text()
        .split()
    ]
    comms = {pid: Path(f"/proc/{pid}/comm").read_text().strip() for pid in children}
    sleep_pid = next(pid for pid, comm in comms.items() if comm == "sleep")
    python_pid = next(pid for pid in children if pid != sleep_pid)

    result = pyspy_capture.capture(target_with_children.pid, subprocesses=True)

    assert result["pid"] == target_with_children.pid
    assert {thread["pid"] for thread in result["threads"]} == {
        target_with_children.pid,
        python_pid,
    }
    assert [failure["pid"] for failure in result["failed_subprocesses"]] == [sleep_pid]
    assert result["failed_subprocesses"][0]["error"]


@_needs_ptrace
def test_capture_without_subprocesses_dumps_only_the_target(
    target_with_children: subprocess.Popen[bytes],
) -> None:
    result = pyspy_capture.capture(target_with_children.pid)

    assert {thread["pid"] for thread in result["threads"]} == {target_with_children.pid}
    assert "failed_subprocesses" not in result


def test_capture_unknown_pid_raises_pyspy_error() -> None:
    with pytest.raises(pyspy_capture.PySpyError):
        pyspy_capture.capture(1)


def test_capture_hung_py_spy_raises_pyspy_error(tmp_path: Path) -> None:
    hung_py_spy = tmp_path / "py-spy"
    hung_py_spy.write_text("#!/bin/sh\nexec sleep 120\n")
    hung_py_spy.chmod(0o755)
    with pytest.raises(pyspy_capture.PySpyError, match="timed out"):
        pyspy_capture.capture(1, py_spy_bin=str(hung_py_spy))
