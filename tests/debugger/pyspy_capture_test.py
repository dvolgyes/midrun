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
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from trainctl.debugger import pyspy_capture

pytestmark = pytest.mark.skipif(
    shutil.which("py-spy") is None, reason="py-spy not installed"
)

_PTRACE_SCOPE = Path("/proc/sys/kernel/yama/ptrace_scope")
# With a Yama scope above 0, py-spy can attach only to its own descendants (or as
# root), and the target here is a sibling of py-spy.
_needs_ptrace = pytest.mark.skipif(
    _PTRACE_SCOPE.exists()
    and _PTRACE_SCOPE.read_text().strip() != "0"
    and os.geteuid() != 0,
    reason="ptrace of a non-child is not permitted (kernel.yama.ptrace_scope > 0)",
)


@pytest.fixture
def target_process() -> Iterator[subprocess.Popen[bytes]]:
    proc = subprocess.Popen(  # noqa: S603 -- fixed, non-shell argv; test-only target process
        [sys.executable, "-c", "import time\nwhile True:\n    time.sleep(0.01)\n"]
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


def test_capture_unknown_pid_raises_pyspy_error() -> None:
    with pytest.raises(pyspy_capture.PySpyError):
        pyspy_capture.capture(1)
