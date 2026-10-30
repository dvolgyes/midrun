"""Behavioral tests for `descendant_pids`."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from midrun.debugger.descendant_pids import descendant_pids


def _spawn_then_sleep(child_code: str | None) -> str:
    spawn = (
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        if child_code is not None
        else ""
    )
    return f"import subprocess, sys, time\n{spawn}time.sleep(60)\n"


def _parent_pid(pid: int) -> int:
    # Field 4 of /proc/<pid>/stat, after the parenthesized command name.
    stat = Path(f"/proc/{pid}/stat").read_text()
    return int(stat.rsplit(")", 1)[1].split()[1])


def test_descendant_pids_finds_children_and_grandchildren() -> None:
    code = _spawn_then_sleep(_spawn_then_sleep(_spawn_then_sleep(None)))
    proc = subprocess.Popen(  # noqa: S603 -- fixed argv; test-only process tree
        [sys.executable, "-c", code], start_new_session=True
    )
    try:
        deadline = time.monotonic() + 10
        found = descendant_pids(proc.pid)
        while len(found) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
            found = descendant_pids(proc.pid)

        assert len(found) == 2
        child, grandchild = found
        assert _parent_pid(child) == proc.pid
        assert _parent_pid(grandchild) == child
    finally:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


def test_descendant_pids_of_a_missing_process_is_empty() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])  # noqa: S603 -- fixed argv
    proc.wait(timeout=10)
    assert descendant_pids(proc.pid) == []
