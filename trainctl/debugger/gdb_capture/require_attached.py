"""Checks that a gdb session is attached to its target process."""

import time
from typing import Any

from pygdbmi.gdbcontroller import GdbController

from trainctl.debugger.gdb_capture.gdb_attach_error import GdbAttachError

_TIMEOUT_SEC = 10.0


def require_attached(
    gdbmi: GdbController, pid: int, startup_records: list[dict[str, Any]]
) -> None:
    """Raises unless gdb is attached to `pid`.

    gdb keeps running when `-p` cannot attach, for example when
    `kernel.yama.ptrace_scope` refuses the ptrace. Without this check, the first
    expression then fails with an unrelated "No symbol table" error.

    pygdbmi returns the records that arrive before gdb pauses its output, so the
    startup output can come before the `-list-thread-groups` result. This reads
    until the result arrives or `_TIMEOUT_SEC` passes.

    Raises:
        GdbAttachError: gdb is not attached to `pid`. The message holds gdb's own
            output, which names the cause.
    """
    records = startup_records + gdbmi.write(
        "-list-thread-groups", timeout_sec=_TIMEOUT_SEC
    )
    deadline = time.monotonic() + _TIMEOUT_SEC
    results = [record for record in records if record["type"] == "result"]
    while not results and time.monotonic() < deadline:
        new_records = gdbmi.get_gdb_response(
            timeout_sec=1, raise_error_on_timeout=False
        )
        records += new_records
        results = [record for record in new_records if record["type"] == "result"]
    pids = (
        {group.get("pid") for group in (results[0]["payload"] or {}).get("groups", [])}
        if results
        else set()
    )
    if str(pid) in pids:
        return
    gdb_output = " ".join(
        str(record["payload"])
        for record in records
        if record["type"] in ("console", "log") and record["payload"]
    )
    gdb_output = " ".join(gdb_output.split())
    raise GdbAttachError(f"gdb could not attach to pid {pid}: {gdb_output}")
