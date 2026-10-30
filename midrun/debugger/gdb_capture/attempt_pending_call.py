"""Makes one gdb attach, pending-call and detach attempt."""

from pathlib import Path

from pygdbmi.gdbcontroller import GdbController

from midrun.debugger.gdb_capture.evaluate_mi_expression import (
    evaluate_mi_expression,
)
from midrun.debugger.gdb_capture.gdb_attach_error import GdbAttachError
from midrun.debugger.gdb_capture.require_attached import require_attached


def attempt_pending_call(pid: int, snippet_path: Path, gdb_bin: str) -> None:
    """One attach-call-detach attempt; see `schedule_pending_call` for why this can need retries."""
    gdbmi = GdbController(
        command=[gdb_bin, "--nx", "--quiet", "--interpreter=mi3", "-p", str(pid)]
    )
    try:
        startup_records = gdbmi.get_gdb_response(timeout_sec=5)
        require_attached(gdbmi, pid, startup_records)
        gdbmi.write("-gdb-set scheduler-locking on", timeout_sec=5)
        # The call can run after the caller has timed out and deleted the
        # snippet file; a failed open must not raise into the target.
        c_call = (
            "(int) Py_AddPendingCall("
            "(int (*)(void *)) PyRun_SimpleString, "
            "(void *) \\\"with __import__('contextlib').suppress(OSError): "
            f"exec(open('{snippet_path}').read(), {{}})\\\")"
        )
        rc = evaluate_mi_expression(gdbmi, c_call)
        if rc != "0":
            raise GdbAttachError(f"Py_AddPendingCall failed to enqueue (rc={rc})")
        gdbmi.write("-target-detach", timeout_sec=5)
    finally:
        gdbmi.exit()
