"""Evaluates a C expression in a gdb-attached target through MI."""

from pygdbmi.gdbcontroller import GdbController

from trainctl.debugger.gdb_capture.gdb_attach_error import GdbAttachError


class GdbAbandonedEvaluationError(RuntimeError):
    """gdb abandoned an evaluation mid-call.

    Typically a forking/exiting DataLoader worker delivering a stray signal
    during the call. Retryable: stress-testing found this never corrupts or
    hangs the target, even after dozens of consecutive failures in one trial.
    """


def evaluate_mi_expression(
    gdbmi: GdbController, expr: str, timeout_sec: float = 10.0
) -> str:
    """Evaluates a C expression (including function calls) in the target via MI.

    Raises:
        GdbAbandonedEvaluationError: gdb returned `^done` with no `value` payload --
            a call abandoned mid-evaluation (signal collision), not a real error.
        GdbAttachError: gdb reported a genuine evaluation error, or no result
            record was produced at all.
    """
    for record in gdbmi.write(
        f'-data-evaluate-expression "{expr}"', timeout_sec=timeout_sec
    ):
        if record["type"] != "result":
            continue
        if record["message"] == "done":
            if not record["payload"] or "value" not in record["payload"]:
                raise GdbAbandonedEvaluationError(
                    f"call to {expr!r} was abandoned mid-evaluation (signal collision)"
                )
            return str(record["payload"]["value"])
        raise GdbAttachError(f"gdb error evaluating {expr!r}: {record['payload']}")
    raise GdbAttachError(f"no result record for {expr!r}")
