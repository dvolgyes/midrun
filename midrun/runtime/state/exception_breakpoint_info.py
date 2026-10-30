"""What the run state reports about a held backward exception."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExceptionBreakpointInfo:
    """A backward exception currently holding the training thread.

    Diagnostic only: gradient state at this point is not a valid complete gradient
    set, and distributed collectives may be inconsistent across ranks (see
    `MidrunRuntime.handle_backward_exception`). Not a normal cooperative safe point.

    Attributes:
        exception_type: The intercepted exception's class name.
        exception_message: `str(exception)`.
        epoch: Epoch at the time of the exception.
        global_step: Global step at the time of the exception.
        batch_idx: Batch index at the time of the exception, when known.
        rank: Global rank the exception occurred on.
        started_at: Unix timestamp the breakpoint was entered.
    """

    exception_type: str
    exception_message: str
    epoch: int
    global_step: int
    batch_idx: int | None
    rank: int
    started_at: float
