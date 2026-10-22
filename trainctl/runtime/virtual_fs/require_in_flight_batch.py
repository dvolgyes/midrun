"""Returns the batch that a live capture traces."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trainctl.runtime.runtime import TrainctlRuntime


def require_in_flight_batch(rt: "TrainctlRuntime", what: str) -> Any:
    """Returns the batch of the current step.

    Raises:
        ValueError: There is no in-flight batch. The message names `what`.
    """
    if rt.current_step.batch is None:
        raise ValueError(f"{what} requires an in-flight batch")
    return rt.current_step.batch
