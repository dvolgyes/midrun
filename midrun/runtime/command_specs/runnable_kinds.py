"""The command kinds that can run now, given the hold policy and the in-flight batch."""

from midrun.runtime.command_specs.command_specs import COMMAND_SPECS
from midrun.runtime.command_specs.needs_batch import NEEDS_BATCH


def runnable_kinds(
    allowed: frozenset[str] | None, batch_in_flight: bool
) -> frozenset[str] | None:
    """Narrows a hold policy to the kinds that can run now.

    An epoch end, or a hold that began at one, has no batch: the commands that read
    it wait for the next batch.

    Args:
        allowed: The kinds that the current hold policy runs; None allows every kind.
        batch_in_flight: Whether a batch is in flight.

    Returns:
        The kinds that can run now; None allows every kind.
    """
    if batch_in_flight:
        return allowed
    return (frozenset(COMMAND_SPECS) if allowed is None else allowed) - NEEDS_BATCH
