"""Short process-local ids for commands and other run objects."""

import itertools

_id_counter = itertools.count(1)


def next_id(prefix: str) -> str:
    """Generates a short, monotonically increasing, process-local id."""
    return f"{prefix}_{next(_id_counter):06d}"
