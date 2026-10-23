"""The thread-safe store of the current runtime snapshot."""

import threading
import time
from dataclasses import replace
from typing import Any, cast

from trainctl.runtime.state.runtime_snapshot import RuntimeSnapshot


class StateStore:
    """Publishes and reads `RuntimeSnapshot` instances under a small lock.

    Attributes:
        None: State is private; use `read`/`publish`/`update`.
    """

    def __init__(self, snapshot: RuntimeSnapshot) -> None:
        self._lock = threading.Lock()
        self._snapshot = snapshot

    def read(self) -> RuntimeSnapshot:
        """Returns the current snapshot."""
        with self._lock:
            return self._snapshot

    def publish(self, snapshot: RuntimeSnapshot) -> None:
        """Replaces the current snapshot outright."""
        with self._lock:
            self._snapshot = snapshot

    def update(self, **changes: object) -> RuntimeSnapshot:
        """Publishes a new snapshot derived from the current one.

        Args:
            **changes: Field overrides applied via `dataclasses.replace`.

        Returns:
            The newly published snapshot.
        """
        with self._lock:
            current = self._snapshot
            new = replace(
                current,
                generation=current.generation + 1,
                updated_at=time.time(),
                # The keys are not typed per field; `replace` rejects an unknown field.
                **cast(dict[str, Any], changes),
            )
            self._snapshot = new
            return new
