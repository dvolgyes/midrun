"""The allocator-history recording configuration of this process."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CudaMemoryHistoryState:
    """Current allocator-history recording configuration.

    Attributes:
        enabled: Whether `torch.cuda.memory._record_memory_history` is active.
        max_entries: Bound passed to `_record_memory_history`, if any.
        started_at: Unix timestamp history recording was last enabled, if any.
    """

    enabled: bool
    max_entries: int | None
    started_at: float | None
