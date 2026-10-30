"""Turns bounded CUDA allocator-history recording on or off."""

import time

import torch

from midrun.runtime.cuda_memory.cuda_memory_history_state import (
    CudaMemoryHistoryState,
)

DEFAULT_MAX_ENTRIES = 100_000


def set_cuda_memory_history(
    enabled: bool, max_entries: int | None
) -> CudaMemoryHistoryState:
    """Enables or disables bounded CUDA allocator-history recording.

    A None `max_entries` records at most `DEFAULT_MAX_ENTRIES` entries: torch's
    own default keeps every entry, so the history grows without bound.

    Raises:
        RuntimeError: CUDA is not available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_memory history requires a CUDA device")
    if enabled:
        if max_entries is None:
            max_entries = DEFAULT_MAX_ENTRIES
        torch.cuda.memory._record_memory_history(enabled="all", max_entries=max_entries)
        return CudaMemoryHistoryState(
            enabled=True, max_entries=max_entries, started_at=time.time()
        )
    torch.cuda.memory._record_memory_history(enabled=None)
    return CudaMemoryHistoryState(enabled=False, max_entries=None, started_at=None)
