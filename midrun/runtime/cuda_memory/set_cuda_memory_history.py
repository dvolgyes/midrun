"""Turns bounded CUDA allocator-history recording on or off."""

import time
from typing import Any

import torch

from midrun.runtime.cuda_memory.cuda_memory_history_state import (
    CudaMemoryHistoryState,
)


def set_cuda_memory_history(
    enabled: bool, max_entries: int | None
) -> CudaMemoryHistoryState:
    """Enables or disables bounded CUDA allocator-history recording.

    Raises:
        RuntimeError: CUDA is not available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_memory history requires a CUDA device")
    if enabled:
        kwargs: dict[str, Any] = {"enabled": "all"}
        if max_entries is not None:
            kwargs["max_entries"] = max_entries
        torch.cuda.memory._record_memory_history(**kwargs)
        return CudaMemoryHistoryState(
            enabled=True, max_entries=max_entries, started_at=time.time()
        )
    torch.cuda.memory._record_memory_history(enabled=None)
    return CudaMemoryHistoryState(enabled=False, max_entries=None, started_at=None)
