"""Reads the cheap CUDA allocator counters."""

from typing import Any

import torch


def cheap_cuda_memory_stats() -> dict[str, Any]:
    """Returns cheap, always-safe-to-read CUDA allocator counters.

    Returns `{"available": False}` when this process has not initialized CUDA (no
    CUDA device, or the run has not used one yet). Reading the counters would
    otherwise create a CUDA context on the calling FUSE or REST thread.
    """
    if not torch.cuda.is_initialized():
        return {"available": False}
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "available": True,
        "memory_allocated": torch.cuda.memory_allocated(),
        "memory_reserved": torch.cuda.memory_reserved(),
        "max_memory_allocated": torch.cuda.max_memory_allocated(),
        "max_memory_reserved": torch.cuda.max_memory_reserved(),
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
    }
