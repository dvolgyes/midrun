"""Reads the cheap CUDA allocator counters."""

from typing import Any

import torch


def cheap_cuda_memory_stats() -> dict[str, Any]:
    """Returns cheap, always-safe-to-read CUDA allocator counters.

    Returns `{"available": False}` when no CUDA device is present.
    """
    if not torch.cuda.is_available():
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
