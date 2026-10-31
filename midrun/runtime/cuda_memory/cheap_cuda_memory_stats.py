"""Reads the cheap CUDA allocator counters."""

from typing import Any

import torch


def cheap_cuda_memory_stats(device: str) -> dict[str, Any]:
    """Returns cheap, always-safe-to-read CUDA allocator counters of `device`.

    Returns `{"available": False}` when `device` is not a CUDA device, or when this
    process has not initialized CUDA (no CUDA device, or the run has not used one
    yet). Reading the counters would otherwise create a CUDA context on the calling
    FUSE or REST thread.

    Args:
        device: The rank's training device, e.g. `cuda:1`. The calling thread's
            current device can be another GPU.
    """
    target = torch.device(device)
    # torch's is_initialized has no annotations.
    initialized: bool = torch.cuda.is_initialized()  # type: ignore[no-untyped-call]
    if target.type != "cuda" or not initialized:
        return {"available": False}
    free_bytes, total_bytes = torch.cuda.mem_get_info(target)
    return {
        "available": True,
        "memory_allocated": torch.cuda.memory_allocated(target),
        "memory_reserved": torch.cuda.memory_reserved(target),
        "max_memory_allocated": torch.cuda.max_memory_allocated(target),
        "max_memory_reserved": torch.cuda.max_memory_reserved(target),
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
    }
