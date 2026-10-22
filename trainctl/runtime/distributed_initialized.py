"""Checks whether torch.distributed has a process group."""


def distributed_initialized() -> bool:
    """Returns whether torch.distributed is available and has a process group."""
    import torch.distributed as dist

    return bool(dist.is_available() and dist.is_initialized())
