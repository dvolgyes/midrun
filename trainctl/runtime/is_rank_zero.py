"""Checks whether this process is global rank 0."""

from trainctl.runtime.distributed_initialized import distributed_initialized


def is_rank_zero() -> bool:
    """Returns whether this process is global rank 0; `True` without a process group."""
    import torch.distributed as dist

    if distributed_initialized():
        rank: int = dist.get_rank()
        return rank == 0
    return True
