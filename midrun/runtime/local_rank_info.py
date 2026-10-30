"""Describes this process as one rank of the run."""

import os
import socket

import torch.distributed as dist

from midrun.runtime.state.rank_info import RankInfo


def local_rank_info(device: str) -> RankInfo:
    """Returns this process's `RankInfo`, without DataLoader worker pids.

    Without an initialized process group, this process is global rank 0.
    """
    initialized = dist.is_available() and dist.is_initialized()
    return RankInfo(
        global_rank=dist.get_rank() if initialized else 0,
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        pid=os.getpid(),
        hostname=socket.gethostname(),
        device=device,
    )
