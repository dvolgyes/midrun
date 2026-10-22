"""What the run state reports about one rank."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RankInfo:
    """Per-rank process metadata: mostly gathered once after distributed init.

    Attributes:
        global_rank: Rank across all processes.
        local_rank: Rank within the local node.
        pid: Process ID.
        hostname: Host the rank runs on.
        device: Selected compute device, e.g. `cuda:0` or `cpu`.
        worker_pids: Live DataLoader worker process IDs. Unlike the other fields,
            this is refreshed on every publish rather than gathered once, since
            workers are spawned lazily and non-persistent ones churn per epoch.
    """

    global_rank: int
    local_rank: int
    pid: int
    hostname: str
    device: str
    worker_pids: tuple[int, ...] = ()
