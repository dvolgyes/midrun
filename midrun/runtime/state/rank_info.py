"""What the run state reports about one rank."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RankInfo:
    """Per-rank process metadata: mostly gathered once after distributed init.

    :ivar global_rank: Rank across all processes.
    :ivar local_rank: Rank within the local node.
    :ivar pid: Process ID.
    :ivar hostname: Host the rank runs on.
    :ivar device: Selected compute device, e.g. `cuda:0` or `cpu`.
    :ivar worker_pids: Active multiprocessing children of this rank, not necessarily
        DataLoader workers. The publishing rank refreshes its own entry;
        remote entries retain the empty tuple gathered at setup.
    """

    global_rank: int
    local_rank: int
    pid: int
    hostname: str
    device: str
    worker_pids: tuple[int, ...] = ()
