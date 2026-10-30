"""Starts the Lightning logging bridge and the run-log file of a logging session."""

from pathlib import Path

from midrun.config import MidrunConfig
from midrun.logging.logging_session import LoggingSession


def start_logging_session(
    session: LoggingSession, config: MidrunConfig, base_dir: Path, rank: int
) -> None:
    """Installs the bridge and attaches the run-log file that `config` enables.

    The run-log file is `midrun-rank-<rank>.log` under the `all` hooks rank policy,
    else `midrun.log`. The rank policy selects the name also when hooks are disabled.
    Each line of the file names `rank`, the global rank of this process.
    """
    if config.intercept_lightning_logging:
        session.install()
    if config.run_log_enabled:
        filename = (
            f"midrun-rank-{rank}.log"
            if config.hooks_rank_policy == "all"
            else "midrun.log"
        )
        session.attach_run_file(base_dir / filename, rank=rank)
