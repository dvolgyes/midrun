"""Behavioral tests for the run-log file that start_logging_session attaches."""

from pathlib import Path

from midrun.config import MidrunConfig
from midrun.logging.logging_session import LoggingSession
from midrun.runtime.start_logging_session import start_logging_session


def test_shared_run_log_names_the_global_rank_of_each_record(tmp_path: Path) -> None:
    config = MidrunConfig(intercept_lightning_logging=False)
    session = LoggingSession("session-rank-three")
    start_logging_session(session, config, tmp_path, 3)
    try:
        session.bind().info("written by rank three")
    finally:
        session.close()

    (line,) = (tmp_path / "midrun.log").read_text().splitlines()
    assert " | rank 3 | " in line
    assert line.endswith("written by rank three")
