"""Behavioral tests: each Trainctl runtime gets a run id that no other process shares."""

import re
import subprocess
import sys

from trainctl.config import TrainctlConfig
from trainctl.lightning_backend import load_backend
from trainctl.runtime.runtime import TrainctlRuntime

_PRINT_RUN_ID = """
from trainctl.config import TrainctlConfig
from trainctl.lightning_backend import load_backend
from trainctl.runtime.runtime import TrainctlRuntime
print(TrainctlRuntime(TrainctlConfig(), load_backend("lightning.pytorch")).run_id)
"""


def _run_id_in_new_process() -> str:
    completed = subprocess.run(  # noqa: S603 -- fixed interpreter and script
        [sys.executable, "-c", _PRINT_RUN_ID],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return completed.stdout.strip()


def test_run_ids_differ_across_processes() -> None:
    local = TrainctlRuntime(TrainctlConfig(), load_backend("lightning.pytorch")).run_id

    run_ids = {local, _run_id_in_new_process(), _run_id_in_new_process()}

    assert len(run_ids) == 3
    # The run id names the fallback run directory, so it must be one safe name.
    assert all(re.fullmatch(r"[A-Za-z0-9._-]+", run_id) for run_id in run_ids)
