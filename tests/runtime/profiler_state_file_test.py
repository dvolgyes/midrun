"""`/state/profiler.json` reads the profiler run once, so a run that ends mid-read is safe."""

import json

from midrun.runtime.virtual_fs.static_files import STATIC_FILES


class _Run:
    def phase_state(self) -> dict[str, str]:
        return {"state": "active"}


class _RunEndsAfterFirstRead:
    """The training thread clears `profiler_run` right after the first read."""

    def __init__(self) -> None:
        self._run: _Run | None = _Run()

    @property
    def profiler_run(self) -> _Run | None:
        run, self._run = self._run, None
        return run


def test_profiler_state_file_survives_a_run_that_ends_during_the_read() -> None:
    content = STATIC_FILES["/state/profiler.json"](_RunEndsAfterFirstRead())

    assert json.loads(content) == {"state": "active"}
