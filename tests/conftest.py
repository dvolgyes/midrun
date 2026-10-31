"""Fixtures shared by the Midrun test packages."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def run_base_dir(tmp_path: Path) -> Callable[[Any], Path]:
    """Returns the run directory of a model fit with `logger=False` under `tmp_path`.

    The Trainer must use `default_root_dir=tmp_path`. The directory is known as soon as
    the model exists, because it depends only on the model's runtime run id.
    """

    def resolve(model: Any) -> Path:
        return tmp_path / "midrun" / model.midrun.run_id

    return resolve


@pytest.fixture(autouse=True)
def steady_cpu_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lets the features that need a steady CPU clock run on any test host.

    The check itself is tested in `tests/runtime/require_steady_cpu_clock_test.py`;
    a test of a refusal replaces the check again.
    """
    for module in (
        "midrun.runtime.handlers.handle_profile",
        "midrun.runtime.handlers.handle_set_lightning_profiler",
        "midrun.runtime.cuda_memory.set_cuda_memory_history",
    ):
        monkeypatch.setattr(f"{module}.require_steady_cpu_clock", lambda feature: None)
