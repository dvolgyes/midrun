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
        return tmp_path / "midrun" / model._midrun.run_id

    return resolve
