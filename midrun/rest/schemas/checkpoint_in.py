"""The body of `POST /checkpoints`."""

from typing import Any

from pydantic import BaseModel, Field

from midrun.runtime.artifacts import ARTIFACT_NAME_PATTERN


class CheckpointIn(BaseModel):
    """Body for `POST /checkpoints`."""

    when: Any = "now"
    label: str = Field(default="manual", pattern=ARTIFACT_NAME_PATTERN)
