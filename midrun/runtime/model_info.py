"""The static facts about the model that a fit start records."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelInfo:
    """The model's class, hparams and basic summary, as of the last fit start.

    FUSE serves these under `/model`; they do not change during a fit.

    Attributes:
        class_name: The model's class name, or None before the first fit start.
        hparams: The model's `hparams`.
        summary: The basic text summary of the model.
        summary_json: The basic summary of the model as a JSON document.
    """

    class_name: str | None = None
    hparams: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    summary_json: dict[str, Any] = field(default_factory=dict)
