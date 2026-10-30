"""The base of the command argument models."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class CommandArgsModel(BaseModel):
    """Base of the argument models: no extra keys, no coercion, finite floats only."""

    # strict: a JSON "false" stays a string and 7.9 stays a float, so neither is coerced.
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False
    )
