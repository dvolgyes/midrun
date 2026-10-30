"""Encodes the JSON content of a virtual file."""

from typing import Any

import pydantic_core


def encode_json(value: Any) -> bytes:
    """Returns `value` as indented, standard JSON bytes, encoded as the REST API does.

    NaN and infinite floats encode as `null`, and dataclasses as objects. `str()`
    encodes the types that pydantic cannot serialize.
    """
    return (
        pydantic_core.to_json(value, indent=2, inf_nan_mode="null", fallback=str)
        + b"\n"
    )
