"""Encodes the JSON content of a virtual file."""

import json
from typing import Any


def encode_json(value: Any) -> bytes:
    """Returns `value` as indented JSON bytes; `str()` encodes the other types."""
    return (json.dumps(value, indent=2, default=str) + "\n").encode()
