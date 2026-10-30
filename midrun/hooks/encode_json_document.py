"""Encodes a JSON document exactly as the hooks facility writes it to a file."""

import json
from typing import Any


def encode_json_document(document: dict[str, Any]) -> str:
    """Returns the complete file text of `document`: indented, sorted, newline-ended.

    The text is ASCII (non-ASCII characters are escaped), so its length is also
    its size in bytes.

    Raises:
        ValueError: `document` holds a NaN or infinite float, which standard JSON
            cannot represent.
    """
    return json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
