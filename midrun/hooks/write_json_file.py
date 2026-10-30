"""Writes a JSON document to a file."""

import secrets
from pathlib import Path
from typing import Any

from midrun.hooks.encode_json_document import encode_json_document


def write_json_file(path: Path, document: dict[str, Any]) -> Path:
    """Writes `document` as JSON to `path` atomically and returns `path`.

    The JSON goes to a new temporary file in the same directory, which then
    replaces `path`, so a concurrent reader (e.g. another DDP rank reading the
    marker) sees either no file, the previous one, or the complete new one.

    Raises:
        ValueError: `document` holds a NaN or infinite float, which standard JSON
            cannot represent; nothing is written.
    """
    encoded = encode_json_document(document)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    replaced = False
    try:
        with temporary.open("x") as handle:
            handle.write(encoded)
        temporary.replace(path)
        replaced = True
    finally:
        if not replaced:
            temporary.unlink(missing_ok=True)
    return path
