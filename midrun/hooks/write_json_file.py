"""Writes a JSON document to a file."""

from pathlib import Path
from typing import Any

from midrun.hooks.encode_json_document import encode_json_document
from midrun.write_atomically import write_atomically


def write_json_file(path: Path, document: dict[str, Any]) -> Path:
    """Writes `document` as JSON to `path` atomically and returns `path`.

    A concurrent reader (e.g. another DDP rank reading the marker) sees either
    no file, the previous one, or the complete new one (see `write_atomically`).

    Raises:
        ValueError: `document` holds a NaN or infinite float, which standard JSON
            cannot represent; nothing is written.
    """
    # The encoded text is ASCII, so these bytes equal the text.
    write_atomically(path, encode_json_document(document).encode("ascii"))
    return path
