"""Writes a JSON document to a file."""

import json
from pathlib import Path
from typing import Any


def write_json_file(path: Path, document: dict[str, Any]) -> Path:
    """Writes `document` as JSON to `path` and returns `path`."""
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path
