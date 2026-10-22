"""Writes a hooks marker or provenance record."""

import json
from pathlib import Path
from typing import Any


def write_hooks_record(path: Path, record: dict[str, Any]) -> None:
    """Writes `record` to `path` as indented, key-sorted JSON."""
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
