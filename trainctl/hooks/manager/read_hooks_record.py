"""Reads a hooks marker or provenance record, if it is compatible."""

import json
from pathlib import Path
from typing import Any

from trainctl.hooks.manager.hooks_tree_layout import SCHEMA_VERSION


def read_hooks_record(path: Path) -> dict[str, Any] | None:
    """Returns the record in `path`, or `None` if it is missing, unreadable or incompatible."""
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not (
        isinstance(record, dict)
        and record.get("schema_version") == SCHEMA_VERSION
        and isinstance(record.get("session_id"), str)
    ):
        return None
    return record
