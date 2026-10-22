"""Reads one path segment as an attribute or a mapping key."""

from collections.abc import Mapping
from typing import Any


def get_field(obj: Any, key: str) -> Any:
    """Reads `obj.<key>`, falling back to `obj[key]` when `obj` lacks that
    attribute but is a `Mapping` (e.g. a dict segment of a config tree).
    Attribute access is tried first, so a plain attribute path resolves exactly
    as it always has.

    Raises:
        AttributeError: `key` is neither an attribute of `obj` nor, for a
            `Mapping` `obj`, one of its keys.
    """
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, Mapping) and key in obj:
        return obj[key]
    raise AttributeError(f"{key!r} is not an attribute or dict key of {obj!r}")
