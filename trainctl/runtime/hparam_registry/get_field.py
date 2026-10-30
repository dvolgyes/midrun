"""Reads one path segment as an attribute or a mapping key."""

from collections.abc import Mapping
from typing import Any


def get_field(obj: Any, key: str) -> Any:
    """Reads `obj[key]` when `obj` is a `Mapping` (e.g. a dict segment of a config
    tree) with that key, else `obj.<key>`.
    The key is tried first, so a key such as `values` or `update` does not
    resolve to a dict method.

    Raises:
        AttributeError: `key` is neither an attribute of `obj` nor, for a
            `Mapping` `obj`, one of its keys.
    """
    if isinstance(obj, Mapping) and key in obj:
        return obj[key]
    if hasattr(obj, key):
        return getattr(obj, key)
    raise AttributeError(f"{key!r} is not an attribute or dict key of {obj!r}")
