"""Writes one path segment as an attribute or a mapping key."""

from collections.abc import MutableMapping
from typing import Any


def set_field(obj: Any, key: str, value: Any) -> None:
    """Writes `obj.<key> = value`, falling back to `obj[key] = value` for a
    `MutableMapping` `obj` lacking that attribute. Mirrors `get_field`'s
    attribute-first precedence.

    Raises:
        AttributeError: `key` is neither a settable attribute of `obj` nor, for a
            `MutableMapping` `obj`, an assignable key.
    """
    if hasattr(obj, key):
        setattr(obj, key, value)
    elif isinstance(obj, MutableMapping):
        obj[key] = value
    else:
        raise AttributeError(f"{key!r} is not an attribute or dict key of {obj!r}")
