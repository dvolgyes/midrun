"""Writes one path segment as an attribute or a mapping key."""

from collections.abc import MutableMapping


def set_field(obj: object, key: str, value: object) -> None:
    """Writes `obj[key] = value` when `obj` is a `MutableMapping` with that key,
    else `obj.<key> = value`, falling back to `obj[key] = value` for a
    `MutableMapping` `obj` lacking that attribute. Mirrors `get_field`'s
    key-first precedence.

    Raises:
        AttributeError: `key` is neither a settable attribute of `obj` nor, for a
            `MutableMapping` `obj`, an assignable key.
    """
    if isinstance(obj, MutableMapping) and key in obj:
        obj[key] = value
    elif hasattr(obj, key):
        setattr(obj, key, value)
    elif isinstance(obj, MutableMapping):
        obj[key] = value
    else:
        raise AttributeError(f"{key!r} is not an attribute or dict key of {obj!r}")
