"""Validates the `rest_allowed_hosts` option."""

from collections.abc import Iterable

from midrun.rest.host_pattern import host_pattern


def check_allowed_hosts(entries: Iterable[object]) -> tuple[str, ...]:
    """Returns `entries` as a tuple after it checks that each is a bare host.

    Raises:
        ValueError: An entry is not a string, or is not a host name or IP
            address that a Host header can carry (see `host_pattern`).
    """
    checked: list[str] = []
    for entry in entries:
        if not isinstance(entry, str) or host_pattern(entry) is None:
            raise ValueError(
                f"rest_allowed_hosts entry {entry!r} must be a host name or IP"
                " address without scheme, port, path, wildcard or whitespace"
            )
        checked.append(entry)
    return tuple(checked)
