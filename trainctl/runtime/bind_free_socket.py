"""Binds a listening socket on the first free port of a range."""

import socket

from trainctl.runtime.bind_socket import bind_socket


def bind_free_socket(
    host: str, start: int, search_range: int
) -> tuple[socket.socket, int]:
    """Binds and holds a listening socket on the first free candidate port in
    `[start, start + search_range)`, tried sequentially. See `bind_socket`.
    """
    for candidate in range(start, start + search_range):
        try:
            return bind_socket(host, candidate), candidate
        except OSError:
            continue
    raise RuntimeError(f"no free REST port in [{start}, {start + search_range})")
