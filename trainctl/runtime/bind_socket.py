"""Binds a listening socket on one port."""

import socket


def bind_socket(host: str, port: int) -> socket.socket:
    """Binds and holds a listening socket on exactly `port`. Raises `OSError` if taken.

    Unlike a probe-then-release check, the socket is kept open and handed straight to
    uvicorn (`Server.run(sockets=[...])`), so there is no gap between "found free" and
    "actually listening" for a second concurrent caller to race into.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError:
        sock.close()
        raise
    sock.listen(2048)  # matches uvicorn.Config's own default backlog
    return sock
