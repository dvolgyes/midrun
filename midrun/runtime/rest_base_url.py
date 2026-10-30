"""Builds the base URL that reaches a REST server on this machine."""

from midrun.rest.host_pattern import host_pattern


def rest_base_url(bind_host: str, port: int) -> str:
    """Returns the `http://host:port` URL that reaches a server bound to `bind_host`.

    A wildcard bind host (`0.0.0.0`, `::` or empty) becomes the loopback address
    of its family, because the REST API rejects a wildcard in the Host header.
    An IPv6 address gets brackets.
    """
    pattern = host_pattern(bind_host)
    if bind_host == "" or pattern == "0.0.0.0":  # noqa: S104 -- compared, never bound
        pattern = "127.0.0.1"
    elif pattern == "[::]":
        pattern = "[::1]"
    elif pattern is None:
        pattern = bind_host
    return f"http://{pattern}:{port}"
