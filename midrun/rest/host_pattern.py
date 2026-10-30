"""Converts a host name or IP address to the form that a Host header carries."""

import ipaddress
import re

_HOST_NAME = re.compile(r"[a-z0-9_-]+(?:\.[a-z0-9_-]+)*", re.IGNORECASE)


def host_pattern(host: str) -> str | None:
    """Returns `host` as Starlette's `TrustedHostMiddleware` matches it, or None.

    The middleware compares the Host header without its port, case-insensitively,
    and keeps the brackets of an IPv6 literal. So an IPv6 address becomes
    `[compressed-form]`, which is also the form that browsers send. A bracketed
    IPv6 literal is accepted as input too.

    Returns:
        The lowercase pattern, or None when `host` is not a bare host name or IP
        address: for example, when it has a scheme, a port, a path, a user, a
        wildcard, whitespace, or an IPv6 zone index.
    """
    bracketed = host.startswith("[") and host.endswith("]")
    candidate = host[1:-1] if bracketed else host
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        if bracketed or _HOST_NAME.fullmatch(host) is None:
            return None
        return host.lower()
    if isinstance(address, ipaddress.IPv4Address):
        return None if bracketed else str(address)
    if address.scope_id is not None:
        return None
    return f"[{address.compressed}]"
