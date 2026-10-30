"""Builds the Host-header allowlist that protects the REST API from DNS rebinding."""

import ipaddress
from collections.abc import Mapping, Sequence

from midrun.rest.host_pattern import host_pattern

_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")
_WILDCARD_PATTERNS = ("0.0.0.0", "[::]")  # noqa: S104 -- compared, never bound


def trusted_hosts(
    bind_host: str,
    extra_hosts: Sequence[str],
    environ: Mapping[str, str],
    machine_names: Sequence[str],
) -> list[str]:
    """Returns the Host-header values (without port) that the REST API accepts.

    A web page can rebind its own domain name to 127.0.0.1 and then read the
    REST API as a same-origin page. Its requests still carry that domain name
    in the Host header, so the API accepts only names that a user of this
    machine types: the loopback names, the bind host when it is not a
    wildcard, the machine's names, the local IP address of the current SSH
    connection, the `MIDRUN_WHITELIST_IP` address, and `extra_hosts`.

    Args:
        bind_host: The configured REST bind host.
        extra_hosts: Additional names, such as a reverse proxy's host name.
        environ: The process environment. `SSH_CONNECTION` (`client_ip
            client_port server_ip server_port`) supplies the server IP address;
            a malformed value is ignored. `MIDRUN_WHITELIST_IP`, when not empty
            after stripping, supplies one more IP address.
        machine_names: The machine's names, such as `socket.gethostname()` and
            `socket.getfqdn()`. A name that is not a valid host is skipped.

    Returns:
        The sorted, distinct patterns in the form that `host_pattern` gives.

    Raises:
        ValueError: `MIDRUN_WHITELIST_IP` is set but is not one IPv4 or IPv6
            address that a Host header can carry.
    """
    candidates = [*_LOOPBACK_HOSTS, *machine_names, *extra_hosts]
    bind_pattern = host_pattern(bind_host)
    if bind_pattern not in _WILDCARD_PATTERNS:
        candidates.append(bind_host)
    ssh_fields = environ.get("SSH_CONNECTION", "").split()
    if len(ssh_fields) == 4 and _is_ip_address(ssh_fields[2]):
        candidates.append(ssh_fields[2])
    whitelist_ip = environ.get("MIDRUN_WHITELIST_IP", "").strip()
    if whitelist_ip:
        if not _is_ip_address(whitelist_ip) or host_pattern(whitelist_ip) is None:
            raise ValueError(
                f"MIDRUN_WHITELIST_IP {whitelist_ip!r} must be one IPv4 or IPv6"
                " address without port, prefix length or zone index"
            )
        candidates.append(whitelist_ip)
    patterns = {host_pattern(candidate) for candidate in candidates}
    return sorted(pattern for pattern in patterns if pattern is not None)


def _is_ip_address(text: str) -> bool:
    try:
        ipaddress.ip_address(text)
    except ValueError:
        return False
    return True
