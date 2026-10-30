"""Behavioral tests for trusted_hosts, the REST Host-header allowlist."""

import pytest

from midrun.rest.trusted_hosts import trusted_hosts


def test_loopback_names_are_always_trusted() -> None:
    hosts = trusted_hosts("127.0.0.1", (), {}, ())
    assert {"localhost", "127.0.0.1", "[::1]"} <= set(hosts)


@pytest.mark.parametrize("bind_host", ["0.0.0.0", "::", "", "0:0:0:0:0:0:0:0"])
def test_wildcard_bind_host_is_not_trusted(bind_host: str) -> None:
    hosts = trusted_hosts(bind_host, (), {}, ())
    assert set(hosts) == {"localhost", "127.0.0.1", "[::1]"}


def test_specific_bind_host_is_trusted_in_host_header_form() -> None:
    assert "10.1.2.3" in trusted_hosts("10.1.2.3", (), {}, ())
    assert "[fd00::1]" in trusted_hosts("FD00:0::1", (), {}, ())
    assert "box.example" in trusted_hosts("Box.Example", (), {}, ())


def test_machine_names_are_trusted() -> None:
    hosts = trusted_hosts("127.0.0.1", (), {}, ("gpu7", "gpu7.lab.example"))
    assert {"gpu7", "gpu7.lab.example"} <= set(hosts)


def test_ssh_connection_server_ip_is_trusted() -> None:
    environ = {"SSH_CONNECTION": "192.0.2.10 51234 198.51.100.7 22"}
    hosts = trusted_hosts("127.0.0.1", (), environ, ())
    assert "198.51.100.7" in hosts
    assert "192.0.2.10" not in hosts


def test_ipv6_ssh_connection_server_ip_is_bracketed() -> None:
    environ = {"SSH_CONNECTION": "2001:db8::10 51234 2001:db8::7 22"}
    assert "[2001:db8::7]" in trusted_hosts("127.0.0.1", (), environ, ())


@pytest.mark.parametrize(
    "value",
    ["", "garbage", "a b evil.example d", "1.2.3.4 1 5.6.7.8", "x y 999.1.1.1 z"],
)
def test_malformed_ssh_connection_adds_nothing(value: str) -> None:
    hosts = trusted_hosts("127.0.0.1", (), {"SSH_CONNECTION": value}, ())
    assert set(hosts) == {"localhost", "127.0.0.1", "[::1]"}


def test_extra_hosts_are_trusted() -> None:
    hosts = trusted_hosts("127.0.0.1", ("proxy.example", "::2"), {}, ())
    assert {"proxy.example", "[::2]"} <= set(hosts)


def test_invalid_machine_names_are_dropped() -> None:
    hosts = trusted_hosts("127.0.0.1", (), {}, ("bad name", "*.example", ""))
    assert set(hosts) == {"localhost", "127.0.0.1", "[::1]"}


@pytest.mark.parametrize(
    ("value", "pattern"),
    [
        ("203.0.113.9", "203.0.113.9"),
        (" 203.0.113.9\n", "203.0.113.9"),
        ("2001:DB8:0::9", "[2001:db8::9]"),
    ],
)
def test_midrun_whitelist_ip_is_trusted(value: str, pattern: str) -> None:
    hosts = trusted_hosts("127.0.0.1", (), {"MIDRUN_WHITELIST_IP": value}, ())
    assert pattern in hosts


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_midrun_whitelist_ip_adds_nothing(value: str) -> None:
    hosts = trusted_hosts("127.0.0.1", (), {"MIDRUN_WHITELIST_IP": value}, ())
    assert set(hosts) == {"localhost", "127.0.0.1", "[::1]"}


@pytest.mark.parametrize(
    "value",
    ["evil.example", "203.0.113.9:8090", "203.0.113.0/24", "999.1.1.1", "fe80::1%eth0"],
)
def test_invalid_midrun_whitelist_ip_is_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="MIDRUN_WHITELIST_IP"):
        trusted_hosts("127.0.0.1", (), {"MIDRUN_WHITELIST_IP": value}, ())
