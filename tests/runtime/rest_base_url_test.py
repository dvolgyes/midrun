"""Behavioral tests for rest_base_url, the URL that reaches a bound REST server."""

import pytest

from midrun.runtime.rest_base_url import rest_base_url


@pytest.mark.parametrize(
    ("bind_host", "url"),
    [
        ("127.0.0.1", "http://127.0.0.1:8090"),
        ("0.0.0.0", "http://127.0.0.1:8090"),  # noqa: S104
        ("", "http://127.0.0.1:8090"),
        ("::", "http://[::1]:8090"),
        ("::1", "http://[::1]:8090"),
        ("fd00::5", "http://[fd00::5]:8090"),
        ("gpu7.lab.example", "http://gpu7.lab.example:8090"),
    ],
)
def test_rest_base_url_reaches_the_bound_server(bind_host: str, url: str) -> None:
    assert rest_base_url(bind_host, 8090) == url
