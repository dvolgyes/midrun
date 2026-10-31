"""Behavioral tests for GET /files/{path} -- the REST mirror of the FUSE virtual tree.

`get_file` reuses `virtual_file_content`/`is_virtual_dir`/`list_virtual_dir`, the same functions
`MidrunFS` calls -- these tests check the REST-specific wiring (status codes, JSON
listing shape, error-boundary translation), not the virtual tree's content, which
`tests/fuse/*` already covers.
"""

import pytest
from fastapi.testclient import TestClient

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.rest.app import build_app
from midrun.runtime.model_info import ModelInfo
from midrun.runtime.runtime import MidrunRuntime
from midrun.runtime.virtual_fs.live_files import LIVE_FILES
from midrun.runtime.virtual_fs.virtual_file_content import virtual_file_content


def _client() -> tuple[TestClient, MidrunRuntime]:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = build_runtime(config, load_backend("lightning.pytorch"))
    return TestClient(build_app(runtime), base_url="http://localhost"), runtime


def test_get_file_matches_virtual_file_content() -> None:
    client, runtime = _client()
    runtime.model_info = ModelInfo(class_name="DemoModule")

    response = client.get("/files/model/class")

    assert response.status_code == 200
    assert response.text == virtual_file_content(runtime, "/model/class").decode()


def test_get_file_json_content_type_for_json_path() -> None:
    client, runtime = _client()

    response = client.get("/files/model/hparams.json")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")


def test_get_file_on_virtual_directory_returns_json_listing() -> None:
    client, _runtime = _client()

    response = client.get("/files/model")

    assert response.status_code == 200
    body = response.json()
    assert body["path"] == "/model"
    assert "class" in body["entries"]


def test_get_file_root_directory_returns_json_listing() -> None:
    client, _runtime = _client()

    response = client.get("/files/")

    assert response.status_code == 200
    assert response.json()["path"] == "/"


def test_get_file_unknown_path_is_404() -> None:
    client, _runtime = _client()

    response = client.get("/files/does/not/exist")

    assert response.status_code == 404


def test_get_file_live_view_timeout_maps_to_504(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _runtime = _client()

    def _raise_timeout(rt: object) -> bytes:
        raise TimeoutError("no safe point reached")

    monkeypatch.setitem(LIVE_FILES, "/model/summary-live.txt", _raise_timeout)

    response = client.get("/files/model/summary-live.txt")

    assert response.status_code == 504


def test_get_file_other_failure_maps_to_500(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _runtime = _client()

    def _raise_value_error(rt: object) -> bytes:
        raise ValueError("requires an in-flight batch")

    monkeypatch.setitem(LIVE_FILES, "/model/summary-live.txt", _raise_value_error)

    response = client.get("/files/model/summary-live.txt")

    assert response.status_code == 500


_BROWSER_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"


def test_get_file_directory_is_a_linked_page_for_a_browser() -> None:
    client, _runtime = _client()

    response = client.get("/files/", headers={"Accept": _BROWSER_ACCEPT})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["vary"] == "Accept"
    assert "default-src 'none'" in response.headers["content-security-policy"]
    assert '<a href="/files/model">model/</a>' in response.text
    page = client.get("/files/model", headers={"Accept": _BROWSER_ACCEPT}).text
    assert '<a href="/files/model/class">class</a>' in page
    assert '<a href="/files/">' in page


@pytest.mark.parametrize(
    "accept",
    [
        "*/*",
        "application/json",
        "text/html;q=0.5, application/json",
        "text/html;q=2",
        "text/html;q=abc",
    ],
)
def test_get_file_directory_stays_json_unless_html_is_preferred(accept: str) -> None:
    client, _runtime = _client()

    response = client.get("/files/model", headers={"Accept": accept})

    assert response.json()["path"] == "/model"
    assert response.headers["vary"] == "Accept"


def test_get_file_directory_page_escapes_entry_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _runtime = _client()
    monkeypatch.setitem(LIVE_FILES, '/model/<b>"x&y', lambda rt: b"")

    page = client.get("/files/model", headers={"Accept": _BROWSER_ACCEPT}).text

    assert '<a href="/files/model/%3Cb%3E%22x%26y">&lt;b&gt;&quot;x&amp;y</a>' in page
    assert "<b>" not in page
