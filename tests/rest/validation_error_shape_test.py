"""Behavioral tests for the answer to an invalid request argument.

The REST API answers every invalid argument with status 422 and the same detail
shape, whether the request schema or a check in the route rejects it.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.rest.app import build_app
from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.runtime import MidrunRuntime


def _client(tmp_path: Path) -> TestClient:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = MidrunRuntime(config, load_backend("lightning.pytorch"))
    runtime.artifacts = ArtifactStore(tmp_path)
    return TestClient(build_app(runtime), base_url="http://localhost")


@pytest.mark.parametrize(
    ("path", "body"),
    [
        # Rejected by the request schema.
        ("/checkpoints", {"label": "../../../escape"}),
        # Rejected by the command argument check.
        ("/commands", {"kind": "no_such_kind"}),
        ("/commands", {"kind": "set_learning_rate", "args": {"value": "1"}}),
        # Rejected by the execution time check.
        ("/commands", {"kind": "snapshot", "execution": {"when": "later"}}),
        ("/holds", {"when": "later"}),
        # Rejected by the artifact name check.
        ("/debug/captures", {"kind": "bad", "files": {"../x.txt": "y"}}),
    ],
)
def test_invalid_argument_answers_422_with_a_list_of_errors(
    tmp_path: Path, path: str, body: dict[str, object]
) -> None:
    response = _client(tmp_path).post(path, json=body)

    assert response.status_code == 422, response.text
    errors = response.json()["detail"]
    assert isinstance(errors, list)
    assert errors
    for error in errors:
        assert isinstance(error["loc"], list)
        assert error["loc"][0] == "body"
        assert isinstance(error["msg"], str)
        assert error["msg"]
        assert isinstance(error["type"], str)
