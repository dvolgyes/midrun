"""Behavioral tests for midrun.debugger.artifact_push."""

import json
from typing import Any

import pytest

from midrun.debugger.artifact_push import push_capture


class _RecordingHoldClient:
    def __init__(self) -> None:
        self.pushed: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def push_capture(
        self, kind: str, files: dict[str, str], metadata: dict[str, Any]
    ) -> dict[str, Any]:
        self.pushed.append((kind, files, metadata))
        return {"id": "dbg-1"}


def _reject_constant(name: str) -> None:
    raise ValueError(f"non-standard JSON constant {name}")


def test_non_finite_floats_in_a_result_become_null() -> None:
    client = _RecordingHoldClient()
    result = {"loss": float("nan"), "stats": [float("inf"), -float("inf"), 1.5]}

    manifest = push_capture(client, "pyspy-stack", result)

    assert manifest == {"id": "dbg-1"}
    [(kind, files, metadata)] = client.pushed
    assert kind == "pyspy-stack"
    assert metadata == {}
    assert isinstance(files["result.json"], str)
    assert json.loads(files["result.json"], parse_constant=_reject_constant) == {
        "loss": None,
        "stats": [None, None, 1.5],
    }


def test_a_value_without_a_json_form_is_written_as_its_string() -> None:
    class _Opaque:
        def __str__(self) -> str:
            return "opaque"

    client = _RecordingHoldClient()

    push_capture(client, "gdb-snapshot", {"frame": _Opaque()}, extra_metadata={"a": 1})

    [(_kind, files, metadata)] = client.pushed
    assert json.loads(files["result.json"]) == {"frame": "opaque"}
    assert metadata == {"a": 1}


@pytest.mark.parametrize("value", ["plain", 3, None, {"nested": [1, 2]}])
def test_finite_results_round_trip(value: object) -> None:
    client = _RecordingHoldClient()

    push_capture(client, "kind", {"value": value})

    [(_kind, files, _metadata)] = client.pushed
    assert json.loads(files["result.json"]) == {"value": value}
