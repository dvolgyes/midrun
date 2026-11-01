"""Behavioral tests for `execution.when` validation at submission and its due-ness."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.rest.app import build_app
from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.commands.parse_when import parse_when
from midrun.runtime.events import SafePoint
from midrun.runtime.runtime import MidrunRuntime


def _runtime_and_client(tmp_path: Path) -> tuple[MidrunRuntime, TestClient]:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = build_runtime(config, load_backend("lightning.pytorch"))
    runtime.artifacts = ArtifactStore(tmp_path)
    return runtime, TestClient(build_app(runtime), base_url="http://localhost")


@pytest.mark.parametrize(
    "when",
    [
        {"safe_point": "train_batch_end", "occurrence": "3"},
        {"safe_point": "train_batch_end", "occurrence": 3},
        {"safe_point": "nowhere"},
        {"safe_point": ["train_batch_end"]},
        {"safe_point": "train_epoch_end", "epoch": None},
        {"safe_point": "train_epoch_end", "epoch": "5"},
        {"safe_point": "train_epoch_end", "epoch": True},
        {"safe_point": "train_epoch_end", "epoch": -1},
        {"global_step": [1]},
        {"global_step": 1.5},
        {"epoch": 1, "global_step": 1},
        {"safe_point": "train_batch_end", "epoch": 1, "occurrence": "next"},
        {"safe_point": "train_batch_end", "colour": "red"},
        {},
        "later",
        5,
    ],
)
def test_invalid_when_is_rejected_at_submission(tmp_path: Path, when: object) -> None:
    runtime, client = _runtime_and_client(tmp_path)

    response = client.post(
        "/commands", json={"kind": "snapshot", "execution": {"when": when}}
    )

    assert response.status_code == 422, response.text
    assert runtime.command_queue.list_all() == []


def test_queued_commands_never_raise_at_a_safe_point(tmp_path: Path) -> None:
    runtime, client = _runtime_and_client(tmp_path)
    accepted = client.post(
        "/commands",
        json={
            "kind": "snapshot",
            "execution": {"when": {"safe_point": "train_batch_end"}},
        },
    )
    assert accepted.status_code == 200

    due = runtime.command_queue.due(
        SafePoint.TRAIN_BATCH_END, epoch=0, global_step=0, allowed=None
    )

    assert [command.id for command in due] == [accepted.json()["id"]]


def test_epoch_without_safe_point_is_due_at_any_safe_point_of_that_epoch() -> None:
    when = parse_when({"epoch": 2})

    assert not when.is_due(SafePoint.TRAIN_BATCH_END, epoch=1, global_step=50)
    assert when.is_due(SafePoint.TRAIN_BATCH_END, epoch=2, global_step=51)


def test_global_step_with_safe_point_waits_for_that_safe_point() -> None:
    when = parse_when({"safe_point": "train_epoch_end", "global_step": 10})

    assert not when.is_due(SafePoint.TRAIN_BATCH_END, epoch=0, global_step=10)
    assert when.is_due(SafePoint.TRAIN_EPOCH_END, epoch=0, global_step=10)
    assert not when.is_due(SafePoint.TRAIN_EPOCH_END, epoch=0, global_step=9)


def test_next_safe_point_waits_outside_a_hold_but_now_runs_inside_one() -> None:
    next_safe_point = parse_when("next_safe_point")
    now = parse_when("now")

    # The hold loops check due-ness without a safe point.
    assert not next_safe_point.is_due(None, epoch=0, global_step=0)
    assert next_safe_point.is_due(SafePoint.TRAIN_BATCH_END, epoch=0, global_step=0)
    assert now.is_due(None, epoch=0, global_step=0)


@pytest.mark.parametrize(
    "when",
    [
        "now",
        "next_safe_point",
        {"safe_point": "train_batch_end"},
        {"epoch": 2},
        {"safe_point": "train_epoch_end", "epoch": 5},
        {"global_step": 607},
        {"safe_point": "validation_epoch_end", "global_step": 10},
    ],
)
def test_a_command_shows_its_when_in_the_shape_that_submission_accepts(
    tmp_path: Path, when: object
) -> None:
    _, client = _runtime_and_client(tmp_path)
    accepted = client.post(
        "/commands", json={"kind": "snapshot", "execution": {"when": when}}
    )
    assert accepted.status_code == 200, accepted.text

    shown = client.get(f"/commands/{accepted.json()['id']}").json()["when"]

    assert parse_when(shown) == parse_when(when)
