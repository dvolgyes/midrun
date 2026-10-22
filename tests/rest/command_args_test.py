"""Behavioral tests for command argument checks at submission.

A command with an unknown kind or invalid args gets a 400 and is never queued, so the
training thread never runs it with a coerced value.
"""

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from trainctl.config import TrainctlConfig
from trainctl.load_backend import load_backend
from trainctl.rest.app import build_app
from trainctl.runtime import runtime
from trainctl.runtime.artifacts import ArtifactStore
from trainctl.runtime.command_args.command_args_by_kind import COMMAND_ARGS
from trainctl.runtime.commands.parse_when import parse_when
from trainctl.runtime.events import SafePoint
from trainctl.runtime.runtime import TrainctlRuntime


def _runtime_and_client(tmp_path: Path) -> tuple[TrainctlRuntime, TestClient]:
    config = TrainctlConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = TrainctlRuntime(config, load_backend("lightning.pytorch"))
    runtime.artifacts = ArtifactStore(tmp_path)
    return runtime, TestClient(build_app(runtime))


def test_every_accepted_kind_has_a_handler() -> None:
    assert set(COMMAND_ARGS) == set(runtime._HANDLERS)


@pytest.mark.parametrize(
    ("kind", "args", "detail"),
    [
        ("no_such_kind", {}, "unknown command kind 'no_such_kind'"),
        ("set_dataloader_finite_check", {"enabled": "false"}, "enabled"),
        ("set_dataloader_finite_check", {}, "enabled: Field required"),
        ("set_learning_rate", {"value": True}, "value"),
        ("set_learning_rate", {"value": -0.1}, "value"),
        ("set_learning_rate", {"value": 0.1, "optimizer": -1}, "optimizer"),
        ("set_learning_rate", {"value": 0.1, "param_group": "0"}, "param_group"),
        ("set_learning_rate", {"value": 0.1, "colour": "red"}, "colour"),
        ("set_hparam", {"value": 1}, "name: Field required"),
        ("torchviz_capture", {"show_attrs": "false"}, "show_attrs"),
        ("profile", {"active_steps": 0}, "active_steps"),
        ("profile", {"level": "everything"}, "level"),
        ("reset_momentum", {"selector": {"param_group": "0"}}, "selector"),
        ("checkpoint", {"label": "../escape"}, "label"),
        ("snapshot", {"kind": "bogus"}, "kind"),
    ],
)
def test_invalid_command_is_rejected_at_submission(
    tmp_path: Path, kind: str, args: dict, detail: str
) -> None:
    runtime, client = _runtime_and_client(tmp_path)

    response = client.post("/commands", json={"kind": kind, "args": args})

    assert response.status_code == 400, response.text
    assert detail in response.json()["detail"]
    assert runtime.command_queue.list_all() == []


def test_nan_learning_rate_is_rejected_at_submission(tmp_path: Path) -> None:
    runtime, client = _runtime_and_client(tmp_path)

    response = client.post(
        "/commands",
        content='{"kind": "set_learning_rate", "args": {"value": NaN}}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400, response.text
    assert runtime.command_queue.list_all() == []


def test_unknown_snapshot_kind_is_rejected_by_the_snapshots_route(
    tmp_path: Path,
) -> None:
    runtime, client = _runtime_and_client(tmp_path)

    response = client.post("/snapshots", json={"kind": "bogus"})

    assert response.status_code == 400, response.text
    assert runtime.command_queue.list_all() == []


def test_accepted_command_shows_its_args_with_defaults(tmp_path: Path) -> None:
    _, client = _runtime_and_client(tmp_path)

    response = client.post(
        "/commands", json={"kind": "profile", "args": {"active_steps": 3}}
    )

    assert response.status_code == 200, response.text
    assert response.json()["args"] == {
        "warmup_steps": 0,
        "active_steps": 3,
        "level": "basic",
        "then_hold": False,
    }


class _FakeState:
    stage = None


class _FakeTrainer:
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.callback_metrics: dict[str, float] = {}
        self.current_epoch = 0
        self.global_step = 0
        self.optimizers = [optimizer]
        self.state = _FakeState()


def test_learning_rate_for_a_missing_optimizer_fails_with_the_optimizer_count(
    tmp_path: Path,
) -> None:
    runtime, _ = _runtime_and_client(tmp_path)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.trainer = _FakeTrainer(optimizer)
    runtime._gather_ranks(model)
    runtime.on_fit_start(model)

    command = runtime.command_queue.submit(
        "set_learning_rate", {"value": 0.5, "optimizer": 3}, parse_when("now")
    )
    runtime.process_safe_point(model, SafePoint.TRAIN_BATCH_END)

    failed = runtime.command_queue.get(command.id)
    assert failed.status == "failed"
    assert failed.error == "optimizer 3 does not exist; the run has 1 optimizer(s)"
    assert optimizer.param_groups[0]["lr"] == 0.1
