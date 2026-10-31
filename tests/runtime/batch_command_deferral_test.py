"""Behavioral tests: a command that needs a batch waits for a safe point that has one.

An epoch-end safe point, or a hold that started at one, has no in-flight batch. A
batch inspector submitted there stays queued and runs at the next batch, instead of
failing because the submission came just after an epoch's last batch.
"""

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.rest.app import build_app
from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.commands.parse_when import parse_when
from midrun.runtime.events import SafePoint
from midrun.runtime.runtime import MidrunRuntime

BATCH_KINDS = (
    "torchinfo_capture",
    "torchview_capture",
    "torchviz_capture",
    "save_current_batch",
)


def _make_runtime(tmp_path: Path) -> MidrunRuntime:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = build_runtime(config, load_backend("lightning.pytorch"))
    runtime.artifacts = ArtifactStore(tmp_path)
    return runtime


class _FakeState:
    stage = None


class _FakeTrainer:
    def __init__(self) -> None:
        self.callback_metrics: dict[str, float] = {}
        self.current_epoch = 0
        self.global_step = 0
        self.optimizers: list[object] = []
        self.state = _FakeState()


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _ready_model(runtime: MidrunRuntime) -> _Model:
    model = _Model()
    model.trainer = _FakeTrainer()
    runtime.on_fit_start(model)
    return model


@pytest.mark.parametrize("kind", BATCH_KINDS)
def test_a_batch_command_waits_through_an_epoch_end_for_the_next_batch(
    tmp_path: Path, kind: str
) -> None:
    runtime = _make_runtime(tmp_path)
    model = _ready_model(runtime)
    command = runtime.command_queue.submit(kind, {}, parse_when("next_safe_point"))

    runtime.process_safe_point(model, SafePoint.TRAIN_EPOCH_END)

    assert runtime.command_queue.get(command.id).status == "queued"

    runtime.current_step.begin(
        batch=torch.randn(3, 4), batch_idx=0, epoch=1, global_step=8, started_ns=0
    )
    runtime.process_safe_point(model, SafePoint.TRAIN_BATCH_END)

    finished = runtime.command_queue.get(command.id)
    assert finished.status == "succeeded", finished.error


def test_a_command_without_a_batch_need_still_runs_at_an_epoch_end(
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    model = _ready_model(runtime)
    command = runtime.command_queue.submit(
        "snapshot", {"kind": "runtime"}, parse_when("next_safe_point")
    )

    runtime.process_safe_point(model, SafePoint.TRAIN_EPOCH_END)

    assert runtime.command_queue.get(command.id).status == "succeeded"


@pytest.mark.parametrize(
    "safe_point", ["train_epoch_end", "validation_epoch_end", "fit_end"]
)
def test_a_batch_command_scheduled_at_a_safe_point_without_a_batch_is_refused(
    tmp_path: Path, safe_point: str
) -> None:
    runtime = _make_runtime(tmp_path)
    client = TestClient(build_app(runtime), base_url="http://localhost")

    response = client.post(
        "/commands",
        json={
            "kind": "torchview_capture",
            "args": {},
            "execution": {"when": {"safe_point": safe_point}},
        },
    )

    assert response.status_code == 422
    assert "in-flight batch" in str(response.json()["detail"])
    assert runtime.command_queue.list_all() == []


@pytest.mark.parametrize("safe_point", ["train_batch_end", "before_optimizer_step"])
def test_a_batch_command_scheduled_at_a_batch_safe_point_is_accepted(
    tmp_path: Path, safe_point: str
) -> None:
    runtime = _make_runtime(tmp_path)
    client = TestClient(build_app(runtime), base_url="http://localhost")

    response = client.post(
        "/commands",
        json={
            "kind": "torchview_capture",
            "args": {},
            "execution": {"when": {"safe_point": safe_point}},
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "queued"
