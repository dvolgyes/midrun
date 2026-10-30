"""Readers get plain floats for metrics and learning rates that a safe point published as tensors."""

import json
import math
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.rest.app import build_app
from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.commands.command import Command
from midrun.runtime.commands.command_when import CommandWhen
from midrun.runtime.events import SafePoint
from midrun.runtime.handlers.handle_snapshot import handle_snapshot
from midrun.runtime.runtime import MidrunRuntime
from midrun.runtime.state.host_snapshot import host_snapshot
from midrun.runtime.virtual_fs.static_files import STATIC_FILES


class _FakeState:
    stage = None


class _FakeTrainer:
    def __init__(self, metrics: dict[str, object], lrs: list[object]) -> None:
        self.callback_metrics = metrics
        self.current_epoch = 1
        self.global_step = 7
        self.optimizers = [
            torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=lr) for lr in lrs
        ]
        self.state = _FakeState()


class _Model(torch.nn.Module):
    pass


def _published(tmp_path: Path, metrics: dict[str, object], lrs: list[object]):
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = MidrunRuntime(config, load_backend("lightning.pytorch"))
    runtime.artifacts = ArtifactStore(tmp_path)
    model = _Model()
    model.trainer = _FakeTrainer(metrics, lrs)
    runtime._gather_ranks(model)
    runtime.on_fit_start(model)
    runtime.process_safe_point(model, SafePoint.TRAIN_BATCH_END)
    return runtime, model


def test_each_reader_gets_float_metrics_and_learning_rates(tmp_path: Path) -> None:
    runtime, model = _published(
        tmp_path,
        {"loss": torch.tensor(0.5), "count": torch.tensor(3), "number": 2},
        [torch.tensor(0.25), 0.125],
    )
    expected = {"loss": 0.5, "count": 3.0, "number": 2.0}
    client = TestClient(build_app(runtime), base_url="http://localhost")

    rest_metrics = client.get("/metrics").json()
    fuse_metrics = json.loads(STATIC_FILES["/state/metrics/index.json"](runtime))
    rest_run = client.get("/run").json()
    fuse_run = json.loads(STATIC_FILES["/state/snapshot.json"](runtime))
    command = Command(
        id="c1",
        kind="snapshot",
        args={"kind": "runtime"},
        when=CommandWhen(safe_point=None, occurrence="immediate"),
        created_at=0.0,
    )
    manifest = handle_snapshot(runtime, model, model.trainer, command, None)
    written = json.loads((Path(manifest["path"]) / "runtime.json").read_text())

    for metrics in (
        rest_metrics,
        fuse_metrics,
        rest_run["metrics"],
        fuse_run["metrics"],
    ):
        assert {key: metrics[key]["value"] for key in metrics} == expected
        assert metrics["loss"]["step"] == 7
        assert metrics["loss"]["epoch"] == 1
    assert {
        key: value["value"] for key, value in written["metrics"].items()
    } == expected
    assert client.get("/optimizers").json() == {"learning_rates": [0.25, 0.125]}
    assert rest_run["learning_rates"] == [0.25, 0.125]
    assert fuse_run["learning_rates"] == [0.25, 0.125]


def test_non_scalar_and_non_numeric_metrics_are_left_out(tmp_path: Path) -> None:
    runtime, _ = _published(
        tmp_path,
        {
            "loss": torch.tensor(0.5),
            "vector": torch.ones(3),
            "empty": torch.empty(0),
            "complex": torch.tensor(1 + 1j),
            "text": "abc",
            "missing": None,
        },
        [0.1],
    )

    assert set(host_snapshot(runtime.state.read()).metrics) == {"loss"}


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_non_finite_metric_reads_as_the_same_float(
    tmp_path: Path, value: float
) -> None:
    runtime, _ = _published(tmp_path, {"loss": torch.tensor(value)}, [0.1])

    read = host_snapshot(runtime.state.read()).metrics["loss"].value

    assert type(read) is float
    assert read == value or (math.isnan(read) and math.isnan(value))


def test_a_learning_rate_changed_in_place_reads_as_of_the_last_safe_point(
    tmp_path: Path,
) -> None:
    lr = torch.tensor(0.5)
    runtime, model = _published(tmp_path, {}, [lr])

    lr.fill_(0.25)
    before = host_snapshot(runtime.state.read()).learning_rates
    runtime.process_safe_point(model, SafePoint.TRAIN_BATCH_END)

    assert before == (0.5,)
    assert host_snapshot(runtime.state.read()).learning_rates == (0.25,)
