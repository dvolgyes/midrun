"""Behavioral tests for the torchinfo inspector and its command handler."""

import json
import sys
from pathlib import Path

import pytest
import torch
from lightning.pytorch import LightningModule

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.command_specs.execute_command import execute_command
from midrun.runtime.commands.parse_when import parse_when
from midrun.runtime.runtime import MidrunRuntime
from midrun.runtime.torchinfo_inspector.basic_summary_json import (
    basic_summary_json,
)
from midrun.runtime.torchinfo_inspector.basic_summary_text import (
    basic_summary_text,
)
from midrun.runtime.torchinfo_inspector.torchinfo_rich_summary import (
    torchinfo_rich_summary,
)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _make_runtime(tmp_path: Path, **overrides: object) -> MidrunRuntime:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False, **overrides
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


def test_basic_summary_json_counts_params_without_torchinfo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torchinfo", None)
    model = _Model()
    payload = basic_summary_json(model)
    assert payload["total_params"] == 10
    assert payload["trainable_params"] == 10
    assert payload["non_trainable_params"] == 0
    assert payload["torchinfo_used"] is False


def test_basic_summary_text_falls_back_to_str_without_torchinfo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torchinfo", None)
    model = _Model()
    assert basic_summary_text(model) == str(model)


def test_basic_summary_text_uses_torchinfo_when_available() -> None:
    model = _Model()
    text = basic_summary_text(model)
    assert "Total params" in text


@pytest.mark.parametrize(("tied", "total"), [(True, 10_200), (False, 20_200)])
def test_summaries_count_unique_model_parameters(tied: bool, total: int) -> None:
    class TwoLayers(LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.first = torch.nn.Linear(100, 100, dtype=torch.float64)
            self.second = torch.nn.Linear(100, 100, dtype=torch.float64)
            if tied:
                self.second.weight = self.first.weight
            self.second.bias.requires_grad_(False)
            self.forward_calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.forward_calls += 1
            return self.second(self.first(x))

    model = TwoLayers()
    trainable = total - 100
    static_text = basic_summary_text(model)
    assert model.forward_calls == 0
    assert basic_summary_json(model) == {
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": 100,
        "torchinfo_used": True,
    }

    live = torchinfo_rich_summary(model, torch.zeros(3, 100, dtype=torch.float64))
    assert model.forward_calls == 1
    assert json.loads(live["summary.json"]) == {
        "total_params": total,
        "trainable_params": trainable,
        "total_mult_adds": 60_600,
    }
    for text in (static_text, live["summary.txt"]):
        assert f"Total params: {total:,}" in text.splitlines()
        assert f"Trainable params: {trainable:,}" in text.splitlines()
        assert "Non-trainable params: 100" in text.splitlines()
        assert text.count("Linear:") == 2
    params_mb = total * 8 / 1e6
    assert f"Params size (MB): {params_mb:.2f}" in live["summary.txt"].splitlines()


def test_rich_summary_raises_without_torchinfo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torchinfo", None)
    model = _Model()
    with pytest.raises(ImportError):
        torchinfo_rich_summary(model, torch.randn(3, 4))


def test_rich_summary_does_not_perturb_batchnorm_running_stats() -> None:
    class BNModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.bn(x)

    model = BNModel()
    model.train()
    running_mean_before = model.bn.running_mean.clone()

    torchinfo_rich_summary(model, torch.randn(8, 4) * 10 + 5)

    assert model.training is True
    assert torch.equal(running_mean_before, model.bn.running_mean)


def test_torchinfo_capture_command_writes_artifact(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    model = _Model()
    runtime.current_step.begin(
        batch=torch.randn(3, 4), batch_idx=0, epoch=0, global_step=0, started_ns=0
    )

    cmd = runtime.command_queue.submit("torchinfo_capture", {}, parse_when("now"))
    execute_command(runtime, model, trainer, cmd, safe_point=None)

    assert cmd.status == "succeeded"
    assert (Path(cmd.result["path"]) / "summary.txt").exists()
    assert (Path(cmd.result["path"]) / "summary.json").exists()


def test_torchinfo_capture_fails_without_in_flight_batch(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    model = _Model()

    cmd = runtime.command_queue.submit("torchinfo_capture", {}, parse_when("now"))
    execute_command(runtime, model, trainer, cmd, safe_point=None)

    assert cmd.status == "failed"
    assert "in-flight batch" in cmd.error
