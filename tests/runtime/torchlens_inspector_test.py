"""Behavioral tests for the TorchLens inspector and its hold-gated command handler."""

import shutil
import sys
import types
from pathlib import Path

import pytest
import torch

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.command_specs.execute_command import execute_command
from midrun.runtime.commands.parse_when import parse_when
from midrun.runtime.runtime import MidrunRuntime
from midrun.runtime.torchlens_capture import torchlens_capture

_HAS_DOT = shutil.which("dot") is not None


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


def test_capture_raises_without_torchlens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torchlens", None)
    model = _Model()
    with pytest.raises(ImportError):
        torchlens_capture(model, torch.randn(3, 4))


def test_torchlens_capture_auto_engages_hold_when_not_already_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """torchlens instruments the model with hooks, which is fragile enough that an
    operator should get a chance to inspect state before resuming -- so the command
    auto-engages a hold rather than rejecting outright, and (unlike a plain capture
    command) never auto-releases it, regardless of whether the capture itself then
    succeeds or fails."""
    # An importable torchlens without a usable API: the capture fails after the hold.
    monkeypatch.setitem(sys.modules, "torchlens", types.ModuleType("torchlens"))
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    model = _Model()
    runtime.current_step.begin(
        batch=torch.randn(3, 4), batch_idx=0, epoch=0, global_step=0, started_ns=0
    )
    assert not runtime.gate.is_held()

    cmd = runtime.command_queue.submit("torchlens_capture", {}, parse_when("now"))
    execute_command(runtime, model, trainer, cmd, safe_point=None)

    assert runtime.gate.is_held()


def test_torchlens_capture_without_torchlens_fails_and_engages_no_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "torchlens", None)
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    model = _Model()
    runtime.current_step.begin(
        batch=torch.randn(3, 4), batch_idx=0, epoch=0, global_step=0, started_ns=0
    )

    cmd = runtime.command_queue.submit("torchlens_capture", {}, parse_when("now"))
    execute_command(runtime, model, trainer, cmd, safe_point=None)

    assert cmd.status == "failed"
    assert "torchlens" in cmd.error
    assert not runtime.gate.is_held()


@pytest.mark.skipif(not _HAS_DOT, reason="requires Graphviz dot")
def test_capture_returns_summary_and_graph() -> None:
    model = _Model()
    add = torch.add
    files = torchlens_capture(model, torch.randn(3, 4))
    assert files["summary.txt"].strip()
    assert b"<svg" in files["graph.svg"]
    assert torch.add is add


@pytest.mark.skipif(not _HAS_DOT, reason="requires Graphviz dot")
def test_torchlens_capture_command_succeeds_during_active_hold(
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    runtime.gate.enter("cmd_1", "train_batch_start", epoch=0, step=0)
    trainer = _FakeTrainer()
    model = _Model()
    runtime.current_step.begin(
        batch=torch.randn(3, 4), batch_idx=0, epoch=0, global_step=0, started_ns=0
    )

    cmd = runtime.command_queue.submit("torchlens_capture", {}, parse_when("now"))
    execute_command(runtime, model, trainer, cmd, safe_point=None)

    assert cmd.status == "succeeded"
    assert (Path(cmd.result["path"]) / "summary.txt").exists()
    assert runtime.gate.is_held()
