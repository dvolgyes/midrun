"""Behavioral tests for the forward passes that inspectors run on the model.

Every inspector command and live model file runs its forward pass in eval mode, so
dropout and batch norm (also SyncBatchNorm, which otherwise does an all-reduce) behave
as in eval, and gives the model back in its previous mode. Under a strategy that shards
the parameters, a forward pass on rank 0 alone would wait on collectives that the other
ranks never join, so the inspectors refuse to run there.
"""

import shutil
import sys
import threading
import time
import types
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Self

import pytest
import torch
from lightning.pytorch.strategies import FSDPStrategy, ModelParallelStrategy
from torch.testing._internal.two_tensor import TwoTensor

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.command_specs.execute_command import execute_command
from midrun.runtime.commands.parse_when import parse_when
from midrun.runtime.inspector_forward import inspector_forward
from midrun.runtime.runtime import MidrunRuntime
from midrun.runtime.virtual_fs.virtual_file_content import virtual_file_content

_COMMANDS = (
    "torchinfo_capture",
    "torchview_capture",
    "torchviz_capture",
    "torchlens_capture",
)
_LIVE_FILES = ("/model/summary-live.txt", "/model/graph.dot")
_NEEDS_DOT = pytest.mark.skipif(
    shutil.which("dot") is None, reason="Graphviz 'dot' binary not installed"
)


class _Recorder(torch.nn.Module):
    """Records the training flag and the grad mode of each forward call."""

    seen: list[tuple[bool, bool]]
    fail: bool

    def __init__(self) -> None:
        super().__init__()
        self.seen = []
        self.fail = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen.append((self.training, torch.is_grad_enabled()))
        if self.fail:
            raise RuntimeError("forward failed on purpose")
        return x


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)
        self.bn = torch.nn.BatchNorm1d(4)
        self.dropout = torch.nn.Dropout(0.5)
        self.recorder = _Recorder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.recorder(self.dropout(self.bn(self.fc(x))))


class _FakeState:
    stage = None


class _FakeTrainer:
    def __init__(self, strategy: object | None = None) -> None:
        self.callback_metrics: dict[str, float] = {}
        self.current_epoch = 0
        self.global_step = 0
        self.optimizers: list[object] = []
        self.state = _FakeState()
        if strategy is not None:
            self.strategy = strategy


class _FakeTrace:
    def summary(self, level: str) -> str:
        return f"fake trace ({level})"

    def draw(self, vis_outpath: str, **kwargs: Any) -> None:
        del kwargs
        Path(vis_outpath).with_suffix(".svg").write_bytes(b"<svg/>")


def _install_fake_torchlens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Installs a torchlens stand-in whose `trace` runs one real forward pass."""

    def trace(model: torch.nn.Module, batch: Any, **kwargs: Any) -> _FakeTrace:
        del kwargs
        model(batch)
        return _FakeTrace()

    module = types.ModuleType("torchlens")
    module.trace = trace
    monkeypatch.setitem(sys.modules, "torchlens", module)


def _make_runtime(tmp_path: Path) -> MidrunRuntime:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = build_runtime(config, load_backend("lightning.pytorch"))
    runtime.artifacts = ArtifactStore(tmp_path)
    runtime.current_step.begin(
        batch=torch.randn(8, 4), batch_idx=0, epoch=0, global_step=0, started_ns=0
    )
    return runtime


def _run_command(
    runtime: MidrunRuntime, model: torch.nn.Module, trainer: _FakeTrainer, kind: str
) -> Any:
    cmd = runtime.command_queue.submit(kind, {}, parse_when("now"))
    execute_command(runtime, model, trainer, cmd, safe_point=None)
    return cmd


def _read_live_file(
    runtime: MidrunRuntime, model: torch.nn.Module, path: str
) -> bytes | BaseException:
    """Reads `path` on a reader thread while this thread serves the safe point."""
    outcome: list[bytes | BaseException] = []

    def read() -> None:
        try:
            outcome.append(virtual_file_content(runtime, path))
        except Exception as exc:  # noqa: BLE001 -- handed back to the test thread
            outcome.append(exc)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    deadline = time.monotonic() + 5.0
    while not runtime.live_captures._pending and time.monotonic() < deadline:
        time.sleep(0.01)
    runtime.live_captures.serve(model)
    reader.join(timeout=5.0)
    assert outcome, "the live read did not finish"
    return outcome[0]


@_NEEDS_DOT
@pytest.mark.parametrize(
    ("kind", "grad_enabled"),
    [
        ("torchinfo_capture", False),
        ("torchview_capture", False),
        # torchviz draws the autograd graph, so its forward pass keeps grad on.
        ("torchviz_capture", True),
        ("torchlens_capture", False),
    ],
)
def test_capture_command_runs_forward_in_eval_and_restores_train_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, grad_enabled: bool
) -> None:
    _install_fake_torchlens(monkeypatch)
    runtime = _make_runtime(tmp_path)
    model = _Model()
    model.train()

    cmd = _run_command(runtime, model, _FakeTrainer(), kind)

    assert cmd.status == "succeeded", cmd.error
    assert model.recorder.seen
    assert all(entry == (False, grad_enabled) for entry in model.recorder.seen)
    assert model.training
    assert all(module.training for module in model.modules())


@pytest.mark.parametrize("path", _LIVE_FILES)
def test_live_model_file_runs_forward_in_eval_and_restores_train_mode(
    tmp_path: Path, path: str
) -> None:
    runtime = _make_runtime(tmp_path)
    model = _Model()
    model.trainer = _FakeTrainer()
    model.train()

    content = _read_live_file(runtime, model, path)

    assert isinstance(content, bytes), content
    assert model.recorder.seen
    assert all(entry == (False, False) for entry in model.recorder.seen)
    assert model.training


@_NEEDS_DOT
@pytest.mark.parametrize("kind", _COMMANDS)
def test_capture_command_keeps_a_frozen_submodule_in_eval_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    _install_fake_torchlens(monkeypatch)
    runtime = _make_runtime(tmp_path)
    model = _Model()
    model.train()
    model.bn.eval()

    cmd = _run_command(runtime, model, _FakeTrainer(), kind)

    assert cmd.status == "succeeded", cmd.error
    assert model.training
    assert model.fc.training
    assert not model.bn.training


def test_capture_command_restores_train_mode_when_forward_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_torchlens(monkeypatch)
    runtime = _make_runtime(tmp_path)
    model = _Model()
    model.recorder.fail = True
    model.train()

    cmd = _run_command(runtime, model, _FakeTrainer(), "torchlens_capture")

    assert cmd.status == "failed"
    assert "on purpose" in cmd.error
    assert model.recorder.seen == [(False, False)]
    assert model.training


class _ModeDependentDropout(torch.nn.Dropout):
    scale: int

    def __init__(self) -> None:
        super().__init__(p=0.0)
        self.scale = 2

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        self.scale = 2 if mode else 1
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) * self.scale


@pytest.mark.parametrize("dropout_training", [False, True], ids=["eval", "train"])
@pytest.mark.parametrize("fails", [False, True], ids=["success", "error"])
def test_inspector_forward_restores_modes_with_shared_dropout(
    fails: bool, dropout_training: bool
) -> None:
    dropout = _ModeDependentDropout()
    recorder = _Recorder()
    recorder.fail = fails
    model = torch.nn.Sequential(
        torch.nn.Sequential(dropout), torch.nn.Sequential(dropout), recorder
    )
    model[0].eval()
    dropout.train(dropout_training)
    sample = torch.ones(4)
    original_output = dropout(sample)
    original_modes = {name: module.training for name, module in model.named_modules()}

    expectation = (
        pytest.raises(RuntimeError, match="forward failed on purpose")
        if fails
        else nullcontext()
    )
    with (
        expectation,
        inspector_forward(
            load_backend("lightning.pytorch"), None, model, "test inspector"
        ),
    ):
        assert not any(module.training for module in model.modules())
        torch.testing.assert_close(model(sample), sample)

    assert recorder.seen == [(False, False)]
    restored_modes = {name: module.training for name, module in model.named_modules()}
    assert restored_modes == original_modes
    torch.testing.assert_close(dropout(sample), original_output)


@_NEEDS_DOT
@pytest.mark.parametrize("layer", [torch.nn.Embedding, torch.nn.EmbeddingBag])
@pytest.mark.parametrize("max_norm", [None, 0.0, 1.0])
def test_graph_capture_preserves_embedding_weights(
    tmp_path: Path,
    layer: type[torch.nn.Embedding | torch.nn.EmbeddingBag],
    max_norm: float | None,
) -> None:
    runtime = _make_runtime(tmp_path)
    runtime.current_step.begin(torch.tensor([[0, 1]]), 0, 0, 0, 0)
    embedding = layer(2, 2, max_norm=max_norm)
    model = torch.nn.Sequential(embedding)
    with torch.no_grad():
        embedding.weight.fill_(3.0)
    before = embedding.weight.detach().clone()

    cmd = _run_command(runtime, model, _FakeTrainer(), "torchviz_capture")

    assert torch.equal(embedding.weight, before)
    if max_norm is None:
        assert cmd.status == "succeeded", cmd.error
    else:
        assert cmd.status == "failed"
        assert "max_norm" in cmd.error


@pytest.mark.parametrize("strategy_class", [FSDPStrategy, ModelParallelStrategy])
@pytest.mark.parametrize("kind", _COMMANDS)
def test_capture_command_is_refused_under_a_sharding_strategy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    strategy_class: type,
) -> None:
    _install_fake_torchlens(monkeypatch)
    runtime = _make_runtime(tmp_path)
    model = _Model()
    trainer = _FakeTrainer(strategy=strategy_class())

    cmd = _run_command(runtime, model, trainer, kind)

    assert cmd.status == "failed"
    assert "refused" in cmd.error
    assert strategy_class.__name__ in cmd.error
    assert model.recorder.seen == []
    assert not runtime.gate.is_held()


@pytest.mark.parametrize("kind", _COMMANDS)
def test_capture_command_is_refused_for_a_wrapper_subclass_parameter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    _install_fake_torchlens(monkeypatch)
    runtime = _make_runtime(tmp_path)
    model = _Model()
    # TwoTensor is a traceable wrapper subclass, as DTensor is.
    model.fc.weight = torch.nn.Parameter(
        TwoTensor(torch.zeros(4, 4), torch.zeros(4, 4))
    )

    cmd = _run_command(runtime, model, _FakeTrainer(), kind)

    assert cmd.status == "failed"
    assert "refused" in cmd.error
    assert "fc.weight" in cmd.error
    assert model.recorder.seen == []


@pytest.mark.parametrize("path", _LIVE_FILES)
def test_live_model_file_is_refused_under_a_sharding_strategy(
    tmp_path: Path, path: str
) -> None:
    runtime = _make_runtime(tmp_path)
    model = _Model()
    model.trainer = _FakeTrainer(strategy=FSDPStrategy())

    outcome = _read_live_file(runtime, model, path)

    assert isinstance(outcome, ValueError)
    assert "refused" in str(outcome)
    assert "FSDPStrategy" in str(outcome)
    assert model.recorder.seen == []
