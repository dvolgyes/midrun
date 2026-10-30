"""A tunable hparam tensor on a GPU is copied to the host when read, not at a safe point."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.events import SafePoint
from midrun.runtime.runtime import MidrunRuntime

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


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
    midrun_tunable_hparams = {"class_weights": {"label": "Class weights"}}

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)
        self.class_weights = torch.tensor([0.25, 0.75], device="cuda")


@contextmanager
def _device_syncs_raise() -> Iterator[None]:
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode("default")


def _ready_model(tmp_path: Path) -> tuple[MidrunRuntime, _Model]:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = build_runtime(config, load_backend("lightning.pytorch"))
    runtime.artifacts = ArtifactStore(tmp_path)
    model = _Model()
    model.trainer = _FakeTrainer()
    runtime.on_fit_start(model)
    return runtime, model


def test_a_safe_point_does_not_copy_a_gpu_hparam_to_the_host(tmp_path: Path) -> None:
    runtime, model = _ready_model(tmp_path)

    with _device_syncs_raise():
        runtime.process_safe_point(model, SafePoint.TRAIN_BATCH_END)
    described = runtime.tunable_hparams["class_weights"]

    assert described.value == [0.25, 0.75]
    assert described.device == "cuda:0"
    assert described.label == "Class weights"


def test_a_read_shows_a_gpu_hparam_changed_in_place_after_the_safe_point(
    tmp_path: Path,
) -> None:
    runtime, model = _ready_model(tmp_path)
    runtime.process_safe_point(model, SafePoint.TRAIN_BATCH_END)

    model.class_weights.fill_(0.5)

    assert runtime.tunable_hparams["class_weights"].value == [0.5, 0.5]
