"""The runtime drops its live batch, loss and exception references when a step ends early."""

import threading
import time
from pathlib import Path

import lightning.pytorch as PL
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.mixin import MidrunMixin
from midrun.runtime.commands.parse_when import parse_when
from midrun.runtime.runtime import MidrunRuntime


class _FailingModel(MidrunMixin, PL.LightningModule):
    def __init__(self) -> None:
        super().__init__(
            fuse_enabled=False,
            rest_enabled=False,
            torch_debug_enabled=False,
            hooks_enabled=False,
        )
        self.layer = torch.nn.Linear(4, 1)

    def training_step(self, batch, batch_idx):
        raise RuntimeError("boom")

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class _FakeState:
    stage = None


class _FakeTrainer:
    def __init__(self) -> None:
        self.callback_metrics: dict[str, float] = {}
        self.current_epoch = 0
        self.global_step = 0
        self.optimizers: list[object] = []
        self.state = _FakeState()


class _FakePLModule:
    def __init__(self) -> None:
        self.trainer = _FakeTrainer()

    def named_parameters(self):
        return iter(())


def _runtime() -> MidrunRuntime:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    return build_runtime(config, load_backend("lightning.pytorch"))


def _assert_released(runtime: MidrunRuntime) -> None:
    step = runtime.current_step
    assert (step.batch, step.loss, step.exception) == (None, None, None)


def test_a_failed_fit_releases_the_in_flight_batch(tmp_path: Path) -> None:
    model = _FailingModel()
    trainer = PL.Trainer(
        default_root_dir=str(tmp_path),
        max_epochs=1,
        limit_train_batches=1,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
    )
    loader = DataLoader(TensorDataset(torch.randn(4, 4)), batch_size=4)

    with pytest.raises(RuntimeError, match="boom"):
        trainer.fit(model, loader)

    _assert_released(model.midrun)


def test_teardown_releases_the_in_flight_batch() -> None:
    runtime = _runtime()
    runtime.current_step.begin(
        batch=torch.ones(2), batch_idx=0, epoch=0, global_step=0, started_ns=0
    )
    runtime.current_step.loss = torch.tensor(1.0)

    runtime.teardown(None, "fit")

    _assert_released(runtime)


def test_a_released_exception_hold_drops_the_exception_and_its_batch() -> None:
    runtime = _runtime()
    pl_module = _FakePLModule()
    runtime.current_step.begin(
        batch=torch.ones(2), batch_idx=0, epoch=0, global_step=0, started_ns=0
    )
    thread = threading.Thread(
        target=runtime.handle_backward_exception,
        args=(pl_module, torch.tensor(1.0), ValueError("boom")),
    )
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while (
            runtime.state.read().status != "exception_held"
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        runtime.command_queue.submit("release_exception", {}, parse_when("now"))
        runtime.gate.notify()
        thread.join(timeout=5)
        assert not thread.is_alive()
    finally:
        if thread.is_alive():
            runtime.exception_hold = None
            runtime.gate.notify()
            thread.join(timeout=5)

    _assert_released(runtime)
