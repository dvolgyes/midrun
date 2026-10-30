"""A CUDA fit publishes metrics that readers see whole, and holds no device memory for them."""

import gc
import threading
import weakref
from pathlib import Path

import lightning.pytorch as PL
import pytest
import torch
from fastapi.testclient import TestClient
from torch.utils.data import DataLoader

from midrun.mixin import MidrunMixin
from midrun.rest.app import build_app
from midrun.runtime.state.host_snapshot import host_snapshot

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

_TRIALS = range(3)


class _Training:
    """The training step of both models: two metrics from one graph."""

    def setup_training(self) -> None:
        self.layer = torch.nn.Linear(8, 1)
        self.loss_refs: list[weakref.ref] = []
        self.memory: list[int] = []

    def training_step(self, batch, batch_idx):
        del batch, batch_idx
        loss = self.layer(torch.randn(16, 8, device=self.device)).pow(2).mean()
        self.loss_refs.append(weakref.ref(loss))
        self.log("loss", loss, on_step=True)
        self.log("half", loss / 2, on_step=True)
        return loss

    def configure_optimizers(self):
        lr = torch.tensor(0.125, device=self.device)
        return torch.optim.Adam(self.parameters(), lr=lr, capturable=True)


class _Plain(_Training, PL.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.setup_training()


class _Midrun(MidrunMixin, _Training, PL.LightningModule):
    def __init__(self, *, compare: bool = False) -> None:
        super().__init__(
            rest_enabled=False,
            fuse_enabled=False,
            torch_debug_enabled=False,
            hooks_enabled=False,
        )
        self.setup_training()
        self.compare = compare
        self.compared: list[tuple[dict[str, float], dict[str, float]]] = []

    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)
        if not self.compare:
            return
        # `float` waits for the device, so only a fit without strict syncs compares.
        reference = {
            key: float(value) for key, value in self.trainer.callback_metrics.items()
        }
        metrics = host_snapshot(self._midrun.state.read()).metrics
        self.compared.append(
            (reference, {key: metric.value for key, metric in metrics.items()})
        )


class _RecordsMemory(PL.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        pl_module.memory.append(torch.cuda.memory_allocated())


class _SyncsRaise(PL.Callback):
    """Makes each device sync in the training thread raise from the second batch."""

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx == 1:
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.set_sync_debug_mode("default")


def _fit(model: PL.LightningModule, tmp_path: Path, steps: int, callback) -> None:
    trainer = PL.Trainer(
        default_root_dir=str(tmp_path),
        max_steps=steps,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
        accelerator="gpu",
        devices=1,
        callbacks=[callback],
    )
    # Integer batches: Lightning has no batch to copy to the device.
    try:
        trainer.fit(model, DataLoader(range(steps), batch_size=None))
    finally:
        torch.cuda.set_sync_debug_mode("default")


def _memory_run(model_type: type, tmp_path: Path) -> tuple[list[int], int, int, int]:
    """Returns the allocation over the start per batch, after the fit, and after the
    model and trainer are gone, and the losses still alive after the fit.

    Only the published snapshot of a Midrun model outlives the model.
    """
    gc.collect()
    torch.cuda.synchronize()
    start = torch.cuda.memory_allocated()
    torch.manual_seed(0)
    model = model_type()
    _fit(model, tmp_path, 60, _RecordsMemory())
    gc.collect()
    torch.cuda.synchronize()
    after_fit = torch.cuda.memory_allocated() - start
    live_losses = sum(ref() is not None for ref in model.loss_refs)
    per_batch = [used - start for used in model.memory]
    snapshot = model._midrun.state.read() if isinstance(model, _Midrun) else None
    del model
    gc.collect()
    torch.cuda.synchronize()
    after_model = torch.cuda.memory_allocated() - start
    if snapshot is not None:
        assert host_snapshot(snapshot).metrics, "the snapshot must keep its metrics"
    return per_batch, after_fit, after_model, live_losses


@pytest.mark.parametrize("trial", _TRIALS)
def test_device_memory_and_graphs_match_a_plain_lightning_run(
    tmp_path: Path, trial: int
) -> None:
    del trial
    plain = _memory_run(_Plain, tmp_path / "plain")
    midrun = _memory_run(_Midrun, tmp_path / "midrun")

    assert midrun == plain
    assert midrun[2] == 0


@pytest.mark.parametrize("trial", _TRIALS)
def test_read_values_equal_float_of_the_same_steps_metrics(
    tmp_path: Path, trial: int
) -> None:
    del trial
    model = _Midrun(compare=True)

    _fit(model, tmp_path, 30, PL.Callback())

    assert len(model.compared) == 30
    for reference, read in model.compared:
        assert read == reference
        assert all(type(value) is float for value in read.values())


@pytest.mark.parametrize("trial", _TRIALS)
def test_a_reader_during_a_strict_sync_fit_sees_whole_monotone_states(
    tmp_path: Path, trial: int
) -> None:
    del trial
    model = _Midrun()
    client = TestClient(build_app(model._midrun), base_url="http://localhost")
    stop = threading.Event()
    reads: list[dict] = []
    errors: list[BaseException] = []

    def hammer() -> None:
        while not stop.is_set():
            try:
                response = client.get("/run")
                response.raise_for_status()
                reads.append(response.json())
            except BaseException as exc:  # noqa: BLE001 -- the test reports every error
                errors.append(exc)
                return

    reader = threading.Thread(target=hammer)
    reader.start()
    try:
        _fit(model, tmp_path, 40, _SyncsRaise())
    finally:
        stop.set()
        reader.join()

    assert errors == []
    with_metrics = [read for read in reads if read["metrics"]]
    assert len(with_metrics) > 10
    for read in with_metrics:
        loss, half = read["metrics"]["loss"], read["metrics"]["half"]
        assert half["value"] == loss["value"] / 2
        assert half["step"] == loss["step"] == read["global_step"]
        assert read["learning_rates"] == [0.125]
    generations = [read["generation"] for read in reads]
    steps = [read["global_step"] for read in reads]
    assert generations == sorted(generations)
    assert steps == sorted(steps)
