"""Behavioral tests for PipelinePressureTracker's phase timing and rolling summary."""

from pathlib import Path

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.mixin import MidrunMixin
from midrun.runtime.pipeline_pressure import PipelinePressureTracker


def test_first_batch_has_no_input_gap() -> None:
    tracker = PipelinePressureTracker(window=8)

    tracker.batch_start(1_000, epoch=0, batch_idx=0)
    sample = tracker.batch_end(2_000, global_step=1)

    assert sample.input_gap_ns is None
    assert sample.batch_total_ns == 1_000


def test_second_batch_records_input_gap_from_previous_batch_end() -> None:
    tracker = PipelinePressureTracker(window=8)
    tracker.batch_start(1_000, epoch=0, batch_idx=0)
    tracker.batch_end(2_000, global_step=1)

    tracker.batch_start(2_500, epoch=0, batch_idx=1)
    sample = tracker.batch_end(3_000, global_step=2)

    assert sample.input_gap_ns == 500


def test_full_phase_breakdown_recorded_in_order() -> None:
    tracker = PipelinePressureTracker(window=8)
    tracker.batch_start(0, epoch=0, batch_idx=0)
    tracker.before_backward(100)
    tracker.after_backward(150)
    sample = tracker.batch_end(200, global_step=1)

    assert sample.forward_loss_ns == 100
    assert sample.backward_ns == 50
    assert sample.post_backward_ns == 50
    assert sample.batch_total_ns == 200


def test_multiple_backward_passes_partition_batch_time() -> None:
    tracker = PipelinePressureTracker(window=8)
    ms = 1_000_000
    tracker.batch_start(0, epoch=0, batch_idx=0)
    tracker.before_backward(2 * ms)
    tracker.after_backward(12 * ms)
    tracker.before_backward(15 * ms)
    tracker.after_backward(18 * ms)
    sample = tracker.batch_end(20 * ms, global_step=1)

    assert sample.forward_loss_ns == 5 * ms
    assert sample.backward_ns == 13 * ms
    assert sample.post_backward_ns == 2 * ms
    assert sample.batch_total_ns == 20 * ms


def test_missing_backward_phases_are_none() -> None:
    tracker = PipelinePressureTracker(window=8)
    tracker.batch_start(0, epoch=0, batch_idx=0)
    sample = tracker.batch_end(100, global_step=1)

    assert sample.forward_loss_ns is None
    assert sample.backward_ns is None
    assert sample.post_backward_ns is None


def test_window_evicts_oldest_samples() -> None:
    tracker = PipelinePressureTracker(window=2)
    for i in range(5):
        tracker.batch_start(i * 100, epoch=0, batch_idx=i)
        tracker.batch_end(i * 100 + 10, global_step=i)

    summary = tracker.summary()
    assert summary["samples"] == 2
    assert summary["latest_step"] == 4


def test_latest_returns_none_before_first_batch() -> None:
    tracker = PipelinePressureTracker(window=8)
    assert tracker.latest() is None


def test_summary_before_any_batch_reports_zero_samples() -> None:
    tracker = PipelinePressureTracker(window=8)
    summary = tracker.summary()
    assert summary == {"window": 8, "samples": 0}


def test_summary_identifies_largest_phase() -> None:
    tracker = PipelinePressureTracker(window=8)
    tracker.batch_start(0, epoch=0, batch_idx=0)
    tracker.before_backward(10)
    tracker.after_backward(1_000)
    sample = tracker.batch_end(1_010, global_step=1)

    assert sample.backward_ns == 990
    summary = tracker.summary()
    assert summary["largest_phase"] == "backward"


def test_nearest_rank_and_input_gap_to_mean_batch_ratio() -> None:
    tracker = PipelinePressureTracker(window=2)
    tracker.batch_start(0, epoch=0, batch_idx=0)
    tracker.batch_end(1_000_000, global_step=0)
    tracker.batch_start(101_000_000, epoch=0, batch_idx=1)
    tracker.batch_end(110_000_000, global_step=1)

    summary = tracker.summary()
    assert summary["batch_total_ms"] == {"p50": 1, "p95": 9}
    assert summary["phases"]["input_gap"] == {
        "p50_ms": 100,
        "p95_ms": 100,
        "max_ms": 100,
        "mean_to_batch_ratio": 20,
    }


class _FakeTrainer:
    current_epoch = 0
    global_step = 0


class _FakePLModule:
    trainer = _FakeTrainer()


def _run_one_batch(pipeline_pressure_enabled: bool) -> PipelinePressureTracker:
    config = MidrunConfig(
        rest_enabled=False,
        fuse_enabled=False,
        torch_debug_enabled=False,
        pipeline_pressure_enabled=pipeline_pressure_enabled,
    )
    runtime = build_runtime(config, load_backend("lightning.pytorch"))
    pl_module = _FakePLModule()
    runtime.begin_train_batch(pl_module, batch=None, batch_idx=0)
    runtime.before_backward(loss=None)
    runtime.after_backward()
    runtime.finish_train_batch(pl_module)
    return runtime.pipeline


def test_runtime_records_a_sample_per_batch_when_enabled() -> None:
    assert _run_one_batch(pipeline_pressure_enabled=True).latest() is not None


def test_runtime_records_no_sample_when_disabled() -> None:
    assert _run_one_batch(pipeline_pressure_enabled=False).latest() is None


def test_accumulated_batches_report_completed_optimizer_steps(tmp_path: Path) -> None:
    steps = []

    class Model(MidrunMixin, pl.LightningModule):
        def __init__(self) -> None:
            super().__init__(
                rest_enabled=False,
                fuse_enabled=False,
                torch_debug_enabled=False,
                hooks_enabled=False,
            )
            self.weight = torch.nn.Parameter(torch.ones(()))

        def training_step(self, batch, batch_idx):
            return self.weight * batch.mean()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

        def on_train_batch_end(self, outputs, batch, batch_idx):
            super().on_train_batch_end(outputs, batch, batch_idx)
            steps.append(self.midrun.pipeline.latest().global_step)

    trainer = pl.Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        accumulate_grad_batches=2,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    trainer.fit(Model(), DataLoader(torch.ones(4), batch_size=1))
    assert steps == [0, 1, 1, 2]
