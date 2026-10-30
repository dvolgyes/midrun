"""Behavioral tests for PipelinePressureTracker's phase timing and rolling summary."""

from trainctl.config import TrainctlConfig
from trainctl.load_backend import load_backend
from trainctl.runtime.pipeline_pressure import PipelinePressureTracker
from trainctl.runtime.runtime import TrainctlRuntime


def test_first_batch_has_no_input_gap() -> None:
    tracker = PipelinePressureTracker(window=8)

    tracker.batch_start(1_000, epoch=0, global_step=0, batch_idx=0)
    sample = tracker.batch_end(2_000)

    assert sample.input_gap_ns is None
    assert sample.batch_total_ns == 1_000


def test_second_batch_records_input_gap_from_previous_batch_end() -> None:
    tracker = PipelinePressureTracker(window=8)
    tracker.batch_start(1_000, epoch=0, global_step=0, batch_idx=0)
    tracker.batch_end(2_000)

    tracker.batch_start(2_500, epoch=0, global_step=1, batch_idx=1)
    sample = tracker.batch_end(3_000)

    assert sample.input_gap_ns == 500


def test_full_phase_breakdown_recorded_in_order() -> None:
    tracker = PipelinePressureTracker(window=8)
    tracker.batch_start(0, epoch=0, global_step=0, batch_idx=0)
    tracker.before_backward(100)
    tracker.after_backward(150)
    sample = tracker.batch_end(200)

    assert sample.forward_loss_ns == 100
    assert sample.backward_ns == 50
    assert sample.post_backward_ns == 50
    assert sample.batch_total_ns == 200


def test_missing_backward_phases_are_none() -> None:
    tracker = PipelinePressureTracker(window=8)
    tracker.batch_start(0, epoch=0, global_step=0, batch_idx=0)
    sample = tracker.batch_end(100)

    assert sample.forward_loss_ns is None
    assert sample.backward_ns is None
    assert sample.post_backward_ns is None


def test_window_evicts_oldest_samples() -> None:
    tracker = PipelinePressureTracker(window=2)
    for i in range(5):
        tracker.batch_start(i * 100, epoch=0, global_step=i, batch_idx=i)
        tracker.batch_end(i * 100 + 10)

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
    tracker.batch_start(0, epoch=0, global_step=0, batch_idx=0)
    tracker.before_backward(10)
    tracker.after_backward(1_000)
    sample = tracker.batch_end(1_010)

    assert sample.backward_ns == 990
    summary = tracker.summary()
    assert summary["largest_phase"] == "backward"


class _FakeTrainer:
    current_epoch = 0
    global_step = 0


class _FakePLModule:
    trainer = _FakeTrainer()


def _run_one_batch(pipeline_pressure_enabled: bool) -> PipelinePressureTracker:
    config = TrainctlConfig(
        rest_enabled=False,
        fuse_enabled=False,
        torch_debug_enabled=False,
        pipeline_pressure_enabled=pipeline_pressure_enabled,
    )
    runtime = TrainctlRuntime(config, load_backend("lightning.pytorch"))
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
