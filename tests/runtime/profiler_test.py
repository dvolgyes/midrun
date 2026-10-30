"""Behavioral tests for the bounded profiler run and its runtime integration."""

import json
import threading
import time
from pathlib import Path

import pytest
import torch

from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.commands.parse_when import parse_when
from midrun.runtime.profiler import ProfilerRun
from midrun.runtime.runtime import MidrunRuntime


def _make_runtime(tmp_path: Path, **overrides: object) -> MidrunRuntime:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False, **overrides
    )
    runtime = MidrunRuntime(config, load_backend("lightning.pytorch"))
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


class _FakePLModule:
    def __init__(self, trainer: _FakeTrainer) -> None:
        self.trainer = trainer


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert predicate(), "condition not met before timeout"


def test_profiler_run_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="level"):
        ProfilerRun("p1", 0, 1, "nonsense", False)


def test_profiler_run_rejects_invalid_step_counts() -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        ProfilerRun("p1", -1, 1, "basic", False)
    with pytest.raises(ValueError, match="active_steps"):
        ProfilerRun("p1", 0, 0, "basic", False)


def test_profiler_run_completes_after_warmup_plus_active_steps() -> None:
    run = ProfilerRun(
        "p1", warmup_steps=1, active_steps=2, level="basic", then_hold=False
    )
    run.start()
    x = torch.randn(8, 8)
    # Ends the batch in which the run started; it is not a counted step.
    assert run.step() is False
    _ = x @ x
    assert run.step() is False
    _ = x @ x
    assert run.step() is False
    _ = x @ x
    assert run.step() is True
    _ = x @ x

    files = run.finalize()
    assert "trace.json" in files
    assert "summary.txt" in files
    assert len(files["trace.json"]) > 0


def test_profile_command_arms_immediately(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    pl_module = _FakePLModule(trainer)

    cmd = runtime.command_queue.submit(
        "profile",
        {"warmup_steps": 0, "active_steps": 1, "level": "basic"},
        parse_when("now"),
    )
    runtime._execute(pl_module, trainer, cmd, safe_point=None)

    assert cmd.status == "succeeded"
    assert cmd.result["armed"] is True
    assert runtime.profiler_run is not None


def test_profile_command_rejects_second_concurrent_arm(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    pl_module = _FakePLModule(trainer)

    first = runtime.command_queue.submit(
        "profile", {"active_steps": 5}, parse_when("now")
    )
    runtime._execute(pl_module, trainer, first, safe_point=None)
    assert first.status == "succeeded"
    try:
        second = runtime.command_queue.submit(
            "profile", {"active_steps": 1}, parse_when("now")
        )
        runtime._execute(pl_module, trainer, second, safe_point=None)

        assert second.status == "failed"
        assert "already armed" in second.error
    finally:
        # torch.profiler's underlying session is process-global; leaving the first
        # run started would corrupt later tests in this process. One step ends the
        # arming batch, so the run has recorded something to finalize.
        runtime.profiler_run.step()
        runtime.profiler_run.finalize()


def test_finish_train_batch_advances_and_finalizes_profiler(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    pl_module = _FakePLModule(trainer)

    cmd = runtime.command_queue.submit(
        "profile",
        {"warmup_steps": 0, "active_steps": 2, "level": "basic"},
        parse_when("now"),
    )
    runtime._execute(pl_module, trainer, cmd, safe_point=None)
    assert runtime.profiler_run is not None
    runtime.finish_train_batch(pl_module)
    assert runtime.profiler_run is not None

    runtime.begin_train_batch(pl_module, batch={"x": 1}, batch_idx=1)
    runtime.finish_train_batch(pl_module)
    assert runtime.profiler_run is not None

    runtime.begin_train_batch(pl_module, batch={"x": 1}, batch_idx=2)
    runtime.finish_train_batch(pl_module)
    assert runtime.profiler_run is None

    debug_dirs = list((runtime.artifacts.root / "debug").iterdir())
    assert len(debug_dirs) == 1


def test_finish_train_batch_with_then_hold_enters_cooperative_hold(
    tmp_path: Path,
) -> None:
    # torch.profiler's CUDA backend is thread-affine (start/step/stop must happen on
    # the same thread), matching real usage where the training thread alone calls all
    # three -- so this drives arm+begin+finish from one background thread, exactly
    # like production, and only waits for the resulting hold from the main thread.
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    pl_module = _FakePLModule(trainer)
    runtime._gather_ranks(pl_module)

    def _drive() -> None:
        cmd = runtime.command_queue.submit(
            "profile",
            {
                "warmup_steps": 0,
                "active_steps": 1,
                "level": "basic",
                "then_hold": True,
            },
            parse_when("now"),
        )
        runtime._execute(pl_module, trainer, cmd, safe_point=None)
        runtime.finish_train_batch(pl_module)
        runtime.begin_train_batch(pl_module, batch={"x": 1}, batch_idx=1)
        runtime.finish_train_batch(pl_module)

    thread = threading.Thread(target=_drive)
    thread.start()
    try:
        _wait_until(lambda: runtime.gate.is_held())
        assert runtime.profiler_run is None
    finally:
        runtime.gate.release()
        thread.join(timeout=5)
        assert not thread.is_alive()


def _traced_op_names(capture_dir: Path) -> set[str]:
    trace = json.loads((capture_dir / "trace.json").read_text())
    return {event.get("name") for event in trace["traceEvents"]}


def test_profile_armed_mid_batch_records_the_next_full_batch(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    pl_module = _FakePLModule(trainer)
    x = torch.randn(4, 8)
    weight = torch.randn(8, 8)

    # The command runs at a safe point inside batch 0, e.g. before the optimizer step
    # or in a hold there; the rest of batch 0 is not a counted step.
    cmd = runtime.command_queue.submit(
        "profile",
        {"warmup_steps": 0, "active_steps": 1, "level": "basic"},
        parse_when("now"),
    )
    runtime._execute(pl_module, trainer, cmd, safe_point=None)
    _ = torch.cumsum(x, dim=0)
    runtime.finish_train_batch(pl_module)
    assert runtime.profiler_run is not None

    runtime.begin_train_batch(pl_module, batch={"x": 1}, batch_idx=1)
    _ = torch.nn.functional.linear(x, weight)
    runtime.finish_train_batch(pl_module)

    assert runtime.profiler_run is None
    (capture_dir,) = (runtime.artifacts.root / "debug").iterdir()
    names = _traced_op_names(capture_dir)
    assert "aten::linear" in names
    assert "aten::cumsum" not in names
