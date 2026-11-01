"""Behavioral tests for the bounded profiler run and its runtime integration."""

import json
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

import lightning.pytorch as PL
import pytest
import torch
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.mixin import MidrunMixin
from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.command_specs.execute_command import execute_command
from midrun.runtime.commands.parse_when import parse_when
from midrun.runtime.profiler import ProfilerRun
from midrun.runtime.runtime import MidrunRuntime


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
    run.start("cpu")
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

    files, metadata = run.finalize()
    assert "trace.json" in files
    assert "summary.txt" in files
    assert len(files["trace.json"]) > 0
    assert metadata == {}


def test_a_failed_summary_still_saves_the_trace_and_records_the_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_key_averages(*args: object, **kwargs: object) -> None:
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(torch.profiler.profile, "key_averages", fail_key_averages)
    store = ArtifactStore(tmp_path)
    run = ProfilerRun(
        "p1", warmup_steps=1, active_steps=1, level="basic", then_hold=False
    )
    run.start("cpu")
    x = torch.randn(8, 8)
    while not run.step():
        _ = x @ x

    run.save(store)

    (capture_dir,) = (store.root / "debug").iterdir()
    manifest = json.loads((capture_dir / "manifest.json").read_text())
    assert manifest["metadata"]["summary_error"] == (
        "RecursionError: maximum recursion depth exceeded"
    )
    assert "aten::mm" in _traced_op_names(capture_dir)
    assert not (capture_dir / "summary.txt").exists()


def test_profile_command_arms_immediately(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    pl_module = _FakePLModule(trainer)

    cmd = runtime.command_queue.submit(
        "profile",
        {"warmup_steps": 0, "active_steps": 1, "level": "basic"},
        parse_when("now"),
    )
    execute_command(runtime, pl_module, trainer, cmd, safe_point=None)

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
    execute_command(runtime, pl_module, trainer, first, safe_point=None)
    assert first.status == "succeeded"
    try:
        second = runtime.command_queue.submit(
            "profile", {"active_steps": 1}, parse_when("now")
        )
        execute_command(runtime, pl_module, trainer, second, safe_point=None)

        assert second.status == "failed"
        assert "already armed" in second.error
    finally:
        # torch.profiler's underlying session is process-global; leaving the first
        # run started would corrupt later tests in this process.
        runtime.profiler_run.stop()


def test_finish_train_batch_advances_and_finalizes_profiler(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    pl_module = _FakePLModule(trainer)

    cmd = runtime.command_queue.submit(
        "profile",
        {"warmup_steps": 0, "active_steps": 2, "level": "basic"},
        parse_when("now"),
    )
    execute_command(runtime, pl_module, trainer, cmd, safe_point=None)
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


def test_a_cpu_clock_that_differs_between_cpus_disables_the_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(feature: str) -> None:
        raise RuntimeError(f"{feature} is disabled: the clocks differ")

    monkeypatch.setattr(
        "midrun.runtime.handlers.handle_profile.require_steady_cpu_clock", refuse
    )
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    pl_module = _FakePLModule(trainer)

    cmd = runtime.command_queue.submit(
        "profile",
        {"warmup_steps": 1, "active_steps": 1, "level": "basic"},
        parse_when("now"),
    )
    execute_command(runtime, pl_module, trainer, cmd, safe_point=None)

    assert cmd.status == "failed"
    assert "profile is disabled" in cmd.error
    assert runtime.profiler_run is None


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
        execute_command(runtime, pl_module, trainer, cmd, safe_point=None)
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
    execute_command(runtime, pl_module, trainer, cmd, safe_point=None)
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


class _ProfiledModel(MidrunMixin, PL.LightningModule):
    def __init__(self, profile_args: dict[str, object], artifact_dir: Path) -> None:
        super().__init__(
            rest_enabled=False,
            fuse_enabled=False,
            torch_debug_enabled=False,
            hooks_enabled=False,
            artifact_dir=artifact_dir,
        )
        self.layer = torch.nn.Linear(4, 1)
        self.profile_args = profile_args

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.midrun.command_queue.submit(
            "profile", self.profile_args, parse_when("now")
        )

    def training_step(self, batch, batch_idx):
        return self.layer(batch[0]).sum()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _fit_with_profile(tmp_path: Path, profile_args: dict[str, object]) -> MidrunRuntime:
    model = _ProfiledModel(profile_args, tmp_path / "artifacts")
    trainer = PL.Trainer(
        default_root_dir=str(tmp_path),
        max_steps=3,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )
    loader = DataLoader(TensorDataset(torch.randn(16, 4)), batch_size=4)
    trainer.fit(model, loader)
    return model.midrun


def test_run_end_stops_an_active_profile_and_saves_it_as_truncated(
    tmp_path: Path,
) -> None:
    runtime = _fit_with_profile(tmp_path, {"warmup_steps": 0, "active_steps": 50})

    assert not torch.autograd.profiler._is_profiler_enabled
    assert runtime.profiler_run is None
    (capture_dir,) = (runtime.artifact_store.root / "debug").iterdir()
    manifest = json.loads((capture_dir / "manifest.json").read_text())
    assert manifest["kind"] == "profiler"
    assert manifest["metadata"]["truncated"] is True
    assert manifest["metadata"]["recorded_steps"] == 2
    assert "aten::linear" in _traced_op_names(capture_dir)


def test_run_end_before_a_profile_records_stops_it_and_warns(tmp_path: Path) -> None:
    warnings: list[str] = []
    sink_id = logger.add(warnings.append, level="WARNING", format="{message}")
    try:
        runtime = _fit_with_profile(tmp_path, {"warmup_steps": 10, "active_steps": 1})
    finally:
        logger.remove(sink_id)

    assert not torch.autograd.profiler._is_profiler_enabled
    assert runtime.profiler_run is None
    assert not list((runtime.artifact_store.root / "debug").iterdir())
    assert any("nothing was saved" in message for message in warnings)


def test_default_profile_runs_with_a_warmup(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    pl_module = _FakePLModule(trainer)
    cmd = runtime.command_queue.submit("profile", {}, parse_when("now"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        execute_command(runtime, pl_module, trainer, cmd, safe_point=None)
    runtime.profiler_run.stop()

    assert runtime.command_queue.get(cmd.id).status == "succeeded"
    assert runtime.profiler_run.warmup_steps >= 1
    assert not [w for w in caught if "warmup" in str(w.message)]


def test_failed_profiler_step_stops_the_profiler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    pl_module = _FakePLModule(trainer)
    cmd = runtime.command_queue.submit(
        "profile", {"warmup_steps": 0, "active_steps": 5}, parse_when("now")
    )
    execute_command(runtime, pl_module, trainer, cmd, safe_point=None)
    runtime.finish_train_batch(pl_module)
    assert torch.autograd.profiler._is_profiler_enabled

    def _fail() -> None:
        raise RuntimeError("step broke")

    monkeypatch.setattr(runtime.profiler_run._prof, "step", _fail)
    runtime.begin_train_batch(pl_module, batch={"x": 1}, batch_idx=1)
    runtime.finish_train_batch(pl_module)

    assert runtime.profiler_run is None
    assert not torch.autograd.profiler._is_profiler_enabled


# torch.profiler fixes the CUDA activity at the first profile of a process, so each
# device case runs in a fresh process.
_PROFILE_ON_A_RANK = """
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.command_specs.execute_command import execute_command
from midrun.runtime.commands.parse_when import parse_when
from midrun.runtime.local_rank_info import local_rank_info

store_dir, device = Path(sys.argv[1]), sys.argv[2]
runtime = build_runtime(
    MidrunConfig(rest_enabled=False, torch_debug_enabled=False, fuse_enabled=False),
    load_backend("lightning.pytorch"),
)
runtime.artifacts = ArtifactStore(store_dir)
runtime.rank_info = local_rank_info(device)
trainer = SimpleNamespace(
    callback_metrics={}, current_epoch=0, global_step=0, optimizers=[],
    state=SimpleNamespace(stage=None),
)
cmd = runtime.command_queue.submit(
    "profile", {"warmup_steps": 0, "active_steps": 1}, parse_when("now")
)
execute_command(runtime, SimpleNamespace(trainer=trainer), trainer, cmd, safe_point=None)
x = torch.randn(64, 64, device=device)
while not runtime.profiler_run.step():
    _ = x @ x
runtime.profiler_run.save(runtime.artifact_store)
(capture_dir,) = (store_dir / "debug").iterdir()
cuda_recorded = "CUDA total" in (capture_dir / "summary.txt").read_text()
print(cmd.status, torch.cuda.is_initialized(), cuda_recorded)
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize(
    ("device", "cuda_used"), [("cpu", "False"), ("cuda:0", "True")]
)
def test_a_profile_records_cuda_activity_only_on_a_cuda_rank(
    tmp_path: Path, device: str, cuda_used: str
) -> None:
    result = subprocess.run(  # noqa: S603 -- fixed command, test-owned script
        [sys.executable, "-c", _PROFILE_ON_A_RANK, str(tmp_path), device],
        capture_output=True,
        check=True,
        text=True,
    )

    # A CPU rank must not create a CUDA context; a CUDA rank must still see CUDA.
    assert result.stdout.split() == ["succeeded", cuda_used, cuda_used]
