"""Behavioral tests for the `set_lightning_profiler` command handler."""

from pathlib import Path

import pytest
import torch
from lightning.pytorch.profilers import PassThroughProfiler, PyTorchProfiler

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.command_specs.execute_command import execute_command
from midrun.runtime.commands.parse_when import parse_when
from midrun.runtime.runtime import MidrunRuntime


def _make_runtime(tmp_path: Path) -> MidrunRuntime:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = build_runtime(config, load_backend("lightning.pytorch"))
    runtime.artifacts = ArtifactStore(tmp_path)
    return runtime


class _FakeState:
    stage = None
    fn = "fit"


class _FakeTrainer:
    def __init__(self, log_dir: Path | None = None) -> None:
        self.callback_metrics: dict[str, float] = {}
        self.current_epoch = 0
        self.global_step = 0
        self.optimizers: list[object] = []
        self.state = _FakeState()
        self.world_size = 1
        self.local_rank = 0
        # As the Trainer does at stage setup.
        self.profiler = PassThroughProfiler()
        self.profiler.setup(
            stage="fit", log_dir=None if log_dir is None else str(log_dir)
        )
        self.lightning_module = _FakePLModule(self)


class _FakePLModule:
    def __init__(self, trainer: _FakeTrainer) -> None:
        self.trainer = trainer

    def named_modules(self):
        return iter(())


def _set_level(runtime: MidrunRuntime, trainer: _FakeTrainer, level: str):
    cmd = runtime.command_queue.submit(
        "set_lightning_profiler", {"level": level}, parse_when("now")
    )
    execute_command(runtime, trainer.lightning_module, trainer, cmd, safe_point=None)
    return cmd


@pytest.mark.parametrize(
    ("level", "class_name"),
    [
        ("simple", "SimpleProfiler"),
        ("advanced", "AdvancedProfiler"),
        ("pytorch", "PyTorchProfiler"),
    ],
)
def test_set_lightning_profiler_installs_the_requested_class(
    tmp_path: Path, level: str, class_name: str
) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()

    cmd = _set_level(runtime, trainer, level)

    assert cmd.status == "succeeded"
    assert cmd.result == {"level": level}
    assert type(trainer.profiler).__name__ == class_name
    assert trainer.profiler is runtime.lightning_profiler
    assert runtime.lightning_profiler_level == level


def test_set_lightning_profiler_off_restores_pass_through(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    _set_level(runtime, trainer, "simple")

    cmd = _set_level(runtime, trainer, "off")

    assert cmd.status == "succeeded"
    assert type(trainer.profiler).__name__ == "PassThroughProfiler"
    assert runtime.lightning_profiler is None
    assert runtime.lightning_profiler_level == "off"


def test_set_lightning_profiler_rejects_unknown_level(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)

    with pytest.raises(ValueError, match="level"):
        runtime.command_queue.submit(
            "set_lightning_profiler", {"level": "nonsense"}, parse_when("now")
        )

    assert runtime.command_queue.list_all() == []


def test_set_lightning_profiler_reselecting_same_level_resets_accumulated_state(
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    _set_level(runtime, trainer, "simple")
    runtime.lightning_profiler.start("some_action")
    runtime.lightning_profiler.stop("some_action")
    assert "some_action" in runtime.lightning_profiler.summary()

    _set_level(runtime, trainer, "simple")

    assert "some_action" not in runtime.lightning_profiler.summary()


def test_set_lightning_profiler_sets_up_the_new_profiler_like_the_trainer(
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer(log_dir=tmp_path / "logs")

    _set_level(runtime, trainer, "simple")

    assert trainer.profiler.dirpath == str(tmp_path / "logs")
    assert trainer.profiler._stage == "fit"


def test_replacing_a_pytorch_profiler_stops_its_torch_profiler(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer(log_dir=tmp_path)
    # No schedule: the torch profiler records from the first start. CPU only, so
    # the test does not depend on CUPTI.
    old = PyTorchProfiler(
        schedule=None, activities=[torch.profiler.ProfilerActivity.CPU]
    )
    old.setup(stage="fit", log_dir=str(tmp_path))
    trainer.profiler = old
    old.start("training_step")
    assert torch.autograd._profiler_enabled()

    cmd = _set_level(runtime, trainer, "off")

    assert cmd.status == "succeeded"
    assert not torch.autograd._profiler_enabled()


def test_a_pytorch_profiler_that_never_recorded_can_be_replaced(
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer(log_dir=tmp_path)
    _set_level(runtime, trainer, "pytorch")
    # The default schedule waits one step, so nothing records yet.
    trainer.profiler.start("training_step")

    cmd = _set_level(runtime, trainer, "off")

    assert cmd.status == "succeeded"
    assert type(trainer.profiler).__name__ == "PassThroughProfiler"


def test_a_cpu_clock_that_differs_between_cpus_disables_only_the_pytorch_profiler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(feature: str) -> None:
        raise RuntimeError(f"{feature} is disabled: the clocks differ")

    monkeypatch.setattr(
        "midrun.runtime.handlers.handle_set_lightning_profiler.require_steady_cpu_clock",
        refuse,
    )
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    _set_level(runtime, trainer, "simple")
    simple = trainer.profiler

    refused = _set_level(runtime, trainer, "pytorch")

    assert refused.status == "failed"
    assert "the pytorch Lightning profiler is disabled" in refused.error
    assert trainer.profiler is simple
    assert runtime.lightning_profiler_level == "simple"
    assert _set_level(runtime, trainer, "advanced").status == "succeeded"
