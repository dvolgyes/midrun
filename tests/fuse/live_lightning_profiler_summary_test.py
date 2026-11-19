"""Behavioral tests for the live model/lightning-profiler-summary.txt file.

A read blocks the reading thread until the training thread reaches
SafePoint.BEFORE_OPTIMIZER_STEP, and the summary is computed there, because the
Lightning profilers are not safe to read from another thread.
"""

import json
import threading
import time
from pathlib import Path

import pytest
import torch
from lightning.pytorch.profilers import (
    AdvancedProfiler,
    PassThroughProfiler,
    Profiler,
    PyTorchProfiler,
    SimpleProfiler,
)

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.fuse.filesystem import MidrunFS
from midrun.load_backend import load_backend
from midrun.runtime.command_specs.execute_command import execute_command
from midrun.runtime.commands.parse_when import parse_when
from midrun.runtime.events import SafePoint
from midrun.runtime.runtime import MidrunRuntime
from midrun.runtime.virtual_fs.virtual_file_content import virtual_file_content

_PATH = "/model/lightning-profiler-summary.txt"


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _FakeState:
    stage = None
    fn = "fit"


class _FakeTrainer:
    world_size = 2
    local_rank = 1
    num_training_batches = 6

    def __init__(self, profiler: Profiler) -> None:
        self.callback_metrics: dict[str, float] = {}
        self.current_epoch = 0
        self.global_step = 0
        self.optimizers: list[object] = []
        self.state = _FakeState()
        self.profiler = profiler


def _make_runtime(profiler: Profiler, level: str) -> tuple[MidrunRuntime, _Model]:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = build_runtime(config, load_backend("lightning.pytorch"))
    model = _Model()
    model.trainer = _FakeTrainer(profiler)
    runtime.lightning_profiler = profiler
    runtime.lightning_profiler_level = level
    return runtime, model


def _record_summary_threads(profiler: Profiler) -> list[int]:
    """Makes `profiler.summary` record the id of each thread that calls it."""
    threads: list[int] = []
    original = profiler.summary

    def summary() -> str:
        threads.append(threading.get_ident())
        return original()

    profiler.summary = summary  # type: ignore[method-assign]
    return threads


def _read_on_reader_thread(runtime: MidrunRuntime, model: _Model) -> bytes:
    """Reads the file on another thread while this thread drives one safe point."""
    result: dict[str, bytes] = {}
    reader = threading.Thread(
        target=lambda: result.update(content=virtual_file_content(runtime, _PATH)),
        daemon=True,
    )
    reader.start()
    deadline = time.monotonic() + 5.0
    while (
        reader.is_alive()
        and not runtime.live_captures._pending
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    if runtime.live_captures._pending:
        runtime.process_safe_point(model, SafePoint.BEFORE_OPTIMIZER_STEP)
    reader.join(timeout=5.0)
    assert "content" in result, "the read did not finish"
    return result["content"]


@pytest.mark.parametrize(
    ("profiler_class", "level"),
    [(SimpleProfiler, "simple"), (AdvancedProfiler, "advanced")],
)
def test_summary_is_computed_on_the_training_thread(
    profiler_class: type[Profiler], level: str
) -> None:
    profiler = profiler_class()
    profiler.start("early_action")
    profiler.stop("early_action")
    runtime, model = _make_runtime(profiler, level)
    threads = _record_summary_threads(profiler)

    text = _read_on_reader_thread(runtime, model).decode()

    assert threads == [threading.get_ident()]
    assert "early_action" in text
    assert runtime.lightning_profiler is profiler
    assert model.trainer.profiler is profiler
    profiler.start("later_action")
    profiler.stop("later_action")
    assert "later_action" in profiler.summary()


def test_pytorch_read_closes_active_scopes_and_reuses_effective_options() -> None:
    profiler = PyTorchProfiler(
        schedule=None,
        activities=[torch.profiler.ProfilerActivity.CPU],
        table_kwargs={"header": "configured summary"},
        row_limit=-1,
    )
    runtime, model = _make_runtime(profiler, "pytorch")
    threads = _record_summary_threads(profiler)

    with profiler.profile("first_action"):
        torch.ones(2).neg()
        text = _read_on_reader_thread(runtime, model).decode()
        assert not profiler._recording_map

    assert threads == [threading.get_ident()]
    assert "first_action" in text
    assert "configured summary" in text
    assert model.trainer.profiler is profiler
    assert runtime.lightning_profiler is profiler
    assert runtime.lightning_profiler_level == "pytorch"
    with profiler.profile("second_action"):
        torch.ones(2).abs()
    second = _read_on_reader_thread(runtime, model)
    assert b"configured summary" in second
    assert b"second_action" in second
    assert b"first_action" not in second


def test_read_leaves_external_trainer_profiler_untouched() -> None:
    managed = PyTorchProfiler()
    runtime, model = _make_runtime(managed, "pytorch")
    external = PyTorchProfiler(
        schedule=None, activities=[torch.profiler.ProfilerActivity.CPU], row_limit=-1
    )
    model.trainer.profiler = external
    threads = _record_summary_threads(managed)

    with external.profile("externally_owned"):
        assert b"not enabled" in _read_on_reader_thread(runtime, model)
        torch.ones(2).neg()
    assert "externally_owned" in external.summary()
    assert model.trainer.profiler is external
    assert not threads


@pytest.mark.parametrize("num_batches", [2, 6])
def test_second_recording_exports_ranked_trace_with_module_scopes(
    tmp_path: Path, num_batches: int
) -> None:
    original = PassThroughProfiler()
    original.setup(stage="fit", local_rank=1, log_dir=str(tmp_path))
    runtime, model = _make_runtime(original, "off")
    model.trainer.num_training_batches = num_batches
    command = runtime.command_queue.submit(
        "set_lightning_profiler", {"level": "pytorch"}, parse_when("now")
    )
    execute_command(runtime, model, model.trainer, command, safe_point=None)
    assert command.status == "succeeded"
    for _ in range(model.trainer.num_training_batches):
        with model.trainer.profiler.profile("first_training_step"):
            model(torch.ones(1, 4))
    _read_on_reader_thread(runtime, model)

    profiler = model.trainer.profiler
    for _ in range(model.trainer.num_training_batches):
        with profiler.profile("second_training_step"):
            model(torch.ones(1, 4))
    _read_on_reader_thread(runtime, model)

    (trace,) = tmp_path.glob("fit-1-second_training_step.*.pt.trace.json")
    events = json.loads(trace.read_bytes())["traceEvents"]
    assert any(
        event.get("name") == "[pl][module]torch.nn.modules.linear.Linear: fc"
        for event in events
    )


def test_fuse_getattr_does_not_compute_the_summary() -> None:
    profiler = SimpleProfiler()
    runtime, _model = _make_runtime(profiler, "simple")
    threads = _record_summary_threads(profiler)

    attrs = MidrunFS(runtime).getattr(_PATH)

    assert attrs["st_size"] == 0
    assert threads == []
