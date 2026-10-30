"""Behavioral tests for the live model/lightning-profiler-summary.txt file.

A read blocks the reading thread until the training thread reaches
SafePoint.BEFORE_OPTIMIZER_STEP, and the summary is computed there, because the
Lightning profilers are not safe to read from another thread.
"""

import threading
import time

import pytest
import torch
from lightning.pytorch.profilers import (
    AdvancedProfiler,
    Profiler,
    PyTorchProfiler,
    SimpleProfiler,
)

from midrun.config import MidrunConfig
from midrun.fuse.filesystem import MidrunFS
from midrun.load_backend import load_backend
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


class _FakeTrainer:
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
    runtime = MidrunRuntime(config, load_backend("lightning.pytorch"))
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


def test_pytorch_level_read_installs_a_fresh_profiler() -> None:
    profiler = PyTorchProfiler()
    runtime, model = _make_runtime(profiler, "pytorch")
    threads = _record_summary_threads(profiler)

    text = _read_on_reader_thread(runtime, model).decode()

    assert threads == [threading.get_ident()]
    assert "reset" in text
    fresh = model.trainer.profiler
    assert isinstance(fresh, PyTorchProfiler)
    assert fresh is not profiler
    assert runtime.lightning_profiler is fresh
    assert runtime.lightning_profiler_level == "pytorch"


def test_fuse_getattr_does_not_compute_the_summary() -> None:
    profiler = SimpleProfiler()
    runtime, _model = _make_runtime(profiler, "simple")
    threads = _record_summary_threads(profiler)

    attrs = MidrunFS(runtime).getattr(_PATH)

    assert attrs["st_size"] == 0
    assert threads == []
