"""Behavioral tests for the inspector-integration static FUSE paths.

Exercises `virtual_file_content`/`is_virtual_dir` directly against a `MidrunRuntime`,
matching the existing static-path test convention -- no real FUSE mount involved.
"""

import json

from midrun.config import MidrunConfig
from midrun.fuse.filesystem import MidrunFS
from midrun.load_backend import load_backend
from midrun.runtime.cuda_memory.cuda_memory_history_state import (
    CudaMemoryHistoryState,
)
from midrun.runtime.profiler import ProfilerRun
from midrun.runtime.runtime import MidrunRuntime
from midrun.runtime.virtual_fs.is_virtual_dir import is_virtual_dir
from midrun.runtime.virtual_fs.virtual_file_content import virtual_file_content


def _make_fs(**overrides: object) -> MidrunFS:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False, **overrides
    )
    runtime = MidrunRuntime(config, load_backend("lightning.pytorch"))
    return MidrunFS(runtime)


def test_model_summary_json_reports_param_counts() -> None:
    fs = _make_fs()
    fs._runtime._model_summary_json = {
        "total_params": 10,
        "trainable_params": 10,
        "non_trainable_params": 0,
        "torchinfo_used": False,
    }
    payload = json.loads(virtual_file_content(fs._runtime, "/model/summary.json"))
    assert payload["total_params"] == 10


def test_model_inspectors_json_lists_all_four() -> None:
    fs = _make_fs()
    payload = json.loads(virtual_file_content(fs._runtime, "/model/inspectors.json"))
    assert set(payload) == {"torchinfo", "torchview", "torchviz", "torchlens"}


def test_state_cuda_memory_json_reflects_history_state() -> None:
    fs = _make_fs()
    fs._runtime.cuda_memory_history = CudaMemoryHistoryState(
        enabled=True, max_entries=500, started_at=123.0
    )
    payload = json.loads(virtual_file_content(fs._runtime, "/state/cuda-memory.json"))
    assert payload["enabled"] is True
    assert payload["max_entries"] == 500


def test_state_profiler_json_is_idle_by_default() -> None:
    fs = _make_fs()
    payload = json.loads(virtual_file_content(fs._runtime, "/state/profiler.json"))
    assert payload == {"state": "idle"}


def test_state_profiler_json_reflects_active_run() -> None:
    fs = _make_fs()
    run = ProfilerRun(
        "p1", warmup_steps=0, active_steps=2, level="basic", then_hold=False
    )
    fs._runtime.profiler_run = run
    run.start()
    try:
        waiting = json.loads(virtual_file_content(fs._runtime, "/state/profiler.json"))
        # The arming batch ends with the wait step.
        run.step()
        active = json.loads(virtual_file_content(fs._runtime, "/state/profiler.json"))
    finally:
        run.stop()

    assert waiting["state"] == "waiting"
    assert active["state"] == "active"
    assert active["active_steps"] == 2


def test_state_lightning_profiler_json_is_off_by_default() -> None:
    fs = _make_fs()
    payload = json.loads(
        virtual_file_content(fs._runtime, "/state/lightning-profiler.json")
    )
    assert payload == {"level": "off"}


def test_model_lightning_profiler_summary_reports_not_enabled_by_default() -> None:
    fs = _make_fs()
    text = virtual_file_content(fs._runtime, "/model/lightning-profiler-summary.txt")
    assert b"not enabled" in text


def test_model_lightning_profiler_summary_reflects_installed_profiler() -> None:
    from lightning.pytorch.profilers import SimpleProfiler

    fs = _make_fs()
    profiler = SimpleProfiler()
    profiler.start("some_action")
    profiler.stop("some_action")
    fs._runtime.lightning_profiler = profiler
    fs._runtime.lightning_profiler_level = "simple"

    text = virtual_file_content(
        fs._runtime, "/model/lightning-profiler-summary.txt"
    ).decode()

    assert "some_action" in text


def test_readme_only_leaf_directories_are_not_projected() -> None:
    fs = _make_fs()
    readme = virtual_file_content(fs._runtime, "/README.md").decode()
    assert "model/" in readme
    assert not is_virtual_dir(fs._runtime, "/model/graph")
    assert not is_virtual_dir(fs._runtime, "/debug/cuda-memory")
    assert not is_virtual_dir(fs._runtime, "/debug/profiler")
    assert not is_virtual_dir(fs._runtime, "/debug/torchlens")
