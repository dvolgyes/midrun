"""Stage cleanup continues after failures and retries only unfinished operations."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.runtime import runtime as runtime_module
from midrun.runtime.commands.parse_when import parse_when
from midrun.runtime.profiler import ProfilerRun


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("late_enable", [False, True])
def test_failed_services_recheck_cuda_after_rest_joins(
    tmp_path, monkeypatch, strict, late_enable
):
    runtime = build_runtime(
        MidrunConfig(
            inspection_strict=strict,
            rest_enabled=False,
            fuse_enabled=False,
            torch_debug_enabled=False,
            artifact_dir=tmp_path / "artifacts",
        ),
        load_backend("lightning.pytorch"),
    )
    module = SimpleNamespace(device="cpu")
    runtime.setup_environment(module, "fit")
    command = runtime.command_queue.submit("stop", {}, parse_when("now"))
    runtime.gate.enter("hold", "train_batch_start", 0, 0)
    calls = []

    def stop(name):
        assert not runtime.gate.is_held()
        assert command.status == "expired"
        calls.append(name)
        if name in {"REST", "profiler"} and calls.count(name) == 1:
            raise RuntimeError(f"{name} unavailable")

    def disable_history(enabled, max_entries):
        stop("CUDA")
        return SimpleNamespace(enabled=False)

    monkeypatch.setattr(runtime.services, "stop_rest", lambda: stop("REST"))
    runtime.cuda_memory_history = SimpleNamespace(enabled=True)
    monkeypatch.setattr(runtime_module, "set_cuda_memory_history", disable_history)
    monkeypatch.setattr(runtime.services, "stop_fuse", lambda: stop("FUSE"))
    monkeypatch.setattr(runtime.torch_debug, "stop", lambda: stop("debug"))
    runtime.profiler_run = SimpleNamespace(save=lambda store: stop("profiler"))
    monkeypatch.setattr(runtime.rank_sync, "close", lambda: stop("group"))

    expected = (
        pytest.raises(RuntimeError, match="REST unavailable")
        if strict
        else nullcontext()
    )
    with expected:
        runtime.stop_after_failure()
    assert calls == ["REST", "CUDA", "FUSE", "debug", "profiler", "group"]
    assert runtime.state.read().status == "stopping"
    assert runtime.profiler_run is not None
    with pytest.raises(RuntimeError, match="shutdown"):
        runtime.setup_environment(module, "test")

    assert not runtime.cuda_memory_history.enabled
    if late_enable:
        runtime.cuda_memory_history = SimpleNamespace(enabled=True)
    runtime.teardown(module, "fit")
    runtime.teardown(module, "fit")
    expected_retry = (
        ["REST", "CUDA", "profiler"] if late_enable else ["REST", "profiler"]
    )
    assert calls[6:] == expected_retry
    assert not runtime.cuda_memory_history.enabled
    assert runtime.state.read().status == "failed"
    assert runtime.profiler_run is None
    runtime.setup_environment(module, "test")
    assert runtime.state.read().status == "running"
    runtime.teardown(module, "test")


def test_retry_after_capture_write_failure_preserves_the_export():
    run = ProfilerRun("profile", 0, 1, "basic", False)
    run.steps_done = 2
    calls = []

    def export(path):
        calls.append("export")
        Path(path).write_bytes(b"trace")

    def write_capture(kind, files, metadata, *, artifact_id):
        assert files["trace.json"] == b"trace"
        assert artifact_id == run.capture_id
        calls.append("write")
        if calls.count("write") == 1:
            raise OSError("write failed")

    run._prof = SimpleNamespace(
        stop=lambda: calls.append("stop"),
        export_chrome_trace=export,
        key_averages=lambda: SimpleNamespace(table=lambda: "summary"),
    )
    store = SimpleNamespace(write_debug_capture=write_capture)
    with pytest.raises(OSError, match="write failed"):
        run.save(store)
    run.save(store)
    assert calls == ["stop", "export", "write", "write"]
