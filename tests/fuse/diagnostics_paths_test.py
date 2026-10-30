"""Behavioral tests for the diagnostics-milestone static FUSE paths.

Exercises `virtual_file_content`/`is_virtual_dir` directly against a `MidrunRuntime`,
matching the existing static-path test convention -- no real FUSE mount involved.
"""

import json

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.runtime.runtime import MidrunRuntime
from midrun.runtime.state.exception_breakpoint_info import ExceptionBreakpointInfo
from midrun.runtime.virtual_fs.is_virtual_dir import is_virtual_dir
from midrun.runtime.virtual_fs.virtual_file_content import virtual_file_content


def _make_runtime(**overrides: object) -> MidrunRuntime:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False, **overrides
    )
    return build_runtime(config, load_backend("lightning.pytorch"))


def test_breakpoint_json_is_null_when_no_exception_held() -> None:
    runtime = _make_runtime()
    assert json.loads(virtual_file_content(runtime, "/state/breakpoint.json")) is None


def test_breakpoint_json_reports_active_exception() -> None:
    runtime = _make_runtime()
    info = ExceptionBreakpointInfo(
        exception_type="ValueError",
        exception_message="boom",
        epoch=0,
        global_step=1,
        batch_idx=2,
        rank=0,
        started_at=123.0,
    )
    runtime.state.update(status="exception_held", exception=info)

    payload = json.loads(virtual_file_content(runtime, "/state/breakpoint.json"))

    assert payload["exception_type"] == "ValueError"
    assert payload["exception_message"] == "boom"


def test_dataloader_finite_check_json_reflects_guard_state() -> None:
    runtime = _make_runtime(dataloader_finite_check_enabled=True)
    payload = json.loads(
        virtual_file_content(runtime, "/state/dataloader/finite-check.json")
    )
    assert payload == {"enabled": True, "checked_batches": 0, "failures": 0}
    assert is_virtual_dir(runtime, "/state/dataloader")


def test_pipeline_config_json_reflects_config() -> None:
    runtime = _make_runtime(pipeline_pressure_enabled=True, pipeline_pressure_window=64)
    payload = json.loads(virtual_file_content(runtime, "/state/pipeline/config.json"))
    assert payload == {"enabled": True, "window": 64}
    assert is_virtual_dir(runtime, "/state/pipeline")


def test_pipeline_latest_json_is_null_before_first_batch() -> None:
    runtime = _make_runtime()
    assert (
        json.loads(virtual_file_content(runtime, "/state/pipeline/latest.json")) is None
    )


def test_pipeline_latest_and_summary_json_after_a_batch() -> None:
    runtime = _make_runtime()
    runtime.pipeline.batch_start(0, epoch=0, global_step=0, batch_idx=0)
    runtime.pipeline.batch_end(1_000)

    latest = json.loads(virtual_file_content(runtime, "/state/pipeline/latest.json"))
    assert latest["batch_total_ns"] == 1_000

    summary = json.loads(virtual_file_content(runtime, "/state/pipeline/summary.json"))
    assert summary["samples"] == 1


def test_pytorch_debug_enabled_is_false_until_the_server_runs() -> None:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=True
    )
    runtime = build_runtime(config, load_backend("lightning.pytorch"))

    enabled = virtual_file_content(runtime, "/meta/pytorch-debug/enabled")

    assert enabled == b"false\n"
