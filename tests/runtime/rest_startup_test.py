"""Behavioral tests: REST starts with default options and warns of a stuck stop."""

import json
import socket
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx2
import pytest
import torch.distributed as dist
from loguru import logger

from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.runtime.runtime import MidrunRuntime
from midrun.runtime.virtual_fs.virtual_file_content import virtual_file_content


class _FakeModule:
    device = "cpu"


def test_rest_starts_without_distributed_process_group(tmp_path: Path) -> None:
    assert not dist.is_initialized()
    config = MidrunConfig(
        fuse_enabled=False,
        artifact_dir=tmp_path / "artifacts",
        rest_port=18490,
        inspection_strict=True,
    )
    assert config.rest_enabled and config.torch_debug_enabled
    runtime = MidrunRuntime(config, load_backend("lightning.pytorch"))
    module = _FakeModule()

    runtime.setup_environment(module, "fit")
    try:
        assert runtime.rest_url is not None
        health = httpx2.get(f"{runtime.rest_url}/health", timeout=5)
        assert health.status_code == 200
        assert health.json()["run_id"] == runtime.run_id
        assert runtime.torch_debug.frontend_url is None
    finally:
        runtime.teardown(module, "fit")


def test_rest_starts_when_the_paired_torch_debug_server_fails(
    tmp_path: Path, monkeypatch
) -> None:
    from midrun.runtime import runtime as runtime_module

    def _failing_start(port: int | None = None) -> None:
        raise RuntimeError(f"debug server cannot start on {port}")

    monkeypatch.setattr(runtime_module, "distributed_initialized", lambda: True)
    monkeypatch.setattr(runtime_module, "is_rank_zero", lambda: True)
    config = MidrunConfig(
        fuse_enabled=False, artifact_dir=tmp_path / "artifacts", rest_port=18495
    )
    runtime = MidrunRuntime(config, load_backend("lightning.pytorch"))
    monkeypatch.setattr(
        runtime.torch_debug, "start_after_distributed_init", _failing_start
    )
    module = _FakeModule()

    runtime.setup_environment(module, "fit")
    try:
        assert runtime.rest_url is not None
        assert httpx2.get(f"{runtime.rest_url}/health", timeout=5).status_code == 200
        assert runtime.state.read().torch_debug_url is None
    finally:
        runtime.teardown(module, "fit")


@pytest.mark.parametrize("rest_enabled", [True, False], ids=["paired", "alone"])
def test_an_unanswering_torch_debug_frontend_url_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rest_enabled: bool
) -> None:
    from midrun.runtime import runtime as runtime_module

    monkeypatch.setattr(runtime_module, "distributed_initialized", lambda: True)
    monkeypatch.setattr(runtime_module, "is_rank_zero", lambda: True)
    config = MidrunConfig(
        fuse_enabled=False,
        rest_enabled=rest_enabled,
        artifact_dir=tmp_path / "artifacts",
        rest_port=18500,
    )
    runtime = MidrunRuntime(config, load_backend("lightning.pytorch"))
    # Accepts connections into its backlog and never answers them.
    with socket.create_server(("127.0.0.1", 0)) as silent:

        def _start_silent_frontend(port: int | None = None) -> None:
            del port
            runtime.torch_debug._selected_port = silent.getsockname()[1]

        monkeypatch.setattr(
            runtime.torch_debug, "start_after_distributed_init", _start_silent_frontend
        )
        warnings: list[str] = []
        sink_id = logger.add(warnings.append, level="WARNING")
        module = _FakeModule()
        runtime.setup_environment(module, "fit")
        try:
            assert runtime.state.read().torch_debug_url is None
            assert any("did not answer" in warning for warning in warnings)
            assert (runtime.rest_url is not None) == rest_enabled
            for path, unpublished in [
                ("/meta/pytorch-debug/enabled", b"false\n"),
                ("/meta/pytorch-debug/url", b"\n"),
                ("/meta/pytorch-debug/port", b"\n"),
            ]:
                assert virtual_file_content(runtime, path) == unpublished, path
            fuse_status = json.loads(
                virtual_file_content(runtime, "/debug/pytorch/status")
            )
            assert fuse_status["url"] is None
            if rest_enabled:
                rest_status = httpx2.get(
                    f"{runtime.rest_url}/debug/pytorch/status", timeout=5
                ).json()
                assert rest_status["url"] is None
        finally:
            logger.remove(sink_id)
            runtime.teardown(module, "fit")


def test_a_rank_without_a_debug_frontend_warns_of_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from midrun.runtime import runtime as runtime_module

    monkeypatch.setattr(runtime_module, "distributed_initialized", lambda: True)
    monkeypatch.setattr(runtime_module, "is_rank_zero", lambda: False)
    config = MidrunConfig(fuse_enabled=False, artifact_dir=tmp_path / "artifacts")
    runtime = MidrunRuntime(config, load_backend("lightning.pytorch"))
    # A worker rank's start serves no frontend, so it selects no port.
    monkeypatch.setattr(
        runtime.torch_debug, "start_after_distributed_init", lambda port=None: None
    )
    warnings: list[str] = []
    sink_id = logger.add(warnings.append, level="WARNING")
    module = _FakeModule()
    runtime.setup_environment(module, "fit")
    try:
        assert runtime.state.read().torch_debug_url is None
        assert warnings == []
    finally:
        logger.remove(sink_id)
        runtime.teardown(module, "fit")


def test_stopping_a_rest_server_that_does_not_exit_warns(tmp_path: Path) -> None:
    config = MidrunConfig(fuse_enabled=False, artifact_dir=tmp_path / "artifacts")
    runtime = MidrunRuntime(config, load_backend("lightning.pytorch"))
    release = threading.Event()
    stuck_thread = threading.Thread(target=release.wait, daemon=True)
    stuck_thread.start()
    runtime.services._rest_server = SimpleNamespace(should_exit=False)
    runtime.services._rest_thread = stuck_thread
    warnings: list[str] = []
    sink_id = logger.add(warnings.append, level="WARNING", format="{message}")
    try:
        runtime._stop_rest()
    finally:
        logger.remove(sink_id)
        release.set()

    assert any("REST server thread did not stop" in message for message in warnings)
