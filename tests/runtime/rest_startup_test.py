"""Behavioral test: REST starts with default service options in a non-distributed run."""

from pathlib import Path

import httpx2
import torch.distributed as dist

from trainctl.config import TrainctlConfig
from trainctl.lightning_backend import load_backend
from trainctl.runtime.runtime import TrainctlRuntime


class _FakeModule:
    device = "cpu"


def test_rest_starts_without_distributed_process_group(tmp_path: Path) -> None:
    assert not dist.is_initialized()
    config = TrainctlConfig(
        fuse_enabled=False,
        artifact_dir=tmp_path / "artifacts",
        rest_port=18490,
        inspection_strict=True,
    )
    assert config.rest_enabled and config.torch_debug_enabled
    runtime = TrainctlRuntime(config, load_backend("lightning.pytorch"))
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
