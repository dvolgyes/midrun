"""Behavioral tests for TrainctlRuntime.artifact_store."""

import pytest

from trainctl.config import TrainctlConfig
from trainctl.load_backend import load_backend
from trainctl.runtime.artifacts import ArtifactStore
from trainctl.runtime.runtime import TrainctlRuntime


def _make_runtime() -> TrainctlRuntime:
    config = TrainctlConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    return TrainctlRuntime(config, load_backend("lightning.pytorch"))


def test_artifact_store_before_setup_names_the_missing_setup() -> None:
    runtime = _make_runtime()
    with pytest.raises(RuntimeError, match="setup_environment"):
        _ = runtime.artifact_store


def test_artifact_store_returns_the_created_store(tmp_path) -> None:
    runtime = _make_runtime()
    store = ArtifactStore(tmp_path)
    runtime.artifacts = store
    assert runtime.artifact_store is store
