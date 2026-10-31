"""The CUDA memory counters describe the GPU that this rank trains on.

The counters are read on FUSE and REST threads, whose current CUDA device is not
the training thread's. Fake per-device counters stand in for a second GPU.
"""

import json
from pathlib import Path

import pytest
import torch

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.runtime.virtual_fs.virtual_file_content import virtual_file_content

_ALLOCATED = {0: 100, 1: 200}
_CURRENT_DEVICE = 0


def _index(device: torch.device | str | int | None) -> int:
    if device is None:
        return _CURRENT_DEVICE
    if isinstance(device, int):
        return device
    index = torch.device(device).index
    return _CURRENT_DEVICE if index is None else index


class _Module:
    def __init__(self, device: str) -> None:
        self.device = device


@pytest.fixture
def two_fake_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda device=None: (_ALLOCATED[_index(device)], 1000),
    )
    for name in (
        "memory_allocated",
        "memory_reserved",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        monkeypatch.setattr(
            torch.cuda, name, lambda device=None: _ALLOCATED[_index(device)]
        )


@pytest.mark.usefixtures("two_fake_gpus")
@pytest.mark.parametrize(("device", "expected"), [("cuda:0", 100), ("cuda:1", 200)])
def test_the_cuda_memory_file_shows_the_training_gpu(
    tmp_path: Path, device: str, expected: int
) -> None:
    config = MidrunConfig(
        rest_enabled=False,
        fuse_enabled=False,
        torch_debug_enabled=False,
        hooks_enabled=False,
        artifact_dir=tmp_path / "artifacts",
    )
    runtime = build_runtime(config, load_backend("lightning.pytorch"))
    runtime.setup_environment(_Module(device), "fit")
    try:
        content = json.loads(virtual_file_content(runtime, "/state/cuda-memory.json"))
    finally:
        runtime.teardown(None, "fit")
    assert content["available"] is True
    assert content["memory_allocated"] == expected
    assert content["max_memory_reserved"] == expected
    assert content["free_bytes"] == expected


@pytest.mark.usefixtures("two_fake_gpus")
def test_a_rank_that_trains_on_the_cpu_has_no_cuda_memory_counters(
    tmp_path: Path,
) -> None:
    config = MidrunConfig(
        rest_enabled=False,
        fuse_enabled=False,
        torch_debug_enabled=False,
        hooks_enabled=False,
        artifact_dir=tmp_path / "artifacts",
    )
    runtime = build_runtime(config, load_backend("lightning.pytorch"))
    runtime.setup_environment(_Module("cpu"), "fit")
    try:
        content = json.loads(virtual_file_content(runtime, "/state/cuda-memory.json"))
    finally:
        runtime.teardown(None, "fit")
    assert content["available"] is False
