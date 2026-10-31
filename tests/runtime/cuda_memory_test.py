"""Behavioral tests for the CUDA memory inspector's pure functions.

`_record_memory_history` toggles process-global torch state, so every test that
enables it must disable it again in a `finally`, regardless of outcome. REST-route
behavior (GET /debug/cuda-memory/snapshot, GET /debug/cuda-memory/history) is
covered separately in tests/rest/cuda_memory_test.py -- these operations don't
touch the Trainer/model, so they execute directly from REST, not as commands.
"""

import json
import subprocess
import sys

import pytest
import torch

from midrun.runtime.cuda_memory.cheap_cuda_memory_stats import (
    cheap_cuda_memory_stats,
)
from midrun.runtime.cuda_memory.dump_cuda_memory_snapshot import (
    dump_cuda_memory_snapshot,
)
from midrun.runtime.cuda_memory.set_cuda_memory_history import (
    set_cuda_memory_history,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def test_cheap_stats_reports_real_allocator_counters() -> None:
    _ = torch.zeros(1, device="cuda")
    stats = cheap_cuda_memory_stats("cuda:0")
    assert stats["available"] is True
    assert stats["free_bytes"] > 0
    assert stats["total_bytes"] > 0


_READ_CUDA_MEMORY_FILE = """
import json
import torch
from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.runtime.virtual_fs.virtual_file_content import virtual_file_content

runtime = build_runtime(
    MidrunConfig(rest_enabled=False, torch_debug_enabled=False, fuse_enabled=False),
    load_backend("lightning.pytorch"),
)
content = virtual_file_content(runtime, "/state/cuda-memory.json")
print(json.dumps([json.loads(content), torch.cuda.is_initialized()]))
"""


def test_cuda_memory_file_does_not_initialize_cuda() -> None:
    """A stat or read of the file must not create a CUDA context for the reader."""
    result = subprocess.run(  # noqa: S603 -- fixed command, test-owned script
        [sys.executable, "-c", _READ_CUDA_MEMORY_FILE],
        capture_output=True,
        check=True,
        text=True,
    )
    content, initialized = json.loads(result.stdout)
    assert initialized is False
    assert content["available"] is False


def test_set_history_enable_and_disable_round_trip() -> None:
    try:
        state = set_cuda_memory_history(True, max_entries=1000)
        assert state.enabled is True
        assert state.max_entries == 1000
        assert state.started_at is not None
    finally:
        disabled = set_cuda_memory_history(False, max_entries=None)
        assert disabled.enabled is False


def test_dump_snapshot_returns_nonempty_pickle_bytes() -> None:
    try:
        set_cuda_memory_history(True, max_entries=1000)
        tensor = torch.randn(1000, 1000, device="cuda")
        _ = tensor @ tensor
        snapshot = dump_cuda_memory_snapshot()
        assert isinstance(snapshot, bytes)
        assert len(snapshot) > 0
    finally:
        set_cuda_memory_history(False, max_entries=None)
