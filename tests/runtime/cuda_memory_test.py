"""Behavioral tests for the CUDA memory inspector's pure functions.

`_record_memory_history` toggles process-global torch state, so every test that
enables it must disable it again in a `finally`, regardless of outcome. REST-route
behavior (GET /debug/cuda-memory/snapshot, GET /debug/cuda-memory/history) is
covered separately in tests/rest/cuda_memory_test.py -- these operations don't
touch the Trainer/model, so they execute directly from REST, not as commands.
"""

import pytest
import torch

from trainctl.runtime.cuda_memory.cheap_cuda_memory_stats import (
    cheap_cuda_memory_stats,
)
from trainctl.runtime.cuda_memory.dump_cuda_memory_snapshot import (
    dump_cuda_memory_snapshot,
)
from trainctl.runtime.cuda_memory.set_cuda_memory_history import (
    set_cuda_memory_history,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def test_cheap_stats_reports_real_allocator_counters() -> None:
    stats = cheap_cuda_memory_stats()
    assert stats["available"] is True
    assert stats["free_bytes"] > 0
    assert stats["total_bytes"] > 0


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
