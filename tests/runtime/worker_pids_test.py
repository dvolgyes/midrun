"""Behavioral tests for RankInfo.worker_pids tracking real OS-level child processes."""

import multiprocessing
import os

from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.runtime.runtime import MidrunRuntime


class _FakeModule:
    device = "cpu"


def _idle(ready: object, stop: object) -> None:
    ready.set()  # type: ignore[attr-defined]
    stop.wait()  # type: ignore[attr-defined]


def _make_runtime() -> MidrunRuntime:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    return MidrunRuntime(config, load_backend("lightning.pytorch"))


def test_gather_ranks_captures_own_pid_with_no_workers() -> None:
    runtime = _make_runtime()
    runtime._gather_ranks(_FakeModule())
    ranks = runtime.state.read().ranks
    assert len(ranks) == 1
    assert ranks[0].pid == os.getpid()
    assert ranks[0].worker_pids == ()


def test_current_rank_info_defaults_to_this_process_before_gather_ranks() -> None:
    info = _make_runtime()._current_rank_info()
    assert info.global_rank == 0
    assert info.pid == os.getpid()
    assert info.worker_pids == ()


def test_current_rank_info_tracks_active_children() -> None:
    runtime = _make_runtime()

    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    stop = ctx.Event()
    proc = ctx.Process(target=_idle, args=(ready, stop))
    proc.start()
    try:
        assert ready.wait(timeout=10)
        info = runtime._current_rank_info()
        assert info.worker_pids == (proc.pid,)
        assert info.pid == os.getpid()
    finally:
        stop.set()
        proc.join(timeout=10)

    assert runtime._current_rank_info().worker_pids == ()


def test_publish_refreshes_worker_pids_on_every_call() -> None:
    runtime = _make_runtime()

    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    stop = ctx.Event()
    proc = ctx.Process(target=_idle, args=(ready, stop))
    proc.start()
    try:
        assert ready.wait(timeout=10)

        class _FakeTrainer:
            callback_metrics: dict[str, float] = {}
            current_epoch = 0
            global_step = 0
            optimizers: list[object] = []

            class state:  # noqa: N801 -- mirrors lightning.pytorch.trainer.states.TrainerState shape
                stage = None

        class _FakePLModule:
            trainer = _FakeTrainer()

        runtime._publish(_FakePLModule(), status="running")
        snapshot = runtime.state.read()
        assert snapshot.ranks[0].worker_pids == (proc.pid,)
    finally:
        stop.set()
        proc.join(timeout=10)
