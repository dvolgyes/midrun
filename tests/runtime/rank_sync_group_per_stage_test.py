"""The control-plane process group of one stage does not outlive the stage."""

import json
import multiprocessing
import os
from datetime import timedelta
from pathlib import Path
from time import monotonic
from unittest.mock import patch

import pytest
import torch.distributed as dist

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend

_WORLD_SIZE = 2


class _FakeModule:
    device = "cpu"


def _open_fds() -> int:
    return len(os.listdir("/proc/self/fd"))


def _run_stages(rank: int, tmp_path: Path) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path / 'store'}",
        rank=rank,
        world_size=_WORLD_SIZE,
    )
    config = MidrunConfig(
        rest_enabled=False,
        fuse_enabled=False,
        torch_debug_enabled=False,
        hooks_enabled=False,
        artifact_dir=tmp_path / "artifacts",
    )
    runtime = build_runtime(config, load_backend("lightning.pytorch"))
    counts = []
    for _ in range(6):
        runtime.setup_environment(_FakeModule(), "fit")
        runtime.teardown(None, "fit")
        counts.append(_open_fds())
    dist.destroy_process_group()
    (tmp_path / f"rank{rank}.json").write_text(json.dumps(counts))


def test_repeated_stages_do_not_grow_the_open_file_descriptors(tmp_path) -> None:
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_run_stages, args=(rank, tmp_path))
        for rank in range(_WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=120)
    assert [process.exitcode for process in processes] == [0] * _WORLD_SIZE

    for rank in range(_WORLD_SIZE):
        counts = json.loads((tmp_path / f"rank{rank}.json").read_text())
        # The first stages may open lazily created descriptors; later ones open none.
        assert counts[-1] == counts[1], counts


def _restart_after_partial_shutdown(rank: int, tmp_path: Path) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path / 'store'}",
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=timedelta(seconds=30),
    )
    try:
        runtime = build_runtime(
            MidrunConfig(
                rest_enabled=False,
                fuse_enabled=False,
                torch_debug_enabled=False,
                hooks_enabled=False,
                inspection_strict=False,
                artifact_dir=tmp_path / "artifacts",
            ),
            load_backend("lightning.pytorch"),
        )
        module = _FakeModule()
        runtime.setup_environment(module, "fit")
        old = runtime.rank_sync
        close = old.close
        fail_once = rank == 1

        def close_old_group() -> None:
            nonlocal fail_once
            if fail_once:
                fail_once = False
                raise OSError("injected control group shutdown error")
            close()

        with patch.object(old, "close", close_old_group):
            runtime.teardown(module, "fit")
        with pytest.raises(RuntimeError, match="shutdown is pending"):
            runtime.setup_environment(module, "test")
        assert runtime.rank_sync is old
        assert old.world_size == (_WORLD_SIZE if rank == 1 else 1)
        runtime.teardown(module, "fit")
        assert old.world_size == 1
        runtime.setup_environment(module, "test")
        assert runtime.rank_sync.all_gather(rank) == list(range(_WORLD_SIZE))
        runtime.teardown(module, "test")
    finally:
        dist.destroy_process_group()


def test_partial_shutdown_rejects_restart_on_every_rank_until_retry(tmp_path) -> None:
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_restart_after_partial_shutdown, args=(rank, tmp_path))
        for rank in range(_WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    try:
        deadline = monotonic() + 120
        for process in processes:
            process.join(timeout=max(0, deadline - monotonic()))
        assert [process.exitcode for process in processes] == [0] * _WORLD_SIZE
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join(timeout=5)
