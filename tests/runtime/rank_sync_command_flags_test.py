"""Ranks exchange command lists only at the safe points where rank 0 has commands due."""

import json
import multiprocessing
from pathlib import Path

import torch.distributed as dist

from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.runtime.commands.parse_when import parse_when
from midrun.runtime.events import SafePoint
from midrun.runtime.runtime import MidrunRuntime

_WORLD_SIZE = 2
_IDLE_SAFE_POINTS = 20


class _FakeModule:
    device = "cpu"


class _FakeTrainer:
    should_stop = False
    current_epoch = 0
    global_step = 0


def _run_safe_points(rank: int, tmp_path: Path) -> None:
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
    runtime = MidrunRuntime(config, load_backend("lightning.pytorch"))
    module, trainer = _FakeModule(), _FakeTrainer()
    runtime.setup_environment(module, "fit")
    object_broadcasts: list[int] = []
    broadcast_object_list = dist.broadcast_object_list

    def counted(*args, **kwargs):
        object_broadcasts.append(1)
        return broadcast_object_list(*args, **kwargs)

    dist.broadcast_object_list = counted

    def safe_point() -> tuple[bool, bool]:
        held = runtime._run_due_commands(
            module, trainer, SafePoint.TRAIN_BATCH_END, allowed=None
        )
        return held, trainer.should_stop

    idle = [safe_point() for _ in range(_IDLE_SAFE_POINTS)]
    idle_object_broadcasts = len(object_broadcasts)
    if rank == 0:
        runtime.command_queue.submit("hold", {"reason": "inspect"}, parse_when("now"))
    hold = safe_point()
    if rank == 0:
        runtime.gate.release()
    release = safe_point()
    if rank == 0:
        runtime.command_queue.submit("stop", {}, parse_when("now"))
    stop = safe_point()
    dist.broadcast_object_list = broadcast_object_list
    runtime.teardown(None, "fit")
    dist.destroy_process_group()
    (tmp_path / f"rank{rank}.json").write_text(
        json.dumps(
            {
                "idle": idle,
                "idle_object_broadcasts": idle_object_broadcasts,
                "after": [hold, release, stop],
            }
        )
    )


def test_commands_and_holds_reach_every_rank_without_idle_object_broadcasts(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_run_safe_points, args=(rank, tmp_path))
        for rank in range(_WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=120)
    assert [process.exitcode for process in processes] == [0] * _WORLD_SIZE

    for rank in range(_WORLD_SIZE):
        result = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert result["idle"] == [[False, False]] * _IDLE_SAFE_POINTS
        assert result["idle_object_broadcasts"] == 0
        # The hold and the stop take effect at the safe point that follows each.
        assert result["after"] == [[True, False], [False, False], [False, True]]
