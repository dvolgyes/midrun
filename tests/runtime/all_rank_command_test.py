"""Real two-process Gloo regressions for all-rank command outcomes."""

import json
import multiprocessing
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.runtime.commands.command import Command
from midrun.runtime.commands.parse_when import parse_when
from midrun.runtime.events import SafePoint
from midrun.runtime.runtime import MidrunRuntime

_WORLD_SIZE = 2


class _Trainer:
    current_epoch = 0
    global_step = 0
    should_stop = False

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers
        self.lr_scheduler_configs: list[object] = []


class _Module(torch.nn.Linear):
    device = "cpu"


def _runtime(tmp_path: Path) -> MidrunRuntime:
    config = MidrunConfig(
        rest_enabled=False,
        fuse_enabled=False,
        torch_debug_enabled=False,
        hooks_enabled=False,
        artifact_dir=tmp_path / "artifacts",
    )
    return build_runtime(config, load_backend("lightning.pytorch"))


def _run_all_rank_commands(rank: int, tmp_path: Path) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path / 'store'}",
        rank=rank,
        world_size=_WORLD_SIZE,
    )
    runtime = _runtime(tmp_path)
    model = _Module(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, fused=True)
    model.bias.grad = torch.full_like(model.bias, rank + 1)
    if rank == 1:
        model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()

    bias_before = optimizer.state[model.bias]["momentum_buffer"].clone()
    trainer = _Trainer([optimizer])
    runtime.setup_environment(model, "fit")

    def run(kind: str, args: dict[str, Any]) -> Command | None:
        queue = runtime.command_queue
        submitted = queue.submit(kind, args, parse_when("now")) if rank == 0 else None
        runtime._run_due_commands(
            model, trainer, SafePoint.TRAIN_BATCH_END, allowed=None
        )
        return queue.get(submitted.id) if submitted is not None else None

    reset = run("reset_momentum", {"selector": {"parameter": "bias"}})
    bias_after = optimizer.state[model.bias].get("momentum_buffer")
    result: dict[str, object] = {
        "bias_unchanged": bias_after is not None
        and torch.equal(bias_after, bias_before)
    }

    if rank == 1:
        trainer.optimizers.clear()
    rate = run("set_learning_rate", {"value": 0.25})
    stop = run("stop", {})
    result["stopped"] = trainer.should_stop
    if rank == 0:
        assert reset is not None and rate is not None and stop is not None
        result.update(
            reset_status=reset.status,
            reset_error=reset.error,
            lr_result=rate.status,
            lr_error=rate.error,
            stop_status=stop.status,
        )
    if rank == 1:
        trainer.optimizers.append(optimizer)
    interrupt = KeyboardInterrupt("interrupted during parameter selection")
    command = (
        runtime.command_queue.submit(
            "reset_momentum", {"selector": {"parameter": "bias"}}, parse_when("now")
        )
        if rank == 0
        else None
    )
    injection = (
        patch.object(model, "named_parameters", side_effect=interrupt)
        if rank == 1
        else nullcontext()
    )
    with injection, pytest.raises(BaseException) as caught:
        runtime._run_due_commands(
            model, trainer, SafePoint.TRAIN_BATCH_END, allowed=None
        )
    assert not isinstance(caught.value, Exception)
    if rank == 1:
        assert caught.value is interrupt
    if command is not None:
        assert command.status == "failed"
        assert command.error is not None and "rank 1" in command.error
    (tmp_path / f"rank{rank}.json").write_text(json.dumps(result))

    runtime.teardown(None, "fit")
    dist.destroy_process_group()


def test_all_rank_failures_are_reported_and_do_not_mutate_reset_state(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_run_all_rank_commands, args=(rank, tmp_path))
        for rank in range(_WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=120)
        assert [process.exitcode for process in processes] == [0] * _WORLD_SIZE
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)

    rank0 = json.loads((tmp_path / "rank0.json").read_text())
    rank1 = json.loads((tmp_path / "rank1.json").read_text())
    assert rank0["reset_status"] == "failed"
    assert "rank 1" in rank0["reset_error"]
    assert "fused" in rank0["reset_error"]
    assert rank0["bias_unchanged"] is True
    assert rank1["bias_unchanged"] is True
    assert rank0["stop_status"] == "succeeded"
    assert rank0["stopped"] and rank1["stopped"]
    assert rank0["lr_result"] == "failed"
    assert "rank 1" in rank0["lr_error"]
