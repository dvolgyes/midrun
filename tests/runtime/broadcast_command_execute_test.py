"""A rank runs a command that rank 0 broadcast, which its own queue never held."""

import pickle

import torch

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.runtime.command_args.validate_command_args import validate_command_args
from midrun.runtime.command_specs.execute_command import execute_command
from midrun.runtime.commands.command_queue import CommandQueue
from midrun.runtime.commands.parse_when import parse_when
from midrun.runtime.runtime import MidrunRuntime


class _Trainer:
    should_stop = False


def _runtime() -> MidrunRuntime:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    return build_runtime(config, load_backend("lightning.pytorch"))


def _broadcast_stop() -> object:
    rank_zero_queue = CommandQueue(validate_command_args)
    command = rank_zero_queue.submit("stop", {}, parse_when("now"))
    return pickle.loads(pickle.dumps(command))


def test_a_broadcast_command_runs_on_a_rank_whose_queue_never_held_it() -> None:
    runtime = _runtime()
    trainer = _Trainer()
    command = _broadcast_stop()

    execute_command(runtime, torch.nn.Module(), trainer, command, None)

    assert trainer.should_stop
    assert command.status == "succeeded"
    assert runtime.command_queue.list_all() == []


def test_a_broadcast_command_leaves_a_local_command_with_the_same_id_queued() -> None:
    runtime = _runtime()
    local = runtime.command_queue.submit("snapshot", {}, parse_when("now"))
    command = _broadcast_stop()
    # Each process numbers its own commands, so rank 0's id can name a local one.
    command.id = local.id

    execute_command(runtime, torch.nn.Module(), _Trainer(), command, None)

    assert runtime.command_queue.list_all(status="queued") == [local]
