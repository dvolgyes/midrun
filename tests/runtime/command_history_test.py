"""The command queue keeps every live command and only the recent finished ones."""

from midrun.runtime.command_args.validate_command_args import validate_command_args
from midrun.runtime.commands.command_queue import CommandQueue
from midrun.runtime.commands.parse_when import parse_when

_FINISHED_KEPT = 1000


def _queue() -> CommandQueue:
    return CommandQueue(validate_command_args)


def _submit(queue: CommandQueue) -> str:
    return queue.submit("stop", {}, parse_when("now")).id


def test_the_oldest_finished_command_is_forgotten_past_the_history_limit() -> None:
    queue = _queue()
    ids = [_submit(queue) for _ in range(_FINISHED_KEPT + 1)]
    for command_id in ids:
        assert queue.cancel(command_id)

    assert queue.get(ids[0]) is None
    assert queue.get(ids[1]) is not None
    assert [command.id for command in queue.list_all()] == ids[1:]


def test_queued_commands_are_never_forgotten() -> None:
    queue = _queue()
    queued_id = _submit(queue)
    for _ in range(_FINISHED_KEPT + 1):
        assert queue.cancel(_submit(queue))

    assert queue.get(queued_id) is not None
    assert queue.list_all(status="queued") == [queue.get(queued_id)]


def test_an_executing_command_is_listed_as_neither_queued_nor_forgotten() -> None:
    queue = _queue()
    command = queue.submit("stop", {}, parse_when("now"))
    assert queue.mark_executing(command)
    for _ in range(_FINISHED_KEPT + 1):
        assert queue.cancel(_submit(queue))

    assert queue.list_all(status="queued") == []
    assert queue.get(command.id) is command
    assert queue.due(None, 0, 0, None) == []
