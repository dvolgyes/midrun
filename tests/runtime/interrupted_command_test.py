"""A command whose handler is interrupted ends as failed, and the interrupt propagates."""

import pytest

from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.runtime.commands.parse_when import parse_when
from midrun.runtime.handlers.command_handlers import COMMAND_HANDLERS
from midrun.runtime.runtime import MidrunRuntime


def _interrupt(*_args: object) -> dict[str, object]:
    raise KeyboardInterrupt


def test_an_interrupted_handler_marks_the_command_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = MidrunRuntime(config, load_backend("lightning.pytorch"))
    monkeypatch.setitem(COMMAND_HANDLERS, "set_lightning_profiler", _interrupt)
    command = runtime.command_queue.submit(
        "set_lightning_profiler", {"level": "off"}, parse_when("now")
    )

    with pytest.raises(KeyboardInterrupt):
        runtime._execute(None, None, command, safe_point=None)

    assert command.status == "failed"
    assert command.error is not None
