"""A command whose handler is interrupted ends as failed, and the interrupt propagates."""

from dataclasses import replace

import pytest

from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.runtime.command_specs.command_specs import COMMAND_SPECS
from midrun.runtime.command_specs.execute_command import execute_command
from midrun.runtime.commands.parse_when import parse_when
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
    spec = COMMAND_SPECS["set_lightning_profiler"]
    monkeypatch.setattr(
        "midrun.runtime.command_specs.execute_command.COMMAND_SPECS",
        {**COMMAND_SPECS, "set_lightning_profiler": replace(spec, handler=_interrupt)},
    )
    command = runtime.command_queue.submit(
        "set_lightning_profiler", {"level": "off"}, parse_when("now")
    )

    with pytest.raises(KeyboardInterrupt):
        execute_command(runtime, None, None, command, safe_point=None)

    assert command.status == "failed"
    assert command.error is not None
