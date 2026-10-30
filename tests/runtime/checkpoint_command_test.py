"""Behavioral tests for the `checkpoint` command handler and its reserved file name."""

from pathlib import Path

import pytest

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.commands.command import Command
from midrun.runtime.commands.parse_when import parse_when
from midrun.runtime.handlers.handle_checkpoint import handle_checkpoint
from midrun.runtime.runtime import MidrunRuntime


def _make_runtime(tmp_path: Path) -> MidrunRuntime:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = build_runtime(config, load_backend("lightning.pytorch"))
    runtime.artifacts = ArtifactStore(tmp_path)
    return runtime


def _checkpoint_command() -> Command:
    return Command(
        id="cmd_test",
        kind="checkpoint",
        args={"label": "manual"},
        when=parse_when("now"),
        created_at=0.0,
    )


class _SingleProcessStrategy:
    def barrier(self, name: str) -> None:
        del name


class _FailingTrainer:
    strategy = _SingleProcessStrategy()

    def save_checkpoint(self, path: str) -> None:
        raise OSError(f"disk full while writing {path}")


class _RecordingTrainer:
    def __init__(self) -> None:
        self.saved: list[str] = []

    def save_checkpoint(self, path: str) -> None:
        self.saved.append(path)


def test_a_failed_save_leaves_no_reserved_checkpoint_file(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)

    with pytest.raises(OSError, match="disk full"):
        handle_checkpoint(runtime, None, _FailingTrainer(), _checkpoint_command(), None)

    assert list((tmp_path / "checkpoints").iterdir()) == []


def test_a_worker_rank_saves_at_the_path_of_rank_zero_and_reserves_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _make_runtime(tmp_path)
    runtime._is_rank_zero = False
    rank_zero_path = tmp_path / "checkpoints" / "ckpt-c_000042-manual.ckpt"
    monkeypatch.setattr(runtime.rank_sync, "broadcast", lambda _: rank_zero_path)
    trainer = _RecordingTrainer()

    result = handle_checkpoint(runtime, None, trainer, _checkpoint_command(), None)

    assert trainer.saved == [str(rank_zero_path)]
    assert result == {"checkpoint_path": str(rank_zero_path)}
    assert list((tmp_path / "checkpoints").iterdir()) == []
