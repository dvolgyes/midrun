"""Behavioral tests for midrun.hooks.session's HookSession."""

import json
import shutil
import threading
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

from midrun.hooks.manager.bootstrap_hooks_tree import bootstrap_hooks_tree
from midrun.hooks.session import HookSession

_SHELL = Path(shutil.which("bash"))


class _FakeLog:
    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def warning(self, message, *args) -> None:
        self.warnings.append(message.format(*args))

    def error(self, message, *args) -> None:
        self.errors.append(message.format(*args))

    def info(self, *_args, **_kwargs) -> None:
        pass


def _make_script(path: Path, body: str, *, executable: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/usr/bin/env bash\n{body}\n")
    path.chmod(0o755 if executable else 0o644)


def _make_session(
    live_dir: Path, *, rank: int, rank_policy: str, log=None
) -> HookSession:
    return HookSession(
        live_dir,
        session_id="s-1",
        rank=rank,
        world_size=2,
        rank_policy=rank_policy,
        shell=_SHELL,
        timeout_s=5.0,
        max_params_bytes=256 * 1024,
        max_metadata_items=1000,
        max_metadata_depth=8,
        max_export_bytes=1024**3,
        max_output_bytes=64 * 1024,
        log=log if log is not None else logger,
    )


def test_rank_zero_policy_dispatches_on_rank_zero(tmp_path) -> None:
    live_dir = tmp_path / "run" / "hooks"
    _make_script(live_dir / "light" / "on_train_batch_start.sh", "exit 0")
    session = _make_session(live_dir, rank=0, rank_policy="rank_zero")

    result = session.handle("on_train_batch_start", stage="fit", epoch=0, global_step=0)

    assert result is not None
    assert result.light.phase == "completed"


def test_rank_zero_policy_skips_non_zero_rank_and_allocates_nothing(tmp_path) -> None:
    live_dir = tmp_path / "run" / "hooks"
    marker = tmp_path / "marker"
    _make_script(live_dir / "light" / "on_train_batch_start.sh", f'touch "{marker}"')
    session = _make_session(live_dir, rank=1, rank_policy="rank_zero")

    result = session.handle("on_train_batch_start", stage="fit", epoch=0, global_step=0)

    assert result is None
    assert not marker.exists()


def test_all_policy_dispatches_on_non_zero_rank(tmp_path) -> None:
    live_dir = tmp_path / "run" / "hooks"
    _make_script(live_dir / "light" / "on_train_batch_start.sh", "exit 0")
    session = _make_session(live_dir, rank=1, rank_policy="all")

    result = session.handle("on_train_batch_start", stage="fit", epoch=0, global_step=0)

    assert result is not None
    assert result.light.phase == "completed"


@pytest.mark.parametrize("rank_policy", ["rank_zero", "all"])
def test_disabled_script_returns_none_on_rank_zero(tmp_path, rank_policy) -> None:
    live_dir = tmp_path / "run" / "hooks"
    _make_script(
        live_dir / "light" / "on_train_batch_start.sh", "exit 0", executable=False
    )
    session = _make_session(live_dir, rank=0, rank_policy=rank_policy)

    result = session.handle("on_train_batch_start", stage="fit", epoch=0, global_step=0)

    assert result is not None
    assert result.light is None
    assert result.heavy is None


def test_handle_threads_identity_and_arguments_into_params(tmp_path) -> None:
    live_dir = tmp_path / "run" / "hooks"
    _make_script(live_dir / "light" / "on_train_batch_start.sh", 'cat "$1"')
    session = _make_session(live_dir, rank=0, rank_policy="rank_zero")

    result = session.handle(
        "on_train_batch_start",
        stage="fit",
        epoch=3,
        global_step=42,
        batch_idx=7,
        dataloader_idx=1,
        arguments={"note": "hello"},
    )

    assert result is not None
    assert result.light.phase == "completed"
    seen = result.light.stdout_tail
    assert '"session_id": "s-1"' in seen
    assert '"hook": "on_train_batch_start"' in seen
    assert '"stage": "fit"' in seen
    assert '"epoch": 3' in seen
    assert '"global_step": 42' in seen
    assert '"batch_idx": 7' in seen
    assert '"dataloader_idx": 1' in seen
    assert '"hello"' in seen


def test_epoch_scan_warns_once_ever_for_unknown_path_across_handle_calls(
    tmp_path,
) -> None:
    live_dir = tmp_path / "run" / "hooks"
    _make_script(live_dir / "light" / "on_train_batch_start.sh", "exit 0")
    _make_script(live_dir / "light" / "not_a_callback.sh", "exit 0", executable=False)
    fake_log = _FakeLog()
    session = _make_session(live_dir, rank=0, rank_policy="rank_zero", log=fake_log)

    session.handle("on_train_batch_start", stage="fit", epoch=0, global_step=0)
    session.handle("on_train_batch_start", stage="fit", epoch=0, global_step=1)
    session.handle("on_train_batch_start", stage="fit", epoch=1, global_step=2)

    assert len(fake_log.warnings) == 1
    assert "not_a_callback.sh" in fake_log.warnings[0]


def test_close_removes_tmp_root_and_is_idempotent(tmp_path) -> None:
    live_dir = tmp_path / "run" / "hooks"
    _make_script(live_dir / "light" / "on_train_batch_start.sh", "exit 0")
    session = _make_session(live_dir, rank=0, rank_policy="rank_zero")
    session.handle("on_train_batch_start", stage="fit", epoch=0, global_step=0)

    shared_tmp = live_dir.parent / ".hooks-tmp"
    (session_tmp,) = shared_tmp.iterdir()

    session.close()
    assert not session_tmp.exists()
    assert list(shared_tmp.iterdir()) == []

    session.close()  # idempotent, must not raise
    assert not session_tmp.exists()


def test_close_leaves_another_sessions_in_flight_invocation_intact(tmp_path) -> None:
    live_dir = tmp_path / "run" / "hooks"
    started = tmp_path / "started"
    proceed = tmp_path / "proceed"
    _make_script(
        live_dir / "light" / "on_train_batch_start.sh",
        f'touch "{started}"\n'
        f'while [ ! -e "{proceed}" ]; do sleep 0.01; done\n'
        'test -e "$1"',
    )
    busy = _make_session(live_dir, rank=0, rank_policy="all")
    other = _make_session(live_dir, rank=1, rank_policy="all")
    other.handle("on_fit_start", stage="fit", epoch=0, global_step=0)
    results = []
    worker = threading.Thread(
        target=lambda: results.append(
            busy.handle("on_train_batch_start", stage="fit", epoch=0, global_step=0)
        )
    )
    worker.start()
    deadline = time.monotonic() + 5.0
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.01)

    other.close()
    proceed.touch()
    worker.join()

    assert results[0].light.exit_code == 0


def test_public_identity_attributes_match_constructor_args(tmp_path) -> None:
    live_dir = tmp_path / "run" / "hooks"
    session = _make_session(live_dir, rank=1, rank_policy="all")

    assert session.live_dir == live_dir
    assert session.session_id == "s-1"
    assert session.rank == 1
    assert session.world_size == 2


def test_handle_dispatches_both_light_and_heavy_when_both_enabled(tmp_path) -> None:
    live_dir = tmp_path / "run" / "hooks"
    _make_script(live_dir / "light" / "on_train_batch_start.sh", 'cat "$1"')
    _make_script(live_dir / "heavy" / "on_train_batch_start.sh", 'cat "$1"')
    session = _make_session(live_dir, rank=0, rank_policy="rank_zero")

    result = session.handle(
        "on_train_batch_start",
        stage="fit",
        epoch=0,
        global_step=0,
        arguments={"batch": {"x": torch.arange(4.0)}},
    )

    assert result.light is not None
    assert result.light.phase == "completed"
    assert result.heavy is not None
    assert result.heavy.phase == "completed"


def test_light_and_heavy_share_occurrence_id_but_not_invocation_id(tmp_path) -> None:
    live_dir = tmp_path / "run" / "hooks"
    _make_script(live_dir / "light" / "on_train_batch_start.sh", 'cat "$1"')
    _make_script(live_dir / "heavy" / "on_train_batch_start.sh", 'cat "$1"')
    session = _make_session(live_dir, rank=0, rank_policy="rank_zero")

    result = session.handle(
        "on_train_batch_start",
        stage="fit",
        epoch=0,
        global_step=0,
        arguments={"batch": {"x": torch.arange(4.0)}},
    )

    light_doc = json.loads(result.light.stdout_tail)
    heavy_doc = json.loads(result.heavy.stdout_tail)
    assert light_doc["occurrence_id"] == heavy_doc["occurrence_id"]
    assert light_doc["invocation_id"] != heavy_doc["invocation_id"]


def test_only_heavy_enabled_light_is_none(tmp_path) -> None:
    live_dir = tmp_path / "run" / "hooks"
    _make_script(live_dir / "heavy" / "on_train_batch_start.sh", 'cat "$1"')
    session = _make_session(live_dir, rank=0, rank_policy="rank_zero")

    result = session.handle(
        "on_train_batch_start",
        stage="fit",
        epoch=0,
        global_step=0,
        arguments={"batch": {"x": torch.arange(4.0)}},
    )

    assert result.light is None
    assert result.heavy is not None
    assert result.heavy.phase == "completed"


def test_only_light_enabled_heavy_is_none(tmp_path) -> None:
    live_dir = tmp_path / "run" / "hooks"
    _make_script(live_dir / "light" / "on_train_batch_start.sh", 'cat "$1"')
    session = _make_session(live_dir, rank=0, rank_policy="rank_zero")

    result = session.handle("on_train_batch_start", stage="fit", epoch=0, global_step=0)

    assert result.heavy is None
    assert result.light is not None
    assert result.light.phase == "completed"


def test_heavy_export_restricted_to_catalogue_tensor_args(tmp_path) -> None:
    live_dir = tmp_path / "run" / "hooks"
    _make_script(live_dir / "heavy" / "on_train_batch_start.sh", 'cat "$1"')
    session = _make_session(live_dir, rank=0, rank_policy="rank_zero")

    result = session.handle(
        "on_train_batch_start",
        stage="fit",
        epoch=0,
        global_step=0,
        arguments={
            "batch": {"x": torch.arange(4.0)},
            "unrelated": {"y": torch.arange(4.0)},
        },
    )

    manifest = json.loads(result.heavy.stdout_tail)
    paths = {record["argument_path"] for record in manifest["tensor_records"]}
    assert "batch/x" in paths
    assert not any(path.startswith("unrelated") for path in paths)


def test_heavy_dispatches_with_empty_manifest_for_tensorless_hook(tmp_path) -> None:
    live_dir = tmp_path / "run" / "hooks"
    _make_script(live_dir / "heavy" / "on_train_epoch_end.sh", 'cat "$1"')
    session = _make_session(live_dir, rank=0, rank_policy="rank_zero")

    result = session.handle("on_train_epoch_end", stage="fit", epoch=0, global_step=0)

    assert result.heavy is not None
    assert result.heavy.phase == "completed"
    manifest = json.loads(result.heavy.stdout_tail)
    assert manifest["tensor_records"] == []


def test_rank_gating_skips_both_modalities_together(tmp_path) -> None:
    live_dir = tmp_path / "run" / "hooks"
    heavy_marker = tmp_path / "heavy-marker"
    _make_script(live_dir / "light" / "on_train_batch_start.sh", "exit 0")
    _make_script(
        live_dir / "heavy" / "on_train_batch_start.sh", f'touch "{heavy_marker}"'
    )
    session = _make_session(live_dir, rank=1, rank_policy="rank_zero")

    result = session.handle(
        "on_train_batch_start",
        stage="fit",
        epoch=0,
        global_step=0,
        arguments={"batch": {"x": torch.arange(4.0)}},
    )

    assert result is None
    assert not heavy_marker.exists()


def test_hooks_run_when_the_run_directory_is_relative(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    tree = bootstrap_hooks_tree(Path("runs"), None, session_id="s-rel", log=logger)
    _make_script(tree.live_dir / "light" / "on_train_batch_start.sh", 'cat "$1"')
    _make_script(tree.live_dir / "heavy" / "on_train_batch_start.sh", 'cat "$1"')
    session = _make_session(tree.live_dir, rank=0, rank_policy="rank_zero")

    result = session.handle("on_train_batch_start", stage="fit", epoch=0, global_step=0)

    assert result.light.exit_code == 0, result.light.stderr_tail
    assert json.loads(result.light.stdout_tail)["session_id"] == "s-1"
    assert result.heavy.exit_code == 0, result.heavy.stderr_tail
    assert json.loads(result.heavy.stdout_tail)["tensor_records"] == []


def _block_the_tmp_area(live_dir: Path) -> None:
    """Puts a file where `.hooks-tmp/` goes, so no payload directory can be made."""
    live_dir.mkdir(parents=True, exist_ok=True)
    (live_dir.parent / ".hooks-tmp").write_text("not a directory\n")


def test_an_unwritable_tmp_area_does_not_fail_a_callback_without_hooks(
    tmp_path,
) -> None:
    live_dir = tmp_path / "run" / "hooks"
    _block_the_tmp_area(live_dir)
    session = _make_session(live_dir, rank=0, rank_policy="rank_zero")

    result = session.handle("on_train_batch_start", stage="fit", epoch=0, global_step=0)

    assert result.light is None
    assert result.heavy is None


def test_an_unwritable_tmp_area_is_a_capture_failure_of_each_enabled_hook(
    tmp_path,
) -> None:
    live_dir = tmp_path / "run" / "hooks"
    canary = tmp_path / "should-not-run"
    _make_script(live_dir / "light" / "on_train_batch_start.sh", f'touch "{canary}"')
    _make_script(live_dir / "heavy" / "on_train_batch_start.sh", f'touch "{canary}"')
    _block_the_tmp_area(live_dir)
    log = _FakeLog()
    session = _make_session(live_dir, rank=0, rank_policy="rank_zero", log=log)

    result = session.handle("on_train_batch_start", stage="fit", epoch=0, global_step=0)

    assert result.light.phase == "capture_failed"
    assert result.heavy.phase == "capture_failed"
    assert not canary.exists()
    assert len(log.errors) == 2
