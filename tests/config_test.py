"""Behavioral tests for MidrunConfig's hook-related validation and canonicalization."""

import os
import stat

import pytest

from midrun.config import MidrunConfig


def test_defaults_construct_without_error() -> None:
    config = MidrunConfig()
    assert config.hooks_enabled is True
    assert config.hooks_source_dir is None
    assert config.hooks_shell is None
    assert config.hooks_timeout_s is None
    assert config.hooks_rank_policy == "rank_zero"


def test_invalid_hooks_rank_policy_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="hooks_rank_policy"):
        MidrunConfig(hooks_rank_policy="everyone")


def test_non_positive_hooks_timeout_s_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="hooks_timeout_s"):
        MidrunConfig(hooks_timeout_s=0)
    with pytest.raises(ValueError, match="hooks_timeout_s"):
        MidrunConfig(hooks_timeout_s=-1.0)
    with pytest.raises(ValueError, match="hooks_timeout_s"):
        MidrunConfig(hooks_timeout_s=float("nan"))


def test_none_hooks_timeout_s_is_accepted_as_unbounded() -> None:
    config = MidrunConfig(hooks_timeout_s=None)
    assert config.hooks_timeout_s is None


@pytest.mark.parametrize(
    "field_name",
    [
        "hooks_max_params_bytes",
        "hooks_max_metadata_items",
        "hooks_max_metadata_depth",
        "hooks_max_export_bytes",
        "hooks_max_output_bytes",
    ],
)
def test_non_positive_hooks_limit_fields_raise_at_construction(field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        MidrunConfig(**{field_name: 0})
    with pytest.raises(ValueError, match=field_name):
        MidrunConfig(**{field_name: -1})


def test_hooks_source_dir_must_exist_and_be_a_directory_when_hooks_enabled(
    tmp_path,
) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(ValueError, match="hooks_source_dir"):
        MidrunConfig(hooks_enabled=True, hooks_source_dir=missing)

    a_file = tmp_path / "not-a-directory"
    a_file.write_text("x")
    with pytest.raises(ValueError, match="hooks_source_dir"):
        MidrunConfig(hooks_enabled=True, hooks_source_dir=a_file)


def test_hooks_source_dir_check_is_skipped_when_hooks_disabled(tmp_path) -> None:
    missing = tmp_path / "does-not-exist"
    config = MidrunConfig(hooks_enabled=False, hooks_source_dir=missing)
    assert config.hooks_source_dir == missing


def test_hooks_source_dir_is_canonicalized_to_a_path(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    config = MidrunConfig(hooks_source_dir=str(source_dir))
    assert config.hooks_source_dir == source_dir


def test_hooks_shell_must_exist_and_be_executable_when_hooks_enabled(tmp_path) -> None:
    missing = tmp_path / "no-such-shell"
    with pytest.raises(ValueError, match="hooks_shell"):
        MidrunConfig(hooks_enabled=True, hooks_shell=missing)

    non_executable = tmp_path / "script.sh"
    non_executable.write_text("#!/bin/sh\n")
    non_executable.chmod(0o644)
    with pytest.raises(ValueError, match="hooks_shell"):
        MidrunConfig(hooks_enabled=True, hooks_shell=non_executable)


def test_hooks_shell_check_is_skipped_when_hooks_disabled(tmp_path) -> None:
    missing = tmp_path / "no-such-shell"
    config = MidrunConfig(hooks_enabled=False, hooks_shell=missing)
    assert config.hooks_shell == missing


def test_hooks_shell_accepts_an_executable_file(tmp_path) -> None:
    shell = tmp_path / "shell.sh"
    shell.write_text("#!/bin/sh\n")
    shell.chmod(shell.stat().st_mode | stat.S_IEXEC)
    assert os.access(shell, os.X_OK)

    config = MidrunConfig(hooks_enabled=True, hooks_shell=shell)
    assert config.hooks_shell == shell
