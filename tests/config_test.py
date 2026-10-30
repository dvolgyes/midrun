"""Behavioral tests for MidrunConfig's hook-related validation and canonicalization."""

import os
import stat

import pytest
from lightning.pytorch import LightningModule

from midrun import MidrunMixin
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


def test_relative_hooks_shell_names_the_validated_file_from_any_directory(
    tmp_path, monkeypatch
) -> None:
    # Hooks launch with cwd set to the live `hooks/` tree, not the process cwd.
    shell = tmp_path / "bin" / "mybash"
    shell.parent.mkdir()
    shell.write_text("#!/bin/sh\n")
    shell.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    config = MidrunConfig(hooks_enabled=True, hooks_shell="bin/mybash")

    elsewhere = tmp_path / "hooks"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert config.hooks_shell is not None
    assert config.hooks_shell.is_file()
    assert config.hooks_shell.samefile(shell)


@pytest.mark.parametrize(
    "entry",
    [
        "",
        "http://proxy.example",
        "proxy.example:8090",
        "[::1]:8090",
        "proxy example",
        " proxy.example",
        "proxy.example/",
        "*",
        "*.example",
        "user@proxy.example",
        "[127.0.0.1]",
    ],
)
def test_invalid_rest_allowed_hosts_entry_raises_at_construction(entry: str) -> None:
    with pytest.raises(ValueError, match="rest_allowed_hosts"):
        MidrunConfig(rest_allowed_hosts=(entry,))


def test_non_string_rest_allowed_hosts_entry_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="rest_allowed_hosts"):
        MidrunConfig(rest_allowed_hosts=(8090,))  # type: ignore[arg-type]


def test_valid_rest_allowed_hosts_entries_are_kept_as_a_tuple() -> None:
    entries = ["proxy.example", "10.0.0.5", "::1", "[fd00::2]"]
    config = MidrunConfig(rest_allowed_hosts=entries)  # type: ignore[arg-type]
    assert config.rest_allowed_hosts == tuple(entries)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("pipeline_pressure_window", 0),
        ("pipeline_pressure_window", -1),
        ("rest_port", 0),
        ("rest_port", 70000),
        ("torch_debug_port", -1),
        ("torch_debug_port", 65536),
        ("rest_port_search", 0),
        ("torch_debug_port_search", 0),
        ("paired_port_search_attempts", -3),
    ],
)
def test_out_of_range_service_option_raises_at_construction(
    field_name: str, value: int
) -> None:
    with pytest.raises(ValueError, match=field_name):
        MidrunConfig(**{field_name: value})


@pytest.mark.parametrize("field_name", ["rest_port", "torch_debug_port"])
def test_port_range_bounds_are_accepted(field_name: str) -> None:
    assert getattr(MidrunConfig(**{field_name: 1}), field_name) == 1
    assert getattr(MidrunConfig(**{field_name: 65535}), field_name) == 65535


def test_empty_rest_host_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="rest_host"):
        MidrunConfig(rest_host="")


def test_mixin_keyword_arguments_are_validated_like_the_config() -> None:
    class _Module(MidrunMixin, LightningModule):
        pass

    with pytest.raises(ValueError, match="pipeline_pressure_window"):
        _Module(pipeline_pressure_window=-1)
