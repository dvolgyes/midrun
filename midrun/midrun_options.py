"""The keyword arguments that `MidrunMixin.__init__` accepts."""

from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict


class MidrunOptions(TypedDict, total=False):
    """The `MidrunMixin.__init__` keyword arguments, with their types.

    A subclass that forwards its own keyword arguments to `MidrunMixin` can
    type them as `**midrun_kwargs: Unpack[MidrunOptions]`, so that mypy checks
    each option at the subclass's call sites. The keys and types match the
    `MidrunMixin.__init__` signature. The `MidrunConfig` docstring describes
    each option; `midrun_enabled` is `MidrunConfig.enabled`.

    Attributes:
        midrun_enabled: See `MidrunConfig`.
        fuse_enabled: See `MidrunConfig`.
        fuse_mountpoint: See `MidrunConfig`.
        rest_enabled: See `MidrunConfig`.
        rest_host: See `MidrunConfig`.
        rest_port: See `MidrunConfig`.
        rest_port_search: See `MidrunConfig`.
        rest_allowed_hosts: See `MidrunConfig`.
        torch_debug_enabled: See `MidrunConfig`.
        torch_debug_port: See `MidrunConfig`.
        torch_debug_port_search: See `MidrunConfig`.
        paired_port_search_attempts: See `MidrunConfig`.
        artifact_dir: See `MidrunConfig`.
        inspection_strict: See `MidrunConfig`.
        pipeline_pressure_enabled: See `MidrunConfig`.
        pipeline_pressure_window: See `MidrunConfig`.
        break_on_backward_exception: See `MidrunConfig`.
        dataloader_finite_check_enabled: See `MidrunConfig`.
        optimizer_surgery_enabled: See `MidrunConfig`.
        hooks_enabled: See `MidrunConfig`.
        hooks_source_dir: See `MidrunConfig`.
        hooks_timeout_s: See `MidrunConfig`.
        hooks_rank_policy: See `MidrunConfig`.
        hooks_shell: See `MidrunConfig`.
        hooks_max_params_bytes: See `MidrunConfig`.
        hooks_max_metadata_items: See `MidrunConfig`.
        hooks_max_metadata_depth: See `MidrunConfig`.
        hooks_max_export_bytes: See `MidrunConfig`.
        hooks_max_output_bytes: See `MidrunConfig`.
        intercept_lightning_logging: See `MidrunConfig`.
        run_log_enabled: See `MidrunConfig`.
    """

    midrun_enabled: bool
    fuse_enabled: bool
    fuse_mountpoint: Path | str | None
    rest_enabled: bool
    rest_host: str
    rest_port: int
    rest_port_search: int
    rest_allowed_hosts: Sequence[str]
    torch_debug_enabled: bool
    torch_debug_port: int
    torch_debug_port_search: int
    paired_port_search_attempts: int
    artifact_dir: Path | str | None
    inspection_strict: bool
    pipeline_pressure_enabled: bool
    pipeline_pressure_window: int
    break_on_backward_exception: bool
    dataloader_finite_check_enabled: bool
    optimizer_surgery_enabled: bool
    hooks_enabled: bool
    hooks_source_dir: Path | str | None
    hooks_timeout_s: float | None
    hooks_rank_policy: str
    hooks_shell: Path | str | None
    hooks_max_params_bytes: int
    hooks_max_metadata_items: int
    hooks_max_metadata_depth: int
    hooks_max_export_bytes: int
    hooks_max_output_bytes: int
    intercept_lightning_logging: bool
    run_log_enabled: bool
