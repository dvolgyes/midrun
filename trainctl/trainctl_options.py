"""The keyword arguments that `TrainctlMixin.__init__` accepts."""

from pathlib import Path
from typing import TypedDict


class TrainctlOptions(TypedDict, total=False):
    """The `TrainctlMixin.__init__` keyword arguments, with their types.

    A subclass that forwards its own keyword arguments to `TrainctlMixin` can
    type them as `**trainctl_kwargs: Unpack[TrainctlOptions]`, so that mypy checks
    each option at the subclass's call sites. The keys and types match the
    `TrainctlMixin.__init__` signature. The `TrainctlConfig` docstring describes
    each option; `trainctl_enabled` is `TrainctlConfig.enabled`.

    Attributes:
        trainctl_enabled: See `TrainctlConfig`.
        fuse_enabled: See `TrainctlConfig`.
        fuse_mountpoint: See `TrainctlConfig`.
        rest_enabled: See `TrainctlConfig`.
        rest_host: See `TrainctlConfig`.
        rest_port: See `TrainctlConfig`.
        rest_port_search: See `TrainctlConfig`.
        torch_debug_enabled: See `TrainctlConfig`.
        torch_debug_port: See `TrainctlConfig`.
        torch_debug_port_search: See `TrainctlConfig`.
        paired_port_search_attempts: See `TrainctlConfig`.
        artifact_dir: See `TrainctlConfig`.
        inspection_strict: See `TrainctlConfig`.
        pipeline_pressure_enabled: See `TrainctlConfig`.
        pipeline_pressure_window: See `TrainctlConfig`.
        break_on_backward_exception: See `TrainctlConfig`.
        dataloader_finite_check_enabled: See `TrainctlConfig`.
        optimizer_surgery_enabled: See `TrainctlConfig`.
        hooks_enabled: See `TrainctlConfig`.
        hooks_source_dir: See `TrainctlConfig`.
        hooks_timeout_s: See `TrainctlConfig`.
        hooks_rank_policy: See `TrainctlConfig`.
        hooks_shell: See `TrainctlConfig`.
        hooks_max_params_bytes: See `TrainctlConfig`.
        hooks_max_metadata_items: See `TrainctlConfig`.
        hooks_max_metadata_depth: See `TrainctlConfig`.
        hooks_max_export_bytes: See `TrainctlConfig`.
        hooks_max_output_bytes: See `TrainctlConfig`.
        intercept_lightning_logging: See `TrainctlConfig`.
        run_log_enabled: See `TrainctlConfig`.
    """

    trainctl_enabled: bool
    fuse_enabled: bool
    fuse_mountpoint: Path | str | None
    rest_enabled: bool
    rest_host: str
    rest_port: int
    rest_port_search: int
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
