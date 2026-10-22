"""Runs the heavy hook script of one Lightning callback occurrence and reports failure.

Kept Lightning-agnostic: callers pass plain values already extracted from the
`Trainer`/`LightningModule`, so this module -- and `HookSession` in
`trainctl.hooks.session`, which calls it -- can be exercised without constructing a
real Trainer.
"""

import shutil
from pathlib import Path
from typing import Any

from trainctl.hooks.is_hook_failure import is_hook_failure
from trainctl.hooks.log_hook_failure import log_hook_failure
from trainctl.hooks.payload.build_heavy_manifest import build_heavy_manifest
from trainctl.hooks.payload.export_tensors import export_tensors
from trainctl.hooks.payload.heavy_capture_failure import HeavyCaptureFailure
from trainctl.hooks.payload.new_hook_id import new_hook_id
from trainctl.hooks.payload.occurrence_context import OccurrenceContext
from trainctl.hooks.payload.prepare_heavy_export import prepare_heavy_export
from trainctl.hooks.payload.write_json_file import write_json_file
from trainctl.hooks.resolve_enabled_script import resolve_enabled_script
from trainctl.hooks.runner import ExecutionResult, run_script


def dispatch_heavy(
    live_dir: Path,
    context: OccurrenceContext,
    occurrence_id: str,
    arguments: dict[str, Any],
    tensor_arg_names: tuple[str, ...],
    *,
    shell: Path,
    timeout_s: float | None,
    max_metadata_items: int,
    max_metadata_depth: int,
    max_export_bytes: int,
    max_output_bytes: int,
    tmp_root: Path,
    log: Any,
) -> ExecutionResult | None:
    """Builds `manifest.json` plus `.npy` tensors and runs the heavy hook, if enabled.

    Validates every tensor `context.hook` would export ("all or failed") before any
    host allocation; a callback with no tensor arguments produces a valid empty
    manifest rather than being skipped. A failed capture is logged as one ERROR
    record with `phase="capture_failed"` and the script is never launched.

    Returns:
        `None` if the heavy script is not currently enabled; otherwise the
        completed (or capture-failed) `ExecutionResult`.
    """
    if resolve_enabled_script(live_dir, "heavy", context.hook) is None:
        return None
    invocation_id = new_hook_id()
    prepared = prepare_heavy_export(
        arguments,
        tensor_arg_names,
        max_items=max_metadata_items,
        max_depth=max_metadata_depth,
        max_export_bytes=max_export_bytes,
    )
    script = resolve_enabled_script(live_dir, "heavy", context.hook)
    if script is None:
        return None
    if isinstance(prepared, HeavyCaptureFailure):
        result = ExecutionResult(
            phase="capture_failed",
            exit_code=None,
            signal_number=None,
            duration_s=0.0,
            stdout_tail="",
            stderr_tail="",
            stdout_truncated=False,
            stderr_truncated=False,
            error=prepared.reason,
        )
        log_hook_failure(
            script, "heavy", context, occurrence_id, invocation_id, result, log
        )
        return result
    leaves, truncated, reason = prepared
    invocation_dir = tmp_root / invocation_id
    invocation_dir.mkdir(mode=0o700, parents=True)
    try:
        tensor_records = export_tensors(leaves, invocation_dir)
        manifest = build_heavy_manifest(
            context=context,
            occurrence_id=occurrence_id,
            invocation_id=invocation_id,
            arguments=arguments,
            tensor_arg_names=tensor_arg_names,
            tensor_records=tensor_records,
            max_metadata_items=max_metadata_items,
            max_metadata_depth=max_metadata_depth,
            truncated=truncated,
            truncated_reason=reason,
        )
        manifest_path = write_json_file(invocation_dir / "manifest.json", manifest)
        result = run_script(
            script,
            [str(manifest_path)],
            cwd=live_dir,
            shell=shell,
            timeout_s=timeout_s,
            max_output_bytes=max_output_bytes,
            log=log,
        )
    finally:
        shutil.rmtree(invocation_dir, ignore_errors=True)
    if is_hook_failure(result):
        log_hook_failure(
            script, "heavy", context, occurrence_id, invocation_id, result, log
        )
    return result
