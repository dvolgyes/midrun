"""Runs the light hook script of one Lightning callback occurrence and reports failure.

Kept Lightning-agnostic: callers pass plain values already extracted from the
`Trainer`/`LightningModule`, so this module -- and `HookSession` in
`midrun.hooks.session`, which calls it -- can be exercised without constructing a
real Trainer.
"""

import shutil
from pathlib import Path
from typing import Any

from midrun.hooks.capture_failed_result import capture_failed_result
from midrun.hooks.create_invocation_dir import create_invocation_dir
from midrun.hooks.is_hook_failure import is_hook_failure
from midrun.hooks.log_hook_failure import log_hook_failure
from midrun.hooks.payload.build_light_params import build_light_params
from midrun.hooks.payload.new_hook_id import new_hook_id
from midrun.hooks.payload.occurrence_context import OccurrenceContext
from midrun.hooks.resolve_enabled_script import resolve_enabled_script
from midrun.hooks.runner import ExecutionResult, run_script
from midrun.hooks.write_json_file import write_json_file


def dispatch_light(
    live_dir: Path,
    context: OccurrenceContext,
    arguments: dict[str, Any],
    *,
    occurrence_id: str,
    shell: Path,
    setpriv: Path | None,
    timeout_s: float | None,
    max_params_bytes: int,
    max_metadata_items: int,
    max_metadata_depth: int,
    max_output_bytes: int,
    tmp_root: Path,
    log: Any,
) -> ExecutionResult | None:
    """Builds `params.json` and runs the light hook for `context.hook`, if enabled.

    Revalidates enablement twice: once before spending time building the metadata
    (which is cheap, but heavy's counterpart is not), and again immediately before
    launch, per the discovery contract's re-check requirement. `occurrence_id` is this
    callback firing's identity, shared with the heavy hook dispatched for the same
    occurrence (each still gets its own invocation id).

    An OS error while writing the payload (e.g. a full disk) is a failed capture:
    the script is never launched and the result has `phase="capture_failed"`.

    Returns:
        `None` if the light script is not currently enabled (an ordinary skip, not
        logged as a failure); otherwise the completed (or capture-failed)
        `ExecutionResult`. A failing result is logged here as one ERROR record
        before being returned.
    """
    if resolve_enabled_script(live_dir, "light", context.hook) is None:
        return None
    invocation_id = new_hook_id()
    params, _truncated, _reason = build_light_params(
        context=context,
        occurrence_id=occurrence_id,
        invocation_id=invocation_id,
        arguments=arguments,
        max_params_bytes=max_params_bytes,
        max_metadata_items=max_metadata_items,
        max_metadata_depth=max_metadata_depth,
    )
    script = resolve_enabled_script(live_dir, "light", context.hook)
    if script is None:
        return None
    invocation_dir = tmp_root / invocation_id
    try:
        try:
            create_invocation_dir(tmp_root, invocation_id)
            params_path = write_json_file(invocation_dir / "params.json", params)
        except OSError as exc:
            # E.g. a full disk: the capture failed, training must not.
            result = capture_failed_result(f"cannot write the light payload: {exc}")
        else:
            result = run_script(
                script,
                [str(params_path)],
                cwd=live_dir,
                shell=shell,
                setpriv=setpriv,
                timeout_s=timeout_s,
                max_output_bytes=max_output_bytes,
                log=log,
            )
    finally:
        shutil.rmtree(invocation_dir, ignore_errors=True)
    if is_hook_failure(result):
        log_hook_failure(
            script, "light", context, occurrence_id, invocation_id, result, log
        )
    return result
