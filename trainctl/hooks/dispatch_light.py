"""Runs the light hook script of one Lightning callback occurrence and reports failure.

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
from trainctl.hooks.payload.build_light_params import build_light_params
from trainctl.hooks.payload.new_hook_id import new_hook_id
from trainctl.hooks.payload.occurrence_context import OccurrenceContext
from trainctl.hooks.resolve_enabled_script import resolve_enabled_script
from trainctl.hooks.runner import ExecutionResult, run_script
from trainctl.hooks.write_json_file import write_json_file


def dispatch_light(
    live_dir: Path,
    context: OccurrenceContext,
    arguments: dict[str, Any],
    *,
    occurrence_id: str,
    shell: Path,
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

    Returns:
        `None` if the light script is not currently enabled (an ordinary skip, not
        logged as a failure); otherwise the completed `ExecutionResult`. A failing
        result is logged here as one ERROR record before being returned.
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
    invocation_dir.mkdir(mode=0o700, parents=True)
    try:
        params_path = write_json_file(invocation_dir / "params.json", params)
        result = run_script(
            script,
            [str(params_path)],
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
            script, "light", context, occurrence_id, invocation_id, result, log
        )
    return result
