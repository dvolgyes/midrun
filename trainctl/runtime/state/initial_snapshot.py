"""Builds the runtime snapshot that is published first."""

import time

from trainctl.runtime.state.runtime_snapshot import RuntimeSnapshot


def initial_snapshot(
    run_id: str, lightning_backend: str, lightning_version: str
) -> RuntimeSnapshot:
    """Builds the starting snapshot published before any Lightning hook has run."""
    return RuntimeSnapshot(
        generation=0,
        updated_at=time.time(),
        run_id=run_id,
        status="initializing",
        stage=None,
        phase="idle",
        epoch=0,
        global_step=0,
        batch_idx=None,
        world_size=1,
        ranks=(),
        metrics={},
        learning_rates=(),
        pending_commands=0,
        hold_state="none",
        rest_host=None,
        rest_port=None,
        torch_debug_url=None,
        artifacts_path=None,
        lightning_backend=lightning_backend,
        lightning_version=lightning_version,
        exception=None,
        backward_interception_supported=True,
        backward_interception_reason=None,
    )
