"""`HookSession`: dispatches one Lightning callback occurrence to its light/heavy script.

Owned by `MidrunRuntime`, constructed once per run right after
`bootstrap_hooks_tree` publishes or resumes the live tree. Lightning-agnostic:
`handle()` takes plain values, not a `Trainer`/`LightningModule`, so it is testable
without constructing either.
"""

import shutil
from pathlib import Path
from typing import Any

from midrun.hooks.catalog import CALLBACK_BY_NAME
from midrun.hooks.discovery import EpochScanState
from midrun.hooks.dispatch_heavy import dispatch_heavy
from midrun.hooks.dispatch_light import dispatch_light
from midrun.hooks.handle_result import HandleResult
from midrun.hooks.hook_process import HookProcess
from midrun.hooks.hook_rank import HookRank
from midrun.hooks.payload.hook_capture_limits import HookCaptureLimits
from midrun.hooks.payload.new_hook_id import new_hook_id
from midrun.hooks.payload.occurrence_context import OccurrenceContext


class HookSession:
    """Dispatches lifecycle callback occurrences against one run's live hooks tree.

    Scripts start through `setpriv` (resolved once by the caller), so that the
    kernel kills a script's shell when the trainer dies. When `setpriv` is `None`,
    an eligible session logs one warning that scripts can outlive a killed trainer.

    :ivar live_dir: Published tree this session dispatches against.
    :ivar session_id: Run identity shared with the bootstrap marker.
    :ivar _rank: Rank identity and dispatch eligibility.
    :ivar _process: Process launch policy and session logger.
    :ivar _limits: Bounds on metadata and tensor capture.
    :ivar _epoch_scan: Discovery state for the current epoch.
    :ivar _tmp_root: Private temporary directory for this session.
    """

    live_dir: Path
    session_id: str
    _rank: HookRank
    _process: HookProcess
    _limits: HookCaptureLimits
    _epoch_scan: EpochScanState
    _tmp_root: Path

    def __init__(
        self,
        live_dir: Path,
        *,
        session_id: str,
        rank: HookRank,
        process: HookProcess,
        limits: HookCaptureLimits,
    ) -> None:
        self.live_dir = live_dir
        self.session_id = session_id
        self._rank = rank
        self._process = process
        self._limits = limits
        if process.setpriv is None and (rank.policy == "all" or rank.rank == 0):
            process.log.warning(
                "midrun: setpriv is not on PATH; a hook script can outlive a "
                "killed trainer"
            )
        self._epoch_scan = EpochScanState()
        # `.hooks-tmp/` is shared by every rank and session of the run; each session
        # owns only its uniquely named directory inside it.
        self._tmp_root = (
            live_dir.parent / ".hooks-tmp" / f"rank-{self.rank}-{new_hook_id()}"
        )

    @property
    def rank(self) -> int:
        """Global rank of this session's process."""
        return self._rank.rank

    @property
    def world_size(self) -> int:
        """Number of ranks in the training job."""
        return self._rank.world_size

    def handle(
        self,
        hook: str,
        *,
        stage: str | None,
        epoch: int | None,
        global_step: int | None,
        batch_idx: int | None = None,
        dataloader_idx: int | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> HandleResult | None:
        """Dispatches one callback occurrence's light and heavy hooks, if this rank is eligible.

        Light and heavy share this firing's occurrence id but get distinct invocation
        ids and capture timestamps -- light runs first, so the two are not a single
        simultaneous snapshot. Heavy's tensor selection comes from the catalogue's
        static `tensor_args` for `hook`; a callback with none still dispatches heavy
        (a valid empty manifest) if its script is enabled.

        Args:
            hook: The exact `Callback` method name.
            stage: `str(trainer.state.fn)`, or `None` if unavailable.
            epoch: `trainer.current_epoch`, or `None`.
            global_step: `trainer.global_step`, or `None`.
            batch_idx: The batch index Lightning passed, if any.
            dataloader_idx: The dataloader index Lightning passed, if any.
            arguments: This callback's own named arguments, for the bounded
                metadata description and, for tensor-carrying arguments, heavy
                export. Defaults to an empty mapping.

        Returns:
            `None` if this rank is ineligible under the configured rank policy.
            Otherwise a `HandleResult` whose `light`/`heavy` fields are each `None`
            when that modality's script isn't currently enabled.
        """
        if not self._rank.eligible:
            return None
        self._epoch_scan.scan_if_due(self.live_dir, (stage, epoch), self._process.log)
        context = OccurrenceContext(
            session_id=self.session_id,
            hook=hook,
            stage=stage,
            epoch=epoch,
            global_step=global_step,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
            rank=self.rank,
            world_size=self.world_size,
        )
        occurrence_id = new_hook_id()
        args = arguments or {}
        light_result = dispatch_light(
            self.live_dir,
            context,
            args,
            occurrence_id=occurrence_id,
            process=self._process,
            limits=self._limits,
            tmp_root=self._tmp_root,
        )
        spec = CALLBACK_BY_NAME.get(hook)
        tensor_arg_names = spec.tensor_args if spec is not None else ()
        heavy_result = dispatch_heavy(
            self.live_dir,
            context,
            occurrence_id,
            args,
            tensor_arg_names,
            process=self._process,
            limits=self._limits,
            tmp_root=self._tmp_root,
        )
        return HandleResult(light=light_result, heavy=heavy_result)

    def close(self) -> None:
        """Removes this session's private temporary-invocation root, if created.

        Idempotent. Any invocation directory still present here is a leak from an
        invocation that did not clean up after itself (e.g. a hard-killed process);
        this only ever removes this session's own directory under the run's
        `.hooks-tmp/`, never another session's or rank's.
        """
        shutil.rmtree(self._tmp_root, ignore_errors=True)
