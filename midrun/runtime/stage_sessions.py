"""Opens and closes the logging session and the hooks session of a stage `_run`."""

import shutil
from pathlib import Path
from typing import Any

from midrun.config import MidrunConfig
from midrun.hooks.manager.bootstrap_hooks_tree import bootstrap_hooks_tree
from midrun.hooks.session import HandleResult, HookSession
from midrun.logging.logging_session import LoggingSession
from midrun.runtime.default_base_dir import default_base_dir
from midrun.runtime.new_hooks_session_id import new_hooks_session_id
from midrun.runtime.start_logging_session import start_logging_session
from midrun.runtime.trainer_stage_name import trainer_stage_name


class StageSessions:
    """Holds the per-stage logging session and hooks session of one runtime.

    Attributes:
        config: Effective `MidrunConfig` for this run.
        run_id: Unique id for this training run.
        hook_session: The hooks session of the current stage, if one is open.
        logging_session: The logging session of the current stage, if one is open.
    """

    config: MidrunConfig
    run_id: str
    hook_session: HookSession | None
    logging_session: LoggingSession | None

    def __init__(self, config: MidrunConfig, run_id: str) -> None:
        self.config = config
        self.run_id = run_id
        self.hook_session = None
        self.logging_session = None

    def open_logging_session(self, trainer: Any, pl_module: Any) -> LoggingSession:
        """Opens this stage `_run`'s logging session, unless it is open, and returns it.

        Idempotent. The first Midrun callback `setup` of the `_run` opens it: the hooks
        adapter's, which needs its logger, else the service guard's. The mixin's
        `teardown` or the guard's `on_exception` closes it (`close_logging_session`).
        """
        if self.logging_session is None:
            # Set before the bridge or the file can fail, so that the close finds it.
            self.logging_session = LoggingSession(new_hooks_session_id())
            start_logging_session(
                self.logging_session,
                self.config,
                default_base_dir(pl_module, self.run_id),
                trainer.global_rank,
            )
        return self.logging_session

    def close_logging_session(self) -> None:
        """Closes the logging session, if one is open. Idempotent."""
        if self.logging_session is not None:
            self.logging_session.close()
            self.logging_session = None

    def ensure_hook_session(self, trainer: Any, pl_module: Any) -> None:
        """Bootstraps this run's hooks tree for one stage `_run`.

        Idempotent -- a no-op if a session from this same stage is already open.
        Called only from the hooks adapter's own `setup(trainer, pl_module, stage)`,
        which Lightning always calls before any other callback method for that
        `_run` and always pairs with a matching `teardown` (barring an uncaught
        exception, handled separately by `close_hook_session`). A hook that
        fires with no stage active at all (e.g. a standalone `Trainer.save_checkpoint`
        call outside `fit`/`validate`/`test`/`predict`) finds no session and is a
        silent no-op in `dispatch_hook` -- there is no run for it to belong to.
        Reads rank/world size directly from `trainer`: the runtime's rank info is not
        populated yet at this point (the adapter's `setup` runs before the composed
        model's own `setup`, which is what gathers the ranks).
        A resumed run dispatches with the session id that the marker records; the
        new id names only this stage's logging session and a freshly published tree.
        """
        if self.hook_session is not None:
            return
        logging_session = self.open_logging_session(trainer, pl_module)
        session_id = logging_session.session_id
        base_dir = default_base_dir(pl_module, self.run_id)
        log = logging_session.bind()
        shell = self.config.hooks_shell
        if shell is None:
            resolved = shutil.which("bash")
            if resolved is None:
                raise RuntimeError(
                    "midrun: hooks_enabled requires bash on PATH or an explicit hooks_shell"
                )
            shell = Path(resolved)
        setpriv = shutil.which("setpriv")
        tree = bootstrap_hooks_tree(
            base_dir, self.config.hooks_source_dir, session_id=session_id, log=log
        )
        self.hook_session = HookSession(
            tree.live_dir,
            session_id=tree.session_id,
            rank=trainer.global_rank,
            world_size=trainer.world_size,
            rank_policy=self.config.hooks_rank_policy,
            shell=shell,
            setpriv=None if setpriv is None else Path(setpriv),
            timeout_s=self.config.hooks_timeout_s,
            max_params_bytes=self.config.hooks_max_params_bytes,
            max_metadata_items=self.config.hooks_max_metadata_items,
            max_metadata_depth=self.config.hooks_max_metadata_depth,
            max_export_bytes=self.config.hooks_max_export_bytes,
            max_output_bytes=self.config.hooks_max_output_bytes,
            log=log,
        )

    def dispatch_hook(
        self,
        trainer: Any,
        hook: str,
        *,
        stage_override: str | None = None,
        arguments: dict[str, Any] | None = None,
        batch_idx: int | None = None,
        dataloader_idx: int | None = None,
    ) -> HandleResult | None:
        """Dispatches one Lightning callback occurrence through the live `HookSession`.

        A no-op if `ensure_hook_session` never installed a session (hooks disabled).
        """
        if self.hook_session is None:
            return None
        stage = (
            stage_override
            if stage_override is not None
            else trainer_stage_name(trainer)
        )
        return self.hook_session.handle(
            hook,
            stage=stage,
            epoch=trainer.current_epoch,
            global_step=trainer.global_step,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
            arguments=arguments,
        )

    def close_hook_session(self) -> None:
        """Closes the hooks session, if one is open.

        Idempotent. Called from the hooks adapter's own `teardown` on the normal
        path, and from its `on_exception` on the failure path, since Lightning skips
        the callback's own `teardown` after an uncaught exception.
        """
        if self.hook_session is not None:
            self.hook_session.close()
            self.hook_session = None
