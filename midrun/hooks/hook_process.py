"""Session-scoped process settings for launching a hook script."""

from dataclasses import dataclass
from pathlib import Path

import loguru

from midrun.hooks.runner import ExecutionResult, run_script


@dataclass(frozen=True, kw_only=True)
class HookProcess:
    """Launch hook processes under the session's execution policy.

    :ivar shell: Executable used to run the script.
    :ivar setpriv: Optional executable that sets the parent-death signal.
    :ivar timeout_s: Deadline for the script, or None for no deadline.
    :ivar max_output_bytes: Maximum output retained by the runner.
    :ivar log: Session logger receiving process and capture diagnostics.
    """

    shell: Path
    setpriv: Path | None
    timeout_s: float | None
    max_output_bytes: int
    log: "loguru.Logger"

    def run(self, script: Path, payload: Path, *, cwd: Path) -> ExecutionResult:
        """Run a script with its payload path as the sole argument."""
        return run_script(
            script,
            [str(payload)],
            cwd=cwd,
            shell=self.shell,
            setpriv=self.setpriv,
            timeout_s=self.timeout_s,
            max_output_bytes=self.max_output_bytes,
            log=self.log,
        )
