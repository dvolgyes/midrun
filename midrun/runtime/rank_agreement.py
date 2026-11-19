"""Agrees on local command failures across participating ranks."""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import TracebackType

from midrun.runtime.rank_sync import RankSync


@dataclass(slots=True)
class RankAgreement(AbstractContextManager["RankAgreement"]):
    """Delay completion until every participating rank has left the operation.

    :ivar rank_sync: Control group containing the participating ranks.
    :ivar operation: Name to include when reporting a local failure.
    """

    rank_sync: RankSync
    operation: str

    @staticmethod
    def agreed_error(error: BaseException) -> str | None:
        """Returns the rank-tagged failure carried through nested phases."""
        message = error.__dict__.get("_midrun_agreed_error")
        return message if isinstance(message, str) else None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Report rank failures and propagate interruptions on all ranks."""
        if self.rank_sync.world_size == 1:
            return

        report = None
        if exc is not None:
            report = (
                self.agreed_error(exc),
                f"{type(exc).__name__}: {exc}",
                not isinstance(exc, Exception),
            )
        failures: dict[str, None] = {}
        interrupted = False
        for rank, failure in enumerate(self.rank_sync.all_gather(report)):
            if failure is not None:
                agreed, detail, termination = failure
                message = agreed or f"{self.operation} failed on rank {rank}: {detail}"
                failures[message] = None
                interrupted |= termination
        if not failures:
            return
        message = "; ".join(failures)
        if exc is not None and not isinstance(exc, Exception):
            exc.__dict__["_midrun_agreed_error"] = message
            return
        error = KeyboardInterrupt(message) if interrupted else RuntimeError(message)
        error.__dict__["_midrun_agreed_error"] = message
        raise error
