"""Lifecycle manager for `torch.distributed.debug`'s experimental HTTP server.

`torch.distributed.debug.start_debug_server()` is a post-process-group-initialization
API: it reads `RANK`, `WORLD_SIZE`, `MASTER_ADDR` and `MASTER_PORT` from the environment
and starts a worker server on every rank plus a frontend HTTP server on rank 0. It
requires `jinja2` and `aiohttp`, is version-sensitive, and is not always available
(e.g. a launcher without `MASTER_ADDR`, missing optional deps). All of that is
isolated here so the rest of Trainctl depends only on this manager's normalized
surface, never on `torch.distributed.debug` directly.
"""

import os
import time
import urllib.request


class TorchDebugManager:
    """Starts, tracks readiness of, and stops the PyTorch distributed debug server.

    Attributes:
        port: First frontend port to try.
        port_search: Number of sequential ports to try if `port` is taken.
    """

    def __init__(self, port: int, port_search: int) -> None:
        self.port = port
        self.port_search = port_search
        self._started = False
        self._selected_port: int | None = None

    @property
    def frontend_url(self) -> str | None:
        """The reachable frontend URL, once started on rank 0; None on other ranks."""
        if self._selected_port is None:
            return None
        return f"http://127.0.0.1:{self._selected_port}"

    def start_after_distributed_init(self, port: int | None = None) -> None:
        """Starts the debug server stack. Must be called after Lightning's strategy has
        initialized `torch.distributed` (see `TrainctlMixin.setup`), never earlier.

        `torch.distributed.debug.start_debug_server()` records its worker/frontend
        handles in module-level globals and asserts on a second call regardless of which
        port is passed, so a failed attempt cannot be retried with a different port
        in-process. Port selection therefore uses a preflight bind check (a known
        check-then-bind race, accepted here because it is strictly safer than the
        alternative of calling the upstream API more than once) rather than
        try-then-fall-back.

        Every rank must call it, because every rank serves its own worker server; only
        rank 0 serves the frontend on the selected port. Lightning's own DDP launcher
        sets `WORLD_SIZE` but not `RANK`, so a missing `RANK` is supplied from the
        process group for the duration of the upstream call.

        Args:
            port: Use this exact port instead of searching `[port, port + port_search)`
                internally. Intended for a caller (`TrainctlRuntime`) that has already
                paired this port's selection with another service's (REST) and just
                re-verified it with `port_is_free`, keeping the check-then-bind gap as
                small as possible; falls back to `_select_free_port` when omitted, for
                standalone use with no REST server to pair with.

        Raises:
            RuntimeError: `torch.distributed` is not initialized, the launch environment
                does not provide `WORLD_SIZE`, `MASTER_ADDR` or `MASTER_PORT`, or no
                candidate port passed the preflight bind check.

        Note:
            Two instances launched at nearly the same moment with the same requested
            `port` scan candidates in the same deterministic order, so a plain
            `range(port, port + port_search)` scan tends to converge on the *same*
            "first free" candidate rather than merely risking it -- observed directly
            under a two-parallel-runs test. `_select_free_port` starts its scan from a
            randomized offset into the range to decorrelate simultaneous scans; this
            narrows the window rather than closing it (the check-then-bind gap
            described above is unavoidable given the upstream constraint).
        """
        if self._started:
            return

        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed is not initialized")

        missing = [
            name
            for name in ("WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
            if name not in os.environ
        ]
        if missing:
            raise RuntimeError(
                "torch.distributed.debug needs the launch environment variables "
                f"{', '.join(missing)}, which this launcher does not set"
            )

        from torch.distributed import debug as torch_debug

        is_frontend = dist.get_rank() == 0
        if port is not None:
            candidate = port
        else:
            candidate = self._select_free_port() if is_frontend else self.port
        supplied_rank = "RANK" not in os.environ
        if supplied_rank:
            os.environ["RANK"] = str(dist.get_rank())
        try:
            torch_debug.start_debug_server(port=candidate, worker_port=0, dump_dir=None)
        except Exception:
            # The upstream start creates the worker server before it can fail, and a
            # leftover worker server makes every later start assert.
            # pylint: disable=protected-access
            leaked = torch_debug._WORKER_SERVER
            if leaked is not None and torch_debug._DEBUG_SERVER_PROC is None:
                leaked.shutdown()
                torch_debug._WORKER_SERVER = None
            # pylint: enable=protected-access
            raise
        finally:
            if supplied_rank:
                del os.environ["RANK"]
        self._selected_port = candidate if is_frontend else None
        self._started = True

    @staticmethod
    def port_is_free(port: int) -> bool:
        """Preflight-probes `port` via bind-then-release.

        Check-then-bind race; see `start_after_distributed_init`'s docstring for why
        this is accepted here.
        """
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("0.0.0.0", port))  # noqa: S104 -- preflight probe only, socket closed immediately
                return True
            except OSError:
                return False

    def _select_free_port(self) -> int:
        import random

        offset = random.randrange(self.port_search)  # noqa: S311 -- decorrelating scan order, not security-sensitive
        for step in range(self.port_search):
            candidate = self.port + (offset + step) % self.port_search
            if self.port_is_free(candidate):
                return candidate
        raise RuntimeError(
            f"no free port for torch distributed debug frontend in [{self.port}, {self.port + self.port_search})"
        )

    def wait_ready(self, timeout: float = 10.0) -> bool:
        """Polls the frontend URL until it answers HTTP requests or `timeout` elapses."""
        if self.frontend_url is None:
            return False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(self.frontend_url, timeout=1)  # noqa: S310 -- localhost-only debug frontend
                return True
            except Exception:  # noqa: BLE001 -- readiness probe boundary, not business logic
                time.sleep(0.2)
        return False

    def stop(self) -> None:
        """Stops the debug server stack. Idempotent."""
        if not self._started:
            return
        from torch.distributed import debug as torch_debug

        try:
            if self._selected_port is not None:
                torch_debug.stop_debug_server()
            else:
                # stop_debug_server() asserts that this process has the frontend, which
                # only rank 0 has, so the other ranks stop their worker server directly.
                # pylint: disable=protected-access
                torch_debug._WORKER_SERVER.shutdown()
                torch_debug._WORKER_SERVER = None
                # pylint: enable=protected-access
        finally:
            self._started = False
            self._selected_port = None
