"""Starts and stops the REST server, the FUSE mount and the torch debug server."""

import socket
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from midrun.config import MidrunConfig
from midrun.fuse import mountpoint_check, process_guard
from midrun.runtime.bind_free_socket import bind_free_socket
from midrun.runtime.bind_socket import bind_socket
from midrun.runtime.rest_base_url import rest_base_url
from midrun.runtime.start_rest_server import start_rest_server
from midrun.runtime.state.state_store import StateStore
from midrun.torch_debug.manager import TorchDebugManager

# REST and torch_debug candidates at attempt N are both `N * _PORT_OFFSET_STEP`
# above their configured base port (e.g. offset 200 pairs REST 8290 with debug 26199).
_PORT_OFFSET_STEP = 100


class RuntimeServices:
    """Owns the threads and servers that serve a run to REST and FUSE readers.

    `MidrunRuntime` decides which services start on which rank; this class holds
    the mechanics of each start and stop and the handles that a stop needs.

    Attributes:
        config: Effective `MidrunConfig` for this run.
        state: Published snapshot store that receives the service ports and URLs.
        torch_debug: PyTorch distributed debug server manager.
        rest_port: The bound REST port, once started.
        fuse_mountpoint: The active FUSE mountpoint, once started.
    """

    config: MidrunConfig
    state: StateStore
    torch_debug: TorchDebugManager
    rest_port: int | None
    fuse_mountpoint: Path | None
    _run_id: str
    _build_app: Callable[[], Any]
    _build_fs: Callable[[], Any]
    _rest_server: Any
    _rest_thread: threading.Thread | None
    _fuse_ops: Any
    _fuse_thread: threading.Thread | None
    _fuse_watcher: Any

    def __init__(
        self,
        config: MidrunConfig,
        state: StateStore,
        torch_debug: TorchDebugManager,
        *,
        run_id: str,
        build_app: Callable[[], Any],
        build_fs: Callable[[], Any],
    ) -> None:
        self.config = config
        self.state = state
        self.torch_debug = torch_debug
        self._run_id = run_id
        self._build_app = build_app
        self._build_fs = build_fs
        self.rest_port = None
        self.fuse_mountpoint = None
        self._rest_server = None
        self._rest_thread = None
        self._fuse_ops = None
        self._fuse_thread = None
        self._fuse_watcher = None

    @property
    def rest_url(self) -> str | None:
        """The reachable REST base URL, once started."""
        if self.rest_port is None:
            return None
        return rest_base_url(self.config.rest_host, self.rest_port)

    def start_torch_debug(self) -> None:
        """Starts the torch debug server alone and publishes its URL."""
        self.torch_debug.start_after_distributed_init()
        self.state.update(torch_debug_url=self.torch_debug.ready_url(timeout=10))

    def start_rest_and_torch_debug(self) -> None:
        """Starts REST and the torch_debug frontend as one paired unit: both ports are
        offset from their configured base port by the same multiple of `_PORT_OFFSET_STEP`
        (e.g. offset 200 pairs REST 8290 with torch_debug 26199) -- a human-legible
        pairing, and a way to keep the two services from independently drifting to
        unrelated ports when several runs share a host and start at nearly the same time.

        Per offset attempt: probe the torch_debug candidate (it can't be held -- see
        `TorchDebugManager.start_after_distributed_init`), then bind-and-hold the REST
        candidate at the same offset (race-free once held), then re-probe the
        torch_debug candidate immediately before starting it, narrowing (not
        eliminating -- an upstream constraint) its check-then-bind race to that final
        gap. A collision at either check abandons the offset and retries the next one.
        """
        max_attempts = self.config.paired_port_search_attempts
        for attempt in range(max_attempts):
            offset = attempt * _PORT_OFFSET_STEP
            debug_candidate = self.config.torch_debug_port + offset
            if not self.torch_debug.port_is_free(debug_candidate):
                continue
            try:
                sock = bind_socket(
                    self.config.rest_host, self.config.rest_port + offset
                )
            except OSError:
                continue
            if not self.torch_debug.port_is_free(debug_candidate):
                sock.close()
                continue
            try:
                self.torch_debug.start_after_distributed_init(port=debug_candidate)
                debug_url = self.torch_debug.ready_url(timeout=10)
            except Exception:
                # REST still starts, so the run stays controllable without the debug server.
                self._start_rest_on_socket(sock, self.config.rest_port + offset)
                raise
            self.state.update(torch_debug_url=debug_url)
            self._start_rest_on_socket(sock, self.config.rest_port + offset)
            return
        raise RuntimeError(
            f"no free paired REST/torch_debug port offset found in {max_attempts} "
            f"attempts (step {_PORT_OFFSET_STEP})"
        )

    def start_rest(self) -> None:
        """Starts REST alone on the first free port of its search range."""
        sock, port = bind_free_socket(
            self.config.rest_host, self.config.rest_port, self.config.rest_port_search
        )
        self._start_rest_on_socket(sock, port)

    def _start_rest_on_socket(self, sock: socket.socket, port: int) -> None:
        server, thread = start_rest_server(
            self._build_app(), sock, self.config.rest_host, port, self._run_id
        )
        self._rest_server = server
        self._rest_thread = thread
        self.rest_port = port
        self.state.update(rest_host=self.config.rest_host, rest_port=port)

    def stop_rest(self) -> None:
        """Stops the REST server, if it runs, and warns when its thread stays alive."""
        if self._rest_server is None:
            return
        self._rest_server.should_exit = True
        if self._rest_thread is not None:
            self._rest_thread.join(timeout=5)
            if self._rest_thread.is_alive():
                logger.warning("midrun: the REST server thread did not stop in 5 s")
        self._rest_server = None
        self._rest_thread = None
        self.rest_port = None

    def start_fuse(self, base_dir: Path) -> None:
        """Mounts the read-only FUSE filesystem under `base_dir`, or the configured path."""
        import mfusepy

        mountpoint = mountpoint_check.prepare_mountpoint(
            self.config.fuse_mountpoint or base_dir / "procfs"
        )
        ops = self._build_fs()

        def _run() -> None:
            mfusepy.FUSE(
                ops,
                str(mountpoint),
                foreground=True,
                direct_io=True,
                ro=True,
                attr_timeout=0,
                fsname=mountpoint_check.FUSE_FSNAME,
            )

        thread = threading.Thread(target=_run, name="midrun-fuse", daemon=True)
        self._fuse_watcher = process_guard.mount_guarded(
            mountpoint, thread, ops.wait_ready
        )
        self._fuse_ops = ops
        self._fuse_thread = thread
        self.fuse_mountpoint = mountpoint

    def stop_fuse(self) -> None:
        """Unmounts the FUSE filesystem, if mounted, and stops its watcher."""
        if self._fuse_thread is None:
            return
        import subprocess

        subprocess.run(["fusermount3", "-u", str(self.fuse_mountpoint)], check=False)  # noqa: S603, S607 -- own mount
        self._fuse_thread.join(timeout=5)
        if self._fuse_watcher is not None:
            # A byte on stdin tells the watcher to check once and exit.
            self._fuse_watcher.communicate(b"\n", timeout=5)
        self._fuse_ops = None
        self._fuse_thread = None
        self.fuse_mountpoint = None
        self._fuse_watcher = None
