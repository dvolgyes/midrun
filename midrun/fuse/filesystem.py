"""Read-only mfusepy filesystem projecting a MidrunRuntime's state onto files.

All virtual files are generated fresh from the current `RuntimeSnapshot`/runtime
object on every `open()`; the mount uses FUSE direct I/O (see `RuntimeServices.start_fuse`)
so the kernel never serves a stale cached read. A generated file's bytes are
materialized once per `open()` and reused for the `read()` calls against that handle
(see `MidrunFS.open`/`read`/`release`), matching the open-handle cache policy.
While a path is open, `getattr` reports the length of the asked handle's bytes (of
the newest handle's for a path-only `stat`): the
kernel limits splice and sendfile reads (system `cat`, `shutil.copyfile`) to that size,
and the mount turns the kernel attribute cache off so `fstat` asks again.
Checkpoints, snapshots, and debug captures are real files on the regular filesystem
under `MidrunRuntime.artifacts.root` (a sibling of this mount, not projected into
it) -- see `meta/artifacts-path`.

This class is a thin mfusepy adapter: what each virtual path actually contains is
owned by `midrun.runtime.virtual_fs`, shared with the `GET /files/{path}` REST
route so the two transports never derive the same content twice.
"""

import errno
import os
import stat
import threading
import time
from typing import TYPE_CHECKING, Any

import mfusepy
from loguru import logger

from midrun.runtime.virtual_fs.is_live_file import is_live_file
from midrun.runtime.virtual_fs.is_virtual_dir import is_virtual_dir
from midrun.runtime.virtual_fs.list_virtual_dir import list_virtual_dir
from midrun.runtime.virtual_fs.virtual_file_content import virtual_file_content

if TYPE_CHECKING:
    from midrun.runtime.runtime import MidrunRuntime


class MidrunFS(mfusepy.Operations):
    """Read-only, direct-I/O mfusepy `Operations` backed by a `MidrunRuntime`."""

    use_ns = True
    _newest_handle: dict[str, int]

    def __init__(self, runtime: "MidrunRuntime") -> None:
        self._runtime = runtime
        self._ready = threading.Event()
        self._handles: dict[int, bytes] = {}
        self._newest_handle: dict[str, int] = {}
        self._next_fh = 1
        self._handles_lock = threading.Lock()

    def init(self, path: str) -> None:
        del path
        self._ready.set()

    def wait_ready(self, timeout: float) -> bool:
        """Blocks until libfuse has called `init()`, signalling the mount is live."""
        return self._ready.wait(timeout)

    def getattr(self, path: str, fh: int | None = None) -> dict[str, Any]:
        owner = {
            "st_uid": os.getuid(),
            "st_gid": os.getgid(),
            "st_mtime": int(time.time() * 1e9),
        }
        if is_virtual_dir(self._runtime, path):
            return {"st_mode": (stat.S_IFDIR | 0o755), "st_nlink": 2, **owner}
        with self._handles_lock:
            handle = self._newest_handle.get(path) if fh is None else fh
            content = self._handles.get(handle) if handle is not None else None
        if content is not None:
            return {
                "st_mode": (stat.S_IFREG | 0o444),
                "st_nlink": 1,
                "st_size": len(content),
                **owner,
            }
        if is_live_file(path):
            # st_size is unknowable without running the blocking capture; never trigger
            # it just to stat() the file -- only open()/read() do that.
            return {
                "st_mode": (stat.S_IFREG | 0o444),
                "st_nlink": 1,
                "st_size": 0,
                **owner,
            }
        content = self._generate(path)
        return {
            "st_mode": (stat.S_IFREG | 0o444),
            "st_nlink": 1,
            "st_size": len(content),
            **owner,
        }

    def readdir(self, path: str, fh: int) -> list[str]:
        del fh
        return [".", "..", *list_virtual_dir(self._runtime, path)]

    def open(self, path: str, flags: int) -> int:
        if flags & os.O_ACCMODE != os.O_RDONLY:
            raise mfusepy.FuseOSError(errno.EROFS)
        content = self._generate(path)
        with self._handles_lock:
            handle = self._next_fh
            self._next_fh += 1
            self._handles[handle] = content
            self._newest_handle[path] = handle
        return handle

    def read(self, path: str, size: int, offset: int, fh: int) -> bytes:
        del path
        with self._handles_lock:
            content = self._handles.get(fh, b"")
        return content[offset : offset + size]

    def release(self, path: str, fh: int) -> int:
        with self._handles_lock:
            self._handles.pop(fh, None)
            if self._newest_handle.get(path) == fh:
                del self._newest_handle[path]
        return 0

    def statfs(self, path: str) -> dict[str, int]:
        del path
        return {}

    def _generate(self, path: str) -> bytes:
        """Generates `path`'s bytes, turning a missing or failing file into its errno."""
        try:
            content = virtual_file_content(self._runtime, path)
        except TimeoutError as exc:
            logger.warning("midrun FUSE read of {} timed out: {}", path, exc)
            raise mfusepy.FuseOSError(errno.ETIMEDOUT) from exc
        except Exception as exc:
            logger.warning("midrun FUSE read of {} failed: {}", path, exc)
            raise mfusepy.FuseOSError(errno.EIO) from exc
        if content is None:
            raise mfusepy.FuseOSError(errno.ENOENT)
        return content
