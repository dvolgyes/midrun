"""The content of a virtual file."""

from typing import TYPE_CHECKING

from trainctl.runtime.virtual_fs.live_files import LIVE_FILES
from trainctl.runtime.virtual_fs.rank_file_content import rank_file_content
from trainctl.runtime.virtual_fs.static_files import STATIC_FILES
from trainctl.runtime.virtual_fs.worker_file_content import worker_file_content

if TYPE_CHECKING:
    from trainctl.runtime.runtime import TrainctlRuntime


def virtual_file_content(runtime: "TrainctlRuntime", path: str) -> bytes | None:
    """Returns `path`'s current content, or `None` if `path` names no virtual file.

    Raises:
        TimeoutError: `path` is a live file and no safe point was reached in time.
    """
    generator = STATIC_FILES.get(path)
    if generator is not None:
        return generator(runtime)
    live_generator = LIVE_FILES.get(path)
    if live_generator is not None:
        return live_generator(runtime)
    rank_content = rank_file_content(runtime, path)
    if rank_content is not None:
        return rank_content
    return worker_file_content(runtime, path)
