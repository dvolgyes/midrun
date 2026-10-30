"""Builds a `MidrunRuntime` with the REST app and FUSE filesystem that serve it."""

from typing import TYPE_CHECKING

from midrun.config import MidrunConfig
from midrun.lightning_backend import LightningBackend
from midrun.runtime.runtime import MidrunRuntime

if TYPE_CHECKING:
    from fastapi import FastAPI

    from midrun.fuse.filesystem import MidrunFS


def build_runtime(config: MidrunConfig, backend: LightningBackend) -> MidrunRuntime:
    """Returns a runtime whose services serve it through `midrun.rest` and `midrun.fuse`.

    The runtime does not import these layers, because they read the runtime. Each
    layer is imported only when its service starts: the FUSE import loads libfuse.

    Args:
        config: Effective `MidrunConfig` for this run.
        backend: The Lightning backend the composed model uses.
    """

    def build_rest_app(runtime: MidrunRuntime) -> "FastAPI":
        """Builds the REST app that serves `runtime`."""
        from midrun.rest.app import build_app

        return build_app(runtime)

    def build_fuse_ops(runtime: MidrunRuntime) -> "MidrunFS":
        """Builds the FUSE operations that serve `runtime`."""
        from midrun.fuse.filesystem import MidrunFS

        return MidrunFS(runtime)

    return MidrunRuntime(
        config, backend, build_rest_app=build_rest_app, build_fuse_ops=build_fuse_ops
    )
