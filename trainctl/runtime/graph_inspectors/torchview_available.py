"""Tells whether torchview can render graphs."""

import shutil

from trainctl.runtime.graph_inspectors.python_module_available import (
    python_module_available,
)


def torchview_available() -> bool:
    """Returns whether torchview and the `dot` binary are both present."""
    return python_module_available("torchview") and shutil.which("dot") is not None
