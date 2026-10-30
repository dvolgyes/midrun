"""Tells whether torchviz can render graphs."""

import shutil

from midrun.runtime.graph_inspectors.python_module_available import (
    python_module_available,
)


def torchviz_available() -> bool:
    """Returns whether torchviz and the `dot` binary are both present."""
    return python_module_available("torchviz") and shutil.which("dot") is not None
