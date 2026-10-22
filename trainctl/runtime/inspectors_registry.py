"""Shared availability discovery for the optional-package inspectors.

One place computing `{name: {"available": bool}}`, called from both `GET /inspectors`
and the FUSE `/model/inspectors.json` projection so availability logic is never
duplicated between the two surfaces.
"""

from trainctl.runtime import graph_inspectors
from trainctl.runtime.torchinfo_inspector.torchinfo_available import (
    torchinfo_available,
)
from trainctl.runtime.torchlens_available import torchlens_available


def availability_snapshot() -> dict[str, dict[str, bool]]:
    """Returns each optional-package inspector's current availability."""
    return {
        "torchinfo": {"available": torchinfo_available()},
        "torchview": {"available": graph_inspectors.torchview_available()},
        "torchviz": {"available": graph_inspectors.torchviz_available()},
        "torchlens": {"available": torchlens_available()},
    }
