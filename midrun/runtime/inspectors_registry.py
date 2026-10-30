"""Shared availability discovery for the optional-package inspectors.

One place computing `{name: {"available": bool}}`, called from both `GET /inspectors`
and the FUSE `/model/inspectors.json` projection so availability logic is never
duplicated between the two surfaces.
"""

from midrun.runtime.graph_inspectors.torchview_available import (
    torchview_available,
)
from midrun.runtime.graph_inspectors.torchviz_available import torchviz_available
from midrun.runtime.torchinfo_inspector.torchinfo_available import (
    torchinfo_available,
)
from midrun.runtime.torchlens_available import torchlens_available


def availability_snapshot() -> dict[str, dict[str, bool]]:
    """Returns each optional-package inspector's current availability."""
    return {
        "torchinfo": {"available": torchinfo_available()},
        "torchview": {"available": torchview_available()},
        "torchviz": {"available": torchviz_available()},
        "torchlens": {"available": torchlens_available()},
    }
