"""Budgets for bounded metadata and tensor capture in a hook session."""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class HookCaptureLimits:
    """Bound the payloads generated for scripts in both hook modalities.

    :ivar max_metadata_items: Maximum items visited per container.
    :ivar max_metadata_depth: Maximum container depth visited.
    :ivar max_params_bytes: Maximum encoded size of a light payload.
    :ivar max_export_bytes: Maximum predicted size of exported tensors.
    """

    max_metadata_items: int
    max_metadata_depth: int
    max_params_bytes: int
    max_export_bytes: int
