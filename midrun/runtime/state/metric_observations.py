"""Tracks the identity of the Lightning cache behind each published callback scalar."""

from collections.abc import Collection
from weakref import ReferenceType, ref

from torch import Tensor


type _Source = tuple[ReferenceType[object], int, bool, ReferenceType[Tensor]]


def _active_sources(trainer: object) -> dict[str, _Source]:
    results = getattr(trainer, "_results", None)
    connector = getattr(trainer, "_logger_connector", None)
    sources: dict[str, _Source] = {}
    if not results or connector is None:
        return sources
    on_step = connector._first_loop_iter is not None
    for _, metric in results.valid_items():
        if on_step and not (results.training and metric.meta.on_step):
            continue
        if not on_step and not metric.meta.on_epoch:
            continue
        cache = metric._forward_cache if on_step else metric._computed
        if isinstance(cache, Tensor):
            source = (ref(metric), metric.update_count, on_step, ref(cache))
            for key in results._forked_name(metric, on_step):
                sources[key] = source
    return sources


class MetricObservations:
    """Tracks Lightning updates without keeping a metric or device tensor alive.

    :ivar _sources: Last known producer and cache identity for each logged name.
    """

    _sources: dict[str, _Source]

    def __init__(self) -> None:
        self._sources = {}

    def fresh(self, trainer: object, keys: Collection[str]) -> set[str]:
        """Returns newly observed keys; names never attributed to Lightning are always fresh.

        Lightning 2.6 detaches the cached tensor on every callback_metrics read.
        Follow the active result and its update counter, not the detached tensor.
        A name absent from active results can still refer to a retained cache.
        """
        sources = _active_sources(trainer)
        fresh: set[str] = set()
        for key in keys:
            source = sources.get(key)
            previous = self._sources.get(key)
            if source is None and previous is not None:
                sources[key] = previous
            elif (
                previous is None
                or source is None
                or source[0]() is not previous[0]()
                or source[1:3] != previous[1:3]
                or source[3]() is not previous[3]()
            ):
                fresh.add(key)
        self._sources = sources
        return fresh
