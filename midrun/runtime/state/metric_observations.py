"""Tracks the identity of the Lightning cache behind each published callback scalar."""

from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING
from weakref import ReferenceType, ref

from torch import Tensor, strided

if TYPE_CHECKING:
    from lightning.pytorch.trainer.connectors.logger_connector.result import (
        _ResultCollection,
    )


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
    if on_step or results.training or results.dataloader_idx is not None:
        return sources
    module = getattr(trainer, "lightning_module", None)
    if getattr(module, "_current_fx_name", None) not in (
        "on_validation_epoch_end",
        "on_test_epoch_end",
    ):
        return sources

    sources.update(_retained_sources(results, connector._callback_metrics, sources))
    return sources


def _retained_sources(
    results: "_ResultCollection",
    published: Mapping[str, object],
    active: Collection[str],
) -> dict[str, _Source]:
    """Match published tensor views to loader caches in publication order."""
    matches: dict[str, tuple[int, _Source]] = {}
    for metric in results.values():
        loader_idx = metric.meta.dataloader_idx
        if (
            metric.has_reset
            or loader_idx is None
            or not metric.meta.on_epoch
            or not isinstance(cache := metric._computed, Tensor)
        ):
            continue
        source = (ref(metric), metric.update_count, False, ref(cache))
        for key in results._forked_name(metric, False):
            if key in active or (key in matches and matches[key][0] > loader_idx):
                continue
            value = published.get(key)
            if cache is value or (
                isinstance(value, Tensor)
                and cache.layout == value.layout == strided
                and cache.is_set_to(value)
            ):
                matches[key] = loader_idx, source
    return {key: source for key, (_, source) in matches.items()}


class MetricObservations:  # pylint: disable=too-few-public-methods
    """Tracks Lightning updates without keeping a metric or device tensor alive.

    :ivar _sources: Last known producer and cache identity for each logged name.
    """

    _sources: dict[str, _Source]

    def __init__(self) -> None:
        self._sources = {}

    def fresh(self, trainer: object, keys: Collection[str]) -> set[str]:
        """Returns newly observed keys; names never attributed to Lightning are always fresh.

        Lightning 2.6 detaches the cached tensor on every callback_metrics read.
        Follow the result and its update counter, not the detached tensor.
        At evaluation epoch end, the retained callback mapping identifies caches
        published earlier for individual dataloaders.
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
