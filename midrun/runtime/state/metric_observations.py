"""Tracks the identity of the Lightning cache behind each published callback scalar."""

from collections.abc import Callable, Collection, Mapping
from math import isnan
from typing import TYPE_CHECKING, NamedTuple
from weakref import ReferenceType, ref

from torch import Tensor, strided

from midrun.runtime.state.host_scalar import HostScalar
from midrun.runtime.state.publishable_value import publishable_value

if TYPE_CHECKING:
    from lightning.pytorch.trainer.connectors.logger_connector.result import (
        _ResultCollection,
    )

    from midrun.runtime.state.scalar_metric import ScalarMetric


class _Source(NamedTuple):
    """Identifies one Lightning result update.

    :ivar producer: Weak reference to the Lightning result metric.
    :ivar update_count: Result metric update count at publication.
    :ivar on_step: Whether Lightning used the step cache.
    :ivar cache: Weak reference to the tensor before Lightning detached it.
    """

    producer: ReferenceType[object]
    update_count: int
    on_step: bool
    cache: ReferenceType[Tensor]

    def same_update(self, previous: "_Source") -> bool:
        """Checks whether both sources identify the same Lightning update."""
        return (
            self.producer() is previous.producer()
            and self.update_count == previous.update_count
            and self.on_step == previous.on_step
            and self.cache() is previous.cache()
        )

    def matches_view(self, value: object) -> bool:
        """Checks whether the callback value is the cache or its detached view."""
        cache = self.cache()
        return cache is not None and (
            cache is value
            or (
                isinstance(value, Tensor)
                and cache.layout == value.layout == strided
                and cache.is_set_to(value)
            )
        )


class _Observation(NamedTuple):
    """Pairs a Lightning source with a published callback tensor.

    :ivar source: Lightning result update that produced the tensor.
    :ivar callback: Weak reference to the published tensor.
    """

    source: _Source
    callback: ReferenceType[Tensor]

    def retains(self, value: Tensor) -> bool:
        """Checks whether the callback still holds the same publication."""
        return self.callback() is value or self.source.matches_view(value)

    @staticmethod
    def matches_teardown_copy(value: Tensor, previous: "ScalarMetric | None") -> bool:
        """Matches a CUDA publication copied to CPU by Lightning fit teardown."""
        if (
            previous is None
            or not isinstance(before := previous.value, HostScalar)
            or value.device.type != "cpu"
            or value.dtype != before.host.dtype
        ):
            return False
        current = publishable_value(value)
        ready: Callable[[], bool] = before.ready.query
        if current is None or not ready():
            return False
        original = float(before.host[before.index])
        return current == original or (isnan(current) and isnan(original))


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
            source = _Source(ref(metric), metric.update_count, on_step, ref(cache))
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
        source = _Source(ref(metric), metric.update_count, False, ref(cache))
        for key in results._forked_name(metric, False):
            if key in active or (key in matches and matches[key][0] > loader_idx):
                continue
            if source.matches_view(published.get(key)):
                matches[key] = loader_idx, source
    return {key: source for key, (_, source) in matches.items()}


class MetricObservations:  # pylint: disable=too-few-public-methods
    """Tracks Lightning updates without keeping a metric or device tensor alive.

    :ivar _observations: Last Lightning source and callback tensor for each logged name.
    """

    _observations: dict[str, _Observation]

    def __init__(self) -> None:
        self._observations = {}

    def fresh(
        self,
        trainer: object,
        published: Mapping[str, object],
        previous: Mapping[str, "ScalarMetric"],
    ) -> set[str]:
        """Returns newly observed keys; names never attributed to Lightning are always fresh.

        An active Lightning result can publish a new detached view on every read.
        Follow the result and its update counter while it is active. Once it is
        inactive, only the same callback tensor or a view of its cache is retained.
        At evaluation epoch end, the retained callback mapping identifies caches
        published earlier for individual dataloaders.
        Before `on_fit_end`, Lightning copies the callback tensors to CPU.
        The copy is indistinguishable from an equal replacement at that boundary.
        """
        sources = _active_sources(trainer)
        module = getattr(trainer, "lightning_module", None)
        after_teardown = getattr(module, "_current_fx_name", None) == "on_fit_end"
        observations: dict[str, _Observation] = {}
        fresh: set[str] = set()
        for key, value in published.items():
            source = sources.get(key)
            prior = self._observations.get(key)
            if not isinstance(value, Tensor):
                fresh.add(key)
                continue
            if (
                source is None
                and prior is not None
                and (
                    prior.retains(value)
                    or (
                        after_teardown
                        and prior.matches_teardown_copy(value, previous.get(key))
                    )
                )
            ):
                source = prior.source
            if source is None or prior is None or not source.same_update(prior.source):
                fresh.add(key)
            if source is not None:
                observations[key] = _Observation(source, ref(value))
        self._observations = observations
        return fresh
