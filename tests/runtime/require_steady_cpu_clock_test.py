"""Behavioral tests for the refusal of features whose CPU clock differs between CPUs."""

import os
import time
from collections.abc import Callable

import pytest

from midrun.runtime.require_steady_cpu_clock import require_steady_cpu_clock

pytestmark = pytest.mark.skipif(
    not hasattr(os, "sched_getaffinity") or len(os.sched_getaffinity(0)) < 2,
    reason="needs a thread that can run on at least two CPUs",
)

_SKEW_NS = 5_000_000_000


def _clock_skewed_on(cpu: int | None) -> Callable[[], int]:
    def read() -> int:
        skewed = cpu is not None and os.sched_getaffinity(0) == {cpu}
        return time.monotonic_ns() + (_SKEW_NS if skewed else 0)

    return read


def test_a_steady_clock_is_accepted_and_the_affinity_is_kept() -> None:
    cpus = os.sched_getaffinity(0)

    require_steady_cpu_clock("profile", read_clock=_clock_skewed_on(None))

    assert os.sched_getaffinity(0) == cpus


@pytest.mark.parametrize("position", [0, 1], ids=["reference cpu", "other cpu"])
def test_a_clock_that_differs_on_one_cpu_disables_the_feature(position: int) -> None:
    cpus = os.sched_getaffinity(0)
    skewed_cpu = sorted(cpus)[position]

    with pytest.raises(RuntimeError, match="profile is disabled") as error:
        require_steady_cpu_clock("profile", read_clock=_clock_skewed_on(skewed_cpu))

    assert "taskset" in str(error.value)
    assert os.sched_getaffinity(0) == cpus


def test_a_thread_on_one_cpu_is_not_checked() -> None:
    cpus = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(cpus)})
    try:
        require_steady_cpu_clock("profile", read_clock=_clock_skewed_on(min(cpus)))
    finally:
        os.sched_setaffinity(0, cpus)
