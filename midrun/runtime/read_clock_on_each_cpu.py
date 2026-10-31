"""Reads a clock on each CPU that the calling thread can run on."""

import os
import time
from collections.abc import Callable

_SAMPLES_PER_CPU = 5
_RATE_INTERVAL_S = 0.01


def read_clock_on_each_cpu(read_clock: Callable[[], int]) -> list[tuple[int, int, int]]:
    """Returns `(cpu, monotonic_ns, ticks)` readings of `read_clock` on each CPU of this thread.

    The thread runs on each of its CPUs in turn, then on the first CPU again after
    a short sleep, so that the first and last readings give the clock's rate. Of a
    few samples per CPU, the one with the shortest monotonic bracket is kept, at the
    bracket's midpoint. Then the thread's CPUs are restored.

    Returns:
        The readings in that order, or none when this thread can run on one CPU only
        or the system cannot pin a thread to a CPU.
    """
    if not hasattr(os, "sched_setaffinity"):
        return []
    cpus = os.sched_getaffinity(0)
    if len(cpus) < 2:
        return []
    order = sorted(cpus)
    readings = []
    try:
        for index, cpu in enumerate([*order, order[0]]):
            os.sched_setaffinity(0, {cpu})
            if index == len(order):
                time.sleep(_RATE_INTERVAL_S)
            samples = []
            for _ in range(_SAMPLES_PER_CPU):
                before = time.monotonic_ns()
                ticks = read_clock()
                after = time.monotonic_ns()
                samples.append((after - before, (before + after) // 2, ticks))
            _, ns, ticks = min(samples)
            readings.append((cpu, ns, ticks))
    finally:
        os.sched_setaffinity(0, cpus)
    return readings
