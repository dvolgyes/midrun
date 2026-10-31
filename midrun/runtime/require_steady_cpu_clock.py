"""Refuses a feature whose times torch reads from a CPU clock that differs between CPUs."""

from collections.abc import Callable

import torch

from midrun.runtime.read_clock_on_each_cpu import read_clock_on_each_cpu

_TOLERANCE_NS = 1_000_000
# torch.profiler and the CUDA allocator history time events with this clock, which
# reads the CPU's time-stamp counter on x86.
# pylint: disable-next=protected-access,c-extension-no-member
_TORCH_CLOCK: Callable[[], int] = torch._C._profiler._get_approximate_time


def require_steady_cpu_clock(
    feature: str, *, read_clock: Callable[[], int] = _TORCH_CLOCK
) -> None:
    """Raises when `read_clock` differs between the CPUs that this thread can run on.

    On some machines the time-stamp counters of the CPUs are out of step, and the
    times of a thread that moves between CPUs are wrong by up to that difference.
    Each CPU's offset of `read_clock` against the monotonic clock is compared with
    the first CPU's; a thread that can run on one CPU only is not checked.

    Args:
        feature: The name of the feature, for the error message.
        read_clock: The clock to compare, in ticks.

    Raises:
        RuntimeError: The clock differs by more than 1 ms between two CPUs, or it
            does not advance.
    """
    readings = read_clock_on_each_cpu(read_clock)
    if not readings:
        return
    (ref_cpu, ref_ns, ref_ticks), *others, (_, end_ns, end_ticks) = readings
    ticks_per_ns = (end_ticks - ref_ticks) / (end_ns - ref_ns)
    if ticks_per_ns <= 0:
        raise RuntimeError(
            f"{feature} is disabled: the CPU clock that torch reads did not advance "
            f"on CPU {ref_cpu}, so the recorded times would be wrong"
        )
    skews = {
        cpu: abs((ticks - ref_ticks) / ticks_per_ns - (ns - ref_ns))
        for cpu, ns, ticks in others
    }
    off = [cpu for cpu, skew in skews.items() if skew > _TOLERANCE_NS]
    if off:
        raise RuntimeError(
            f"{feature} is disabled: the CPU clock that torch reads differs by up to "
            f"{max(skews[cpu] for cpu in off) / 1e6:.1f} ms between CPU {ref_cpu} and "
            f"CPUs {', '.join(map(str, off))} of this thread, usually because the "
            "CPUs' time-stamp counters are out of step, so the recorded times would "
            "be wrong. Run on CPUs whose clocks agree, for example with taskset, or "
            "update the firmware."
        )
