"""Warns when the kernel found the CPU time-stamp counter (TSC) unstable."""

from pathlib import Path

_AVAILABLE_CLOCKSOURCE = Path(
    "/sys/devices/system/clocksource/clocksource0/available_clocksource"
)
_X86_MACHINES = frozenset({"x86_64", "amd64", "i386", "i686"})


def tsc_clock_warning(
    machine: str, *, available_clocksource: Path = _AVAILABLE_CLOCKSOURCE
) -> str | None:
    """Returns a warning when an x86 Linux kernel does not offer the TSC as a clocksource.

    The kernel removes the TSC when its boot check finds the counters of the CPUs
    out of step. torch.profiler still reads the TSC, so the event times of a thread
    that moves between such CPUs are wrong, and the event tree can nest so deeply
    that the summary fails.

    Args:
        machine: The CPU architecture, as `platform.machine()` gives it.
        available_clocksource: The kernel's list of clocksources; a missing file,
            as on other systems, gives no warning.

    Returns:
        The warning text, or None when the TSC is offered or the check does not apply.
    """
    if machine.lower() not in _X86_MACHINES or not available_clocksource.is_file():
        return None
    clocksources = available_clocksource.read_text(encoding="ascii").split()
    if "tsc" in clocksources:
        return None
    return (
        "the kernel does not offer the CPU time-stamp counter (TSC) as a clocksource "
        f"(it offers: {' '.join(clocksources)}), usually because the counters of the "
        "CPUs are out of step; torch.profiler uses the TSC, so event times can be "
        "wrong and the summary can fail. Run on CPUs whose counters agree (for "
        "example with taskset) or update the firmware."
    )
