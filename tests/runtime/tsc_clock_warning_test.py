"""Behavioral tests for the warning about an unstable CPU time-stamp counter."""

from pathlib import Path

from midrun.runtime.tsc_clock_warning import tsc_clock_warning


def _clocksources(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "available_clocksource"
    path.write_text(text)
    return path


def test_x86_without_a_tsc_clocksource_gets_a_warning(tmp_path: Path) -> None:
    warning = tsc_clock_warning(
        "x86_64", available_clocksource=_clocksources(tmp_path, "hpet acpi_pm \n")
    )

    assert warning is not None
    assert "hpet acpi_pm" in warning
    assert "taskset" in warning


def test_x86_with_a_tsc_clocksource_gets_no_warning(tmp_path: Path) -> None:
    path = _clocksources(tmp_path, "tsc hpet acpi_pm \n")

    assert tsc_clock_warning("x86_64", available_clocksource=path) is None


def test_other_architectures_get_no_warning(tmp_path: Path) -> None:
    path = _clocksources(tmp_path, "arch_sys_counter \n")

    assert tsc_clock_warning("aarch64", available_clocksource=path) is None


def test_a_missing_clocksource_file_gets_no_warning(tmp_path: Path) -> None:
    assert (
        tsc_clock_warning("x86_64", available_clocksource=tmp_path / "absent") is None
    )
