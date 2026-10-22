"""Checks that a gdb binary is present and usable."""

import shutil
import subprocess


class GdbUnavailableError(RuntimeError):
    """No usable `gdb --interpreter=mi3` binary was found."""


def check_gdb_available(gdb_bin: str = "gdb") -> None:
    """Raises `GdbUnavailableError` with a clear message if `gdb_bin` is unusable.

    Checked upfront so a missing/broken gdb surfaces here, not as an opaque
    failure deep inside `pygdbmi`.
    """
    resolved = shutil.which(gdb_bin)
    if resolved is None:
        raise GdbUnavailableError(f"{gdb_bin!r} not found on PATH")
    probe = subprocess.run(  # noqa: S603 -- capability probe of an operator-specified gdb binary
        [resolved, "--version"], capture_output=True, text=True, timeout=5, check=False
    )
    if probe.returncode != 0:
        raise GdbUnavailableError(
            f"{resolved} --version failed: {probe.stderr.strip()}"
        )
