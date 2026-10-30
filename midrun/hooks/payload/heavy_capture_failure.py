"""The reason a heavy capture was refused."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HeavyCaptureFailure:
    """Why an entire heavy capture was refused before any tensor was copied off-device.

    Attributes:
        reason: A human-readable description naming the offending argument path or
            the exceeded limit.
    """

    reason: str
