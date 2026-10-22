"""Reads the trainer function that a Lightning trainer runs, as a hooks stage."""

from typing import Any


def trainer_stage_name(trainer: Any) -> str | None:
    """`trainer.state.fn`'s value (`fit`/`validate`/`test`/`predict`), or `None`."""
    fn = trainer.state.fn
    return fn.value if fn is not None else None
