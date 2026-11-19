"""Cleanly close a PyTorch recorder that has not collected events yet."""

import torch


def close_unstarted_recorder[T](recorder: T) -> T | None:
    """Close a waiting Kineto recorder; return any recorder that can supply events."""
    if isinstance(recorder, torch.profiler.profile) and recorder.profiler is None:
        recorder.__exit__(None, None, None)  # type: ignore[no-untyped-call]
        return None
    return recorder
