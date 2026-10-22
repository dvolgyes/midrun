"""The record of one tensor exported for a heavy hook."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorRecord:
    """One tensor exported for a heavy hook invocation.

    Attributes:
        argument_path: Slash-joined path locating this tensor within the callback's
            named tensor arguments, e.g. `batch/0/image` -- never derived from a
            dictionary key as a filename.
        file: The generated `.npy` filename (`tensor-000001.npy`, ...).
        shape: The tensor's shape.
        dtype: The numpy dtype the array was actually written as.
        original_dtype: The tensor's original torch dtype, as a string, only when it
            differs from `dtype` (currently only `bfloat16` -> `float32`); else `None`.
        device: The tensor's original device, as a string.
    """

    argument_path: str
    file: str
    shape: list[int]
    dtype: str
    original_dtype: str | None
    device: str
