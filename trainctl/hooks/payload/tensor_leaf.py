"""One tensor found in the arguments of a heavy hook."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TensorLeaf:
    """One tensor found in the tensor-carrying arguments of a heavy hook.

    Attributes:
        path: Slash-joined path of the tensor in the arguments, e.g. `batch/0/image`.
        tensor: The live tensor; not copied.
    """

    path: str
    tensor: torch.Tensor
