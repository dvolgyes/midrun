"""Control-plane collectives that let every rank act on rank 0's Trainctl decisions."""

from typing import cast

import torch
import torch.distributed as dist


class RankSync:
    """Exchanges small Python objects between the ranks of a distributed run.

    Only rank 0 has the REST server and the command queue, so the other ranks learn
    its decisions through this class. It uses its own gloo process group, so its
    messages never mix with the training collectives, and it works on CPU tensors
    whatever the training backend is. The group has torch's default gloo timeout
    (30 minutes).

    Every rank must call each method in the same order, as with any collective. An
    instance without a group is a single-process one: every method returns its local
    result, so the callers have one code path for both cases.
    """

    _group: dist.ProcessGroup | None

    def __init__(self, *, group: dist.ProcessGroup | None = None) -> None:
        self._group = group

    @classmethod
    def create(cls) -> "RankSync":
        """Makes the control group when a multi-rank process group is initialized.

        This is a collective: every rank must call it, after Lightning's strategy has
        initialized `torch.distributed`.
        """
        if (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_world_size() == 1
        ):
            return cls()
        return cls(group=dist.new_group(backend="gloo"))

    @property
    def world_size(self) -> int:
        """Number of ranks."""
        return 1 if self._group is None else dist.get_world_size(self._group)

    def all_gather[T](self, obj: T) -> list[T]:
        """Returns every rank's `obj`, in rank order."""
        if self._group is None:
            return [obj]
        gathered: list[object] = [None] * self.world_size
        dist.all_gather_object(gathered, obj, group=self._group)
        return cast("list[T]", gathered)

    def broadcast[T](self, obj: T) -> T:
        """Returns rank 0's `obj` on every rank; the other ranks' `obj` is discarded.

        Rank 0 gets back its own object, not a copy.
        """
        if self._group is None:
            return obj
        message: list[object] = [obj]
        dist.broadcast_object_list(message, src=0, group=self._group)
        return obj if dist.get_rank(self._group) == 0 else cast("T", message[0])

    def any_rank(self, flag: bool) -> bool:
        """Returns whether `flag` is true on at least one rank."""
        if self._group is None:
            return flag
        value = torch.tensor([int(flag)])
        dist.all_reduce(value, op=dist.ReduceOp.MAX, group=self._group)
        return bool(value.item())
