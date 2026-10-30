"""The error for a gdb attach or call attempt that cannot be retried."""


class GdbAttachError(RuntimeError):
    """A gdb attach/call attempt failed for a non-retryable reason."""
