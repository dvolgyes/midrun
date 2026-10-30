"""The error of a hooks tree bootstrap step that must not retry silently."""


class HooksBootstrapError(RuntimeError):
    """A baseline copy or live-tree bootstrap step failed in a way that must not retry silently."""
