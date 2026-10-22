"""Tells whether torchinfo can be imported."""


def torchinfo_available() -> bool:
    """Returns whether `torchinfo` can be imported."""
    try:
        import torchinfo  # noqa: F401
    except ImportError:
        return False
    return True
