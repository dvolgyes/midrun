"""Tells whether torchlens can be imported."""


def torchlens_available() -> bool:
    """Returns whether `torchlens` can be imported."""
    try:
        import torchlens  # noqa: F401
    except ImportError:
        return False
    return True
