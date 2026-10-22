"""Tells whether a Python module can be imported."""


def python_module_available(module_name: str) -> bool:
    """Returns whether importing `module_name` succeeds."""
    try:
        __import__(module_name)
    except ImportError:
        return False
    return True
