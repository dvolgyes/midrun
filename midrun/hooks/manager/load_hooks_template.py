"""Reads a template file that is packaged with midrun.hooks."""

from importlib import resources


def load_hooks_template(name: str) -> str:
    """Returns the text of the template `name` in `midrun/hooks/templates/`."""
    return (resources.files("midrun.hooks") / "templates" / name).read_text()
