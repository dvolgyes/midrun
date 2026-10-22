"""Reads a template file that is packaged with trainctl.hooks."""

from importlib import resources


def load_hooks_template(name: str) -> str:
    """Returns the text of the template `name` in `trainctl/hooks/templates/`."""
    return (resources.files("trainctl.hooks") / "templates" / name).read_text()
