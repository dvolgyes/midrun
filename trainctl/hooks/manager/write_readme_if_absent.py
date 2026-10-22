"""Writes the README of a hooks tree, unless the baseline has one."""

from pathlib import Path

from trainctl.hooks.catalog import CALLBACKS, CATALOGUE_VERSION, MODALITIES
from trainctl.hooks.manager.load_hooks_template import load_hooks_template


def write_readme_if_absent(staging_dir: Path) -> None:
    """Writes the hooks README into `staging_dir`, unless the baseline copied one."""
    readme_path = staging_dir / "README.md"
    if readme_path.exists():
        return
    template = load_hooks_template("README.md.tmpl")
    slot_count = len(CALLBACKS) * len(MODALITIES)
    content = (
        template.replace("@@CATALOGUE_VERSION@@", str(CATALOGUE_VERSION))
        .replace("@@CALLBACK_COUNT@@", str(len(CALLBACKS)))
        .replace("@@SLOT_COUNT@@", str(slot_count))
    )
    readme_path.write_text(content)
