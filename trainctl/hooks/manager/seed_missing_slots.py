"""Writes a disabled example script for each callback slot that the baseline does not fill."""

from pathlib import Path
from typing import Any

from trainctl.hooks.catalog import CALLBACKS, MODALITIES
from trainctl.hooks.manager.load_hooks_template import load_hooks_template

_SHEBANG = "#!/usr/bin/env bash\n"


def seed_missing_slots(staging_dir: Path, found_slots: set[tuple[str, str]]) -> None:
    """Writes a disabled (non-executable) example script for each slot not in `found_slots`."""
    templates = {
        modality: load_hooks_template(f"{modality}.sh.tmpl") for modality in MODALITIES
    }
    for spec in CALLBACKS:
        for modality in MODALITIES:
            if (modality, spec.name) in found_slots:
                continue
            script_path = staging_dir / modality / f"{spec.name}.sh"
            script_path.write_text(_SHEBANG + _render(templates[modality], spec))
            script_path.chmod(0o644)


def _render(template_text: str, spec: Any) -> str:
    return (
        template_text.replace("@@NAME@@", spec.name)
        .replace("@@GROUP@@", spec.group)
        .replace("@@DESCRIPTION@@", spec.description)
    )
