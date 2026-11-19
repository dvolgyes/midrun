#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Check that both distributions contain precisely the expected package files."""

import sys
import tarfile
import zipfile
from pathlib import Path


def check_archive(path: Path, expected: set[str]) -> None:
    """Reject missing and unexpected package members without extracting the archive."""
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as wheel:
            files = {entry.filename for entry in wheel.infolist() if not entry.is_dir()}
    else:
        with tarfile.open(path) as sdist:
            files = {
                entry.name.partition("/")[2] for entry in sdist if not entry.isdir()
            }
    package = {name for name in files if name.startswith("midrun/")}
    if package != expected:
        raise ValueError(
            f"{path}: missing {sorted(expected - package)}; unexpected {sorted(package - expected)}"
        )


def _main(directory: Path) -> None:
    assets = {
        "midrun/py.typed",
        "midrun/hooks/templates/README.md.tmpl",
        "midrun/hooks/templates/heavy.sh.tmpl",
        "midrun/hooks/templates/light.sh.tmpl",
        "midrun/rest/static/app.js",
        "midrun/rest/static/index.html",
        "midrun/rest/static/style.css",
    }
    (wheel,) = directory.glob("*.whl")
    (sdist,) = directory.glob("*.tar.gz")
    expected = assets | {
        p.as_posix()
        for p in Path("midrun").rglob("*.py")
        if not p.parts[1].startswith("run-")
    }
    for archive in (wheel, sdist):
        check_archive(archive, expected)


if __name__ == "__main__":
    _main(Path(sys.argv[1]))
