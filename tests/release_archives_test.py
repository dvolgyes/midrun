"""Build both archive formats directly from a source tree with local runtime output."""

from pathlib import Path
import shutil
import subprocess
import sys


def test_dirty_source_excludes_run_data(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z", "midrun"], cwd=root)
        .decode()
        .split("\0")
    )
    for name in ["pyproject.toml", "LICENSE", "README.md", *filter(None, tracked)]:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    run = source / "midrun/run-release-probe"
    run.mkdir()
    (run / "midrun.log").write_text("synthetic log")
    (run / "capture.json").write_text("{}")
    (source / "midrun/midrun.log").write_text("synthetic log outside run directory")
    output = tmp_path / "archives"
    subprocess.run(
        ["uv", "build", "--sdist", "--wheel", "--out-dir", str(output)],
        cwd=source,
        check=True,
    )
    subprocess.run(
        [sys.executable, root / "tools/check_release.py", output],
        cwd=source,
        check=True,
    )
